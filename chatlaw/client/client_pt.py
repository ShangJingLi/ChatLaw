import os
import time
import threading
import binascii
import gradio as gr
import markdown
from transformers import AutoTokenizer
from funasr_onnx import Paraformer

from chatlaw.client.utils.common_utils import (
    recv_exact,
    render_mathml_from_latex,
    connection_acknowledgement,
    speech_to_text,
    load_vectorstore,
    load_exact_index,
    build_law_name_candidates,
    retrieve_laws,
    build_prompt
)
from chatlaw.configuration import config
from chatlaw.client.utils.utils_pt import (
    heartbeat_client,
    stream_from_server,
)
from chatlaw.dataloader import download_resources
from launcher import get_resources_path


# ========== 全局状态 ==========
alive = True
stop_event = threading.Event()


def alive_flag():
    return alive


# ========== 资源加载 ==========
resource_path = get_resources_path()

download_resources(resource_type="tokenizer")
tokenizer_path = os.path.join(resource_path, "tokenizer").replace("\\", "/")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

download_resources(resource_type="audio_model")
AUDIO_MODEL_DIR = os.path.join(resource_path, "audio_model").replace("\\", "/")
TARGET_SR = 16000
AUDIO_CACHE_DIR = os.path.join(get_resources_path(), "_asr_cache")
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

audio_model = Paraformer(
    AUDIO_MODEL_DIR,
    batch_size=1,
    quantize=True,
    device_id=-1
)

download_resources(resource_type="vectorstore")
vectorstore = load_vectorstore(os.path.join(get_resources_path(), "vectorstore"))
exact_index = load_exact_index()
law_name_candidates = build_law_name_candidates(exact_index)


def _extract_doc_fields(item):
    """
    统一提取检索结果字段，兼容两种输入：
    1. FAISS 返回的 LangChain Document
    2. 精确索引返回的 dict
    """
    # FAISS Document
    if hasattr(item, "metadata"):
        metadata = item.metadata or {}
        law_name = (
            metadata.get("law_name_raw")
            or metadata.get("law_name")
            or metadata.get("law_name_norm")
            or "未知法律"
        )
        article = metadata.get("article", "")
        content = metadata.get("content", "") or getattr(item, "page_content", "")
        return law_name, article, content

    # exact 索引 dict
    if isinstance(item, dict):
        law_name = (
            item.get("law_name_raw")
            or item.get("law_name")
            or item.get("law_name_norm")
            or "未知法律"
        )
        article = item.get("article", "")
        content = item.get("content", "")
        return law_name, article, content

    # fallback
    return "未知法律", "", str(item)


def format_retrieved_docs_html(docs):
    """
    将当前检索到的知识库条文整理成 HTML。
    兼容：
    - FAISS Document
    - exact index dict
    """
    if not docs:
        return "<div>未检索到相关法律条文。</div>"

    blocks = []
    for i, doc in enumerate(docs, 1):
        law_name, article, content = _extract_doc_fields(doc)

        blocks.append(
            f"### {i}. 《{law_name}》{article}\n\n{content}"
        )

    md_text = "\n\n---\n\n".join(blocks)
    rendered = markdown.markdown(md_text, extensions=["fenced_code", "tables"])
    html = render_mathml_from_latex(rendered)
    return f"<div>{html}</div>"


def prepare_request_fn(input_audio, input_text):
    """
    第一步：
    - 输入校验
    - 语音转文字
    - 知识库检索
    - 构造最终 prompt
    - 返回检索结果，并开放“查看知识库检索”按钮

    注意：
    这里只更新一次知识库窗口相关状态。
    后续流式推理不再碰它，避免窗口被刷新关闭。
    """
    retrieval_html = ""
    retrieval_box_update = gr.update(visible=False)
    btn_update = gr.update(interactive=False)
    output_update = ""
    final_prompt = ""
    retrieval_visible_state = False

    # ===== 语音 / 文本 二选一校验 =====
    has_audio = input_audio is not None
    has_text = input_text is not None and input_text.strip() != ""

    if not has_audio and not has_text:
        return (
            "⚠️ 请输入语音或文本！",
            output_update,
            retrieval_html,
            retrieval_box_update,
            btn_update,
            final_prompt,
            retrieval_visible_state,
        )

    if has_audio and has_text:
        return (
            "⚠️ 请勿同时输入语音和文本！",
            output_update,
            retrieval_html,
            retrieval_box_update,
            btn_update,
            final_prompt,
            retrieval_visible_state,
        )

    # ===== 语音输入先转文字 =====
    if has_audio:
        input_text = speech_to_text(
            audio=input_audio,
            audio_model=audio_model,
            audio_cache_dir=AUDIO_CACHE_DIR,
            target_sr=TARGET_SR
        )

    # ===== 知识库检索 =====
    docs = retrieve_laws(
        vectorstore=vectorstore,
        query=input_text,
        exact_index=exact_index,
        law_name_candidates=law_name_candidates
    )

    retrieval_html = format_retrieved_docs_html(docs)
    final_prompt = build_prompt(input_text, docs)

    # 检索完成后，按钮开放，但窗口默认仍隐藏
    return (
        "✅ 知识库检索完成。",
        output_update,
        retrieval_html,
        gr.update(visible=False),
        gr.update(interactive=True),
        final_prompt,
        False,
    )


def stream_inference_fn(final_prompt):
    """
    第二步：
    只负责流式推理。
    这里只更新：
    - status_box
    - output_box

    不再更新 retrieval_box / btn_view_kb / retrieval_visible_state，
    因此不会把已经点开的知识库窗口冲掉。
    """
    if not final_prompt:
        yield "⚠️ 没有可用的输入内容。", ""
        return

    global alive
    alive = True
    stop_event.clear()

    threading.Thread(
        target=heartbeat_client,
        args=(
            config.SERVER_IP,
            config.HEARTBEAT_PORT,
            config.HB_INTERVAL,
            config.HB_TIMEOUT,
            alive_flag,
            stop_event,
            recv_exact,
        ),
        daemon=True
    ).start()

    try:
        start_time = time.time()
        yield "🟡 正在建立连接...", ""

        # ---- 构造模型输入 ----
        messages = [{"role": "user", "content": final_prompt}]
        templated = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([templated], return_tensors="pt")

        # ---- 连接检测（短连接） ----
        detail, status = connection_acknowledgement(
            config.SERVER_IP,
            config.DATA_PORT,
            binascii.unhexlify(config.DATA_HANDSHAKE_REQ),
            binascii.unhexlify(config.DATA_HANDSHAKE_RESP),
            recv_exact,
            start_time
        )
        yield detail, ""

        if not status:
            alive = False
            return

        yield "✅ 已连接服务器，开始推理...", ""

        partial_md = ""

        for chunk in stream_from_server(
            config.SERVER_IP,
            config.DATA_PORT,
            binascii.unhexlify(config.DATA_HANDSHAKE_REQ),
            binascii.unhexlify(config.DATA_HANDSHAKE_RESP),
            recv_exact,
            alive_flag,
            stop_event,
            model_inputs
        ):
            if isinstance(chunk, str) and chunk.startswith("[ClientError]"):
                yield f"⚠️ 数据接收异常：{chunk}", ""
                break

            if chunk == "<END>":
                rendered = markdown.markdown(
                    partial_md, extensions=["fenced_code", "tables"]
                )
                html = render_mathml_from_latex(rendered)

                if stop_event.is_set():
                    yield "🛑 推理已中断。", f"<div>{html}</div>"
                else:
                    yield "✅ 推理完成。", f"<div>{html}</div>"
                break

            partial_md += chunk
            rendered = markdown.markdown(
                partial_md, extensions=["fenced_code", "tables"]
            )
            html = render_mathml_from_latex(rendered)

            if stop_event.is_set():
                yield "🟡 等待服务器停止推理...", f"<div>{html}</div>"
            else:
                yield "🟢 推理中...", f"<div>{html}</div>"

    except Exception as e:
        yield f"⚠️ 数据接收异常：{e}", ""
        return

    finally:
        alive = False
        time.sleep(0.5)


def toggle_retrieval_panel(visible_now):
    """
    点击展开 / 点击隐藏知识库检索结果。
    """
    new_visible = not bool(visible_now)
    return gr.update(visible=new_visible), new_visible


with gr.Blocks(
    title="ChatLaw · 智能法律咨询",
    css="""
    body {
        background-color: #f5f7fa;
    }

    .header {
        text-align: center;
        padding: 20px 0 10px 0;
    }

    .header h1 {
        color: #1f2937;
        font-size: 32px;
        margin-bottom: 5px;
    }

    .header p {
        color: #6b7280;
        font-size: 14px;
    }

    .disclaimer {
        background-color: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 8px;
        padding: 12px;
        font-size: 13px;
        color: #9a3412;
        margin-bottom: 15px;
    }

    .card {
        background-color: white;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }

    #model_output {
        border: none;
        border-radius: 10px;
        background-color: #ffffff;
        padding: 18px;
        height: 520px;
        overflow-y: auto;
        font-size: 15px;
        line-height: 1.7;
    }

    .status-box textarea {
        background-color: #f3f4f6 !important;
        font-size: 13px;
    }

    .btn-primary {
        background-color: #1e40af !important;
        color: white !important;
    }

    .btn-stop {
        background-color: #991b1b !important;
        color: white !important;
    }

    .btn-kb {
        background-color: #065f46 !important;
        color: white !important;
    }

    #retrieval_panel {
        margin-top: 12px;
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        padding: 14px;
        max-height: 260px;
        overflow-y: auto;
        font-size: 14px;
        line-height: 1.7;
    }

    .kb-tip {
        margin-top: 8px;
        font-size: 12px;
        color: #64748b;
    }
    """
) as demo:

    gr.HTML(
        """
        <div class="header">
            <h1>⚖️ ChatLaw</h1>
            <p>基于大模型的智能法律咨询助手</p>
        </div>
        """
    )

    gr.HTML(
        """
        <div class="disclaimer">
        ⚠️ <b>重要提示：</b>
        本系统提供的内容仅作为一般法律信息参考，不构成正式法律意见或律师建议。
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=4):
            with gr.Group():
                gr.Markdown("### 📝 咨询输入", elem_classes="card")

                audio_inp = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="🎙️ 中文语音输入（说完整一句）"
                )

                text_inp = gr.Textbox(
                    label="✍️ 文本输入",
                    lines=3,
                    placeholder="例如：小明因过失导致宿舍楼失火应承担什么法律责任？"
                )

                status_box = gr.Textbox(
                    label="📡 连接 / 推理状态",
                    interactive=False,
                    elem_classes="status-box"
                )

                with gr.Row():
                    btn_send = gr.Button("🚀 提交咨询", elem_classes="btn-primary")
                    btn_stop = gr.Button("🛑 停止推理", elem_classes="btn-stop")

                btn_view_kb = gr.Button(
                    "👁️ 查看知识库检索",
                    interactive=False,
                    elem_classes="btn-kb"
                )

                gr.HTML(
                    """
                    <div class="kb-tip">
                    点击按钮展开或隐藏本轮知识库检索到的法律条文。
                    </div>
                    """
                )

                retrieval_box = gr.HTML(
                    value="",
                    elem_id="retrieval_panel",
                    visible=False
                )

                retrieval_visible_state = gr.State(False)
                final_prompt_state = gr.State("")

        with gr.Column(scale=6):
            gr.Markdown("### 📚 分析与解答", elem_classes="card")

            output_box = gr.HTML(
                label="",
                elem_id="model_output"
            )

    # 第一段：准备请求（检索、构造 prompt、开放按钮）
    send_evt = btn_send.click(
        prepare_request_fn,
        inputs=[audio_inp, text_inp],
        outputs=[
            status_box,              # 状态
            output_box,              # 重置输出
            retrieval_box,           # 写入检索结果 HTML
            retrieval_box,           # 隐藏检索面板
            btn_view_kb,             # 开放 / 禁用按钮
            final_prompt_state,      # 保存最终 prompt
            retrieval_visible_state  # 重置面板展开状态
        ]
    )

    # 第二段：流式推理（只更新状态和输出，不碰知识库窗口）
    send_evt.then(
        stream_inference_fn,
        inputs=[final_prompt_state],
        outputs=[status_box, output_box]
    )

    btn_stop.click(
        lambda: stop_event.set(),
        None,
        None
    )

    btn_view_kb.click(
        toggle_retrieval_panel,
        inputs=[retrieval_visible_state],
        outputs=[retrieval_box, retrieval_visible_state]
    )


if __name__ == "__main__":
    demo.launch(inbrowser=True)