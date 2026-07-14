"""ChatLaw 客户端（纯前端）。

重构后客户端不再加载任何模型 / 知识库资源，只负责：
    1. 采集用户的语音或文本输入；
    2. 把原始输入发送给服务端；
    3. 显示服务端返回的「知识库检索结果」与「流式推理输出」。

语音转文字、知识库检索、prompt 构造、分词、推理全部在服务端完成。
"""
import time
import threading
import binascii

import gradio as gr
import markdown

from chatlaw.client.utils.common_utils import recv_exact, render_mathml_from_latex
from chatlaw.client.utils.utils_pt import heartbeat_client, stream_consultation
from chatlaw.configuration import config


# ========== 全局状态 ==========
alive = True
stop_event = threading.Event()


def alive_flag():
    return alive


# ========== 展示辅助（纯前端） ==========
def _extract_doc_fields(item):
    """
    统一提取检索结果字段。服务端已把结果归一化为 dict：
        {"law_name": ..., "article": ..., "content": ...}
    这里同时兼容旧的 FAISS Document，保持健壮。
    """
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

    return "未知法律", "", str(item)


def format_retrieved_docs_html(docs):
    """把服务端返回的检索结果整理成 HTML。"""
    if not docs:
        return "<div>未检索到相关法律条文。</div>"

    blocks = []
    for i, doc in enumerate(docs, 1):
        law_name, article, content = _extract_doc_fields(doc)
        blocks.append(f"### {i}. 《{law_name}》{article}\n\n{content}")

    md_text = "\n\n---\n\n".join(blocks)
    rendered = markdown.markdown(md_text, extensions=["fenced_code", "tables"])
    html = render_mathml_from_latex(rendered)
    return f"<div>{html}</div>"


# ========== 业务流程 ==========
def reset_ui():
    """点击提交时先清空上一轮的界面状态。"""
    return (
        "",                                   # status_box
        "",                                   # output_box
        gr.update(value="", visible=False),   # retrieval_box
        gr.update(interactive=False),         # btn_view_kb
        False,                                # retrieval_visible_state
    )


def consult_fn(input_audio, input_text):
    """
    单连接两阶段流程：
        1. 校验输入（语音 / 文本二选一）；
        2. 发送请求 -> 接收检索结果 -> 写入知识库面板（默认隐藏，按钮开放）；
        3. 接收流式推理输出 -> 增量渲染。

    outputs: [status_box, output_box, retrieval_box, btn_view_kb]
    """
    has_audio = input_audio is not None
    has_text = input_text is not None and input_text.strip() != ""

    if not has_audio and not has_text:
        yield "⚠️ 请输入语音或文本！", gr.update(), gr.update(), gr.update()
        return
    if has_audio and has_text:
        yield "⚠️ 请勿同时输入语音和文本！", gr.update(), gr.update(), gr.update()
        return

    global alive
    alive = True
    stop_event.clear()

    # 启动心跳线程（控制通道）
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
        daemon=True,
    ).start()

    # 构造请求：客户端只搬运原始音频 / 文本，不做任何模型处理
    if has_audio:
        request = {"kind": "audio", "audio": input_audio}  # (sample_rate, ndarray)
    else:
        request = {"kind": "text", "text": input_text.strip()}

    yield "🟡 正在连接服务器...", gr.update(), gr.update(), gr.update()

    partial_md = ""
    try:
        for evt in stream_consultation(
            config.SERVER_IP,
            config.DATA_PORT,
            binascii.unhexlify(config.DATA_HANDSHAKE_REQ),
            binascii.unhexlify(config.DATA_HANDSHAKE_RESP),
            recv_exact,
            alive_flag,
            stop_event,
            request,
        ):
            kind = evt[0]

            if kind == "retrieval":
                query, docs = evt[1], evt[2]
                retrieval_html = format_retrieved_docs_html(docs)
                if query:
                    status = f"✅ 已识别：{query}｜知识库检索完成，开始推理..."
                else:
                    status = "✅ 知识库检索完成，开始推理..."
                yield (
                    status,
                    gr.update(),
                    gr.update(value=retrieval_html),  # 写入内容，保持隐藏，等用户点开
                    gr.update(interactive=True),      # 开放“查看知识库检索”
                )

            elif kind == "chunk":
                partial_md += evt[1]
                rendered = markdown.markdown(
                    partial_md, extensions=["fenced_code", "tables"]
                )
                html = render_mathml_from_latex(rendered)
                status = (
                    "🟡 等待服务器停止推理..."
                    if stop_event.is_set()
                    else "🟢 推理中..."
                )
                yield status, gr.update(value=f"<div>{html}</div>"), gr.update(), gr.update()

            elif kind == "error":
                yield f"⚠️ 数据接收异常：{evt[1]}", gr.update(), gr.update(), gr.update()

            elif kind == "end":
                rendered = markdown.markdown(
                    partial_md, extensions=["fenced_code", "tables"]
                )
                html = render_mathml_from_latex(rendered)
                final_status = "🛑 推理已中断。" if stop_event.is_set() else "✅ 推理完成。"
                yield final_status, gr.update(value=f"<div>{html}</div>"), gr.update(), gr.update()
                break

    except Exception as e:  # pylint: disable=broad-exception-caught
        yield f"⚠️ 数据接收异常：{e}", gr.update(), gr.update(), gr.update()

    finally:
        alive = False
        time.sleep(0.3)


def toggle_retrieval_panel(visible_now):
    """点击展开 / 隐藏知识库检索结果。"""
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

        with gr.Column(scale=6):
            gr.Markdown("### 📚 分析与解答", elem_classes="card")

            output_box = gr.HTML(
                label="",
                elem_id="model_output"
            )

    # 第一段：清空上一轮界面状态
    send_evt = btn_send.click(
        reset_ui,
        inputs=None,
        outputs=[
            status_box,
            output_box,
            retrieval_box,
            btn_view_kb,
            retrieval_visible_state,
        ]
    )

    # 第二段：单连接两阶段流程（检索结果 + 流式推理）
    send_evt.then(
        consult_fn,
        inputs=[audio_inp, text_inp],
        outputs=[status_box, output_box, retrieval_box, btn_view_kb]
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
