"""ChatLaw 客户端（纯前端）。

重构后客户端不再加载任何模型 / 知识库资源，只负责：
    1. 采集用户的语音或文本输入；
    2. 把原始输入发送给服务端；
    3. 显示服务端返回的「知识库检索结果」与「流式推理输出」。

语音转文字、知识库检索、prompt 构造、分词、推理全部在服务端完成。
"""
import os

# ===== 必须在 import gradio 之前执行 =====
# 本机 shell 常驻 ALL_PROXY=socks://... 之类的代理变量；gradio 依赖的 httpx 会在
# 被 import 时立即用这些变量构造全局 Client，而 socks 代理方案会让 `import gradio`
# 直接抛异常。客户端是本地 UI（连服务端走裸 socket，不经 HTTP 代理），因此这里清掉
# 当前进程的代理环境变量，保证 gradio 能正常导入。仅影响本进程，不改动用户 shell。
for _proxy_var in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy",
                   "HTTPS_PROXY", "https_proxy"):
    os.environ.pop(_proxy_var, None)

import time
import uuid
import threading
import binascii

import gradio as gr
import markdown

from chatlaw.client.utils.common_utils import recv_exact, render_mathml_from_latex
from chatlaw.client.utils.utils_pt import heartbeat_client, stream_consultation
from chatlaw.configuration import config


# ========== 每会话控制块（去全局化：多标签页 / 多次咨询互不干扰） ==========
class ControlBlock:
    """持有「当前这次咨询」的停止信号。

    用 gr.State 持有一个持久的 ControlBlock（每浏览器会话一个）；consult_fn 每次咨询
    把 stop_event **原地**换成一个新的 Event —— 因为 State 存的是同一个对象引用，停止
    按钮读到的始终是本次会话的 stop_event，从而不再依赖全局变量、不会误伤其他会话。
    """

    def __init__(self):
        self.stop_event = threading.Event()


def _new_control():
    """页面加载时为每个浏览器会话初始化一个持久 ControlBlock。"""
    return ControlBlock()


def stop_consultation(control):
    """停止按钮：只置位本会话的 stop_event。"""
    if control is not None:
        control.stop_event.set()


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


def consult_fn(input_audio, input_text, control):
    """
    单连接两阶段流程：
        1. 校验输入（语音 / 文本二选一）；
        2. 发送请求 -> 接收检索结果 -> 写入知识库面板（默认隐藏，按钮开放）；
        3. 接收流式推理输出 -> 增量渲染。

    inputs:  [audio, text, control(gr.State: ControlBlock)]
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

    # 本次咨询的独立会话状态（去全局化）：新 session_id + 新 stop_event。
    # stop_event 原地写入 control，供停止按钮读取到「本次」的 event。
    session_id = uuid.uuid4().hex
    stop_event = threading.Event()
    control.stop_event = stop_event
    alive = [True]

    def alive_flag():
        return alive[0]

    # 启动心跳线程（控制通道），先上报 session_id
    threading.Thread(
        target=heartbeat_client,
        args=(
            config.SERVER_IP,
            config.HEARTBEAT_PORT,
            session_id,
            config.HB_INTERVAL,
            config.HB_TIMEOUT,
            alive_flag,
            stop_event,
            recv_exact,
        ),
        daemon=True,
    ).start()

    # 构造请求：客户端只搬运原始音频 / 文本，并带上 session_id
    if has_audio:
        request = {"session_id": session_id, "kind": "audio", "audio": input_audio}
    else:
        request = {"session_id": session_id, "kind": "text", "text": input_text.strip()}

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

            elif kind == "reject":
                # 服务端拒绝（如 transformers 单请求模式繁忙）
                yield f"🚫 {evt[1]}", gr.update(), gr.update(), gr.update()
                break

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
        alive[0] = False
        time.sleep(0.3)


def toggle_retrieval_panel(visible_now):
    """点击展开 / 隐藏知识库检索结果。"""
    new_visible = not bool(visible_now)
    return gr.update(visible=new_visible), new_visible


# Gradio 6.0 起，css / theme 等参数从 Blocks() 迁移到 launch()，
# 因此这里把自定义样式抽成常量，最终在 demo.launch(css=CUSTOM_CSS) 传入。
CUSTOM_CSS = """
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

    /* Gradio 6 会给 .block 加内联样式 border-style:solid / overflow:visible，
       优先级高于外部 CSS，必须用 !important 覆盖，否则出现黑框 / 无法滚动。 */
    #model_output {
        border: none !important;
        border-radius: 10px;
        background-color: #ffffff;
        padding: 18px;
        height: 520px;
        overflow-y: auto !important;
        font-size: 15px;
        line-height: 1.7;
    }

    /* 语音输入组件：去掉 Gradio 6 默认的黑色 .block 边框，改为与其它输入一致的浅灰边 */
    #audio_input {
        border: 1px solid #e5e7eb !important;
        border-radius: 8px;
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
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px;
        padding: 14px;
        max-height: 260px;
        overflow-y: auto !important;
        font-size: 14px;
        line-height: 1.7;
    }

    .kb-tip {
        margin-top: 8px;
        font-size: 12px;
        color: #64748b;
    }
    """

with gr.Blocks(title="ChatLaw · 智能法律咨询") as demo:

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
                    label="🎙️ 中文语音输入（说完整一句）",
                    elem_id="audio_input"
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
                # 每浏览器会话一个持久 ControlBlock（去全局化的停止信号载体）
                control_state = gr.State()

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
        inputs=[audio_inp, text_inp, control_state],
        outputs=[status_box, output_box, retrieval_box, btn_view_kb]
    )

    btn_stop.click(
        stop_consultation,
        inputs=[control_state],
        outputs=None
    )

    btn_view_kb.click(
        toggle_retrieval_panel,
        inputs=[retrieval_visible_state],
        outputs=[retrieval_box, retrieval_visible_state]
    )

    # 页面加载时为本浏览器会话初始化 ControlBlock
    demo.load(_new_control, inputs=None, outputs=[control_state])


if __name__ == "__main__":
    demo.launch(css=CUSTOM_CSS, inbrowser=True)
