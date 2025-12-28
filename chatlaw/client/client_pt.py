import os
import time
import threading
import binascii
import gradio as gr
import markdown
from transformers import AutoTokenizer
from funasr_onnx import Paraformer
from chatlaw.client.utils.common_utils import (recv_exact,
                                               render_mathml_from_latex,
                                               connection_acknowledgement,
                                               speech_to_text)
from chatlaw.configuration import config
from chatlaw.client.utils.utils_pt import (
    heartbeat_client,
    stream_from_server,
)
from chatlaw.dataloader import download_resources
from launcher import get_resources_path


# ========== 全局状态 ==========
alive = True                 # 数据流是否继续
stop_event = threading.Event()  # STOP 信号

def alive_flag():
    return alive

resource_path = get_resources_path()
download_resources(resource_type="tokenizer")
tokenizer_path = os.path.join(resource_path, "tokenizer").replace("\\", "/")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

download_resources(resource_type="audio_model")
AUDIO_MODEL_DIR = os.path.join(resource_path, "audio_model").replace("\\", "/")
TARGET_SR = 16000
AUDIO_CACHE_DIR = os.path.join(get_resources_path(), "_asr_cache")  # 语音临时文件目录
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)
audio_model = Paraformer(
    AUDIO_MODEL_DIR,
    batch_size=1,
    quantize=True,   # 使用 model_quant.onnx
    device_id=-1     # CPU-only
)


def gradio_interface_fn(input_audio, input_text):
    """
    功能：
        Gradio 的核心回调生成器函数，每次用户点击“发送”按钮都会触发一次新的推理流程。
        本函数负责：
        1. 启动心跳线程，确保服务器连接存活；
        2. 将用户输入封装为模型可处理的 prompt；
        3. 通过一次短连接检测服务器是否可用；
        4. 与服务器建立数据连接并进行流式推理接收；
        5. 在推理过程中持续向前端 UI 发送增量渲染结果；
        6. 在 STOP 中断、错误、或服务器返回 <END> 时进行收尾处理。

        该函数是一个 Python generator，每一次 `yield` 都会促使 Gradio 立即更新界面，
        用于实现实时流式输出效果。

    Args:
        input_audio : 用户录入语音，将作为问询内容或 prompt。
        input_text (str): 用户输入的自然语言文本，将作为问询内容或 prompt。

    Inputs:
        - **input_text**: 前端用户输入的文本内容。
        - **input_audio**: 前端用户输入的语音内容。
        - 全局依赖：
            - **alive** (bool): 控制心跳线程继续运行的标志。
            - **stop_event** (Event): 前端用于停止推理的事件信号。
            - **tokenizer**: 构造模型输入的 tokenizer。
            - **heartbeat_client**: 心跳线程函数，用于维护与服务器的存活性检测。
            - **connection_acknowledgement**: 用于短连接测试服务器是否在线。
            - **stream_from_server**: 流式推理数据接收器。
            - **render_mathml_from_latex**: 将 Markdown 中的公式转换为 MathML。
            - **markdown.markdown**: 渲染 Markdown 文本。

    Outputs:
        作为一个生成器（generator），本函数多次 yield：
            (状态文本, HTML渲染内容)
        示例：
            - ("🟡 正在建立连接...", "")
            - ("⌛️ 语音处理中...", "")
            - ("⌛️ 知识库检索中...", "")
            - ("🟢 推理中...", "<html>渲染内容</html>")
            - ("🛑 推理已中断。", "<html>最终渲染</html>")
            - ("⚠️ 数据接收异常：xxx", "")
            - ("✅ 推理完成。", "<html>最终渲染</html>")

        这些值将被 Gradio 自动逐段渲染到 UI 中，实现实时输出体验。

    Raises:
        本函数不向外抛出异常。
        所有连接异常、推理异常等均以 yield 的形式返回给前端，
        格式为："⚠️ 数据接收异常：xxx"。
    """
    # ===== 语音 / 文本 二选一校验 =====
    has_audio = input_audio is not None
    has_text = input_text is not None and input_text.strip() != ""

    if not has_audio and not has_text:
        yield "⚠️ 请输入语音或文本！", ""
        return

    if has_audio and has_text:
        yield "⚠️ 请勿同时输入语音和文本！", ""
        return

    # 只有语音输入：先做 ASR
    if has_audio:
        yield "⌛️ 语音处理中...", ""
        input_text = speech_to_text(
            audio=input_audio,
            audio_model=audio_model,
            audio_cache_dir=AUDIO_CACHE_DIR,
            target_sr=TARGET_SR
        )
    # 只有文本输入：直接使用 input_text

    global alive
    alive = True
    stop_event.clear()

    # 启动心跳
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
        messages = [{"role": "user", "content": input_text}]
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

        # ========== 流式接收 ==========
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
            # ---- 错误情况 ----
            if isinstance(chunk, str) and chunk.startswith("[ClientError]"):
                yield f"⚠️ 数据接收异常：{chunk}", ""
                break

            # ---- 服务端结束 ----
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

            # ---- 增量生成 ----
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
        time.sleep(0.5)  # 给心跳线程一点时间退出


def stop_fn():
    stop_event.set()
    return "🛑 已发送停止信号到服务器", ""


with gr.Blocks(
    title="Qwen 模型客户端（Transformers 版）",
    css="""
        #model_output {
          border: 2px solid #ccc;
          border-radius: 10px;
          background-color: #fff;
          padding: 15px;
          box-shadow: 0 3px 10px rgba(0,0,0,0.1);
          height: 500px;
          overflow-y: auto;
        }
    """
) as demo:

    gr.Markdown("## 🔗 Qwen 模型客户端（Transformers 版）")

    audio_inp = gr.Audio(
        sources=["microphone"],
        type="numpy",
        label="中文语音输入（请说完整一句话）"
    )
    text_inp = gr.Textbox(label="输入文本", lines=2, placeholder="请输入内容...")
    status_box = gr.Textbox(label="连接与状态信息", interactive=False)

    with gr.Row():
        btn_send = gr.Button("🚀 发送到服务器")
        btn_stop = gr.Button("🛑 停止推理")

    output_box = gr.HTML(label="模型输出", elem_id="model_output")

    btn_send.click(gradio_interface_fn, inputs=[audio_inp, text_inp], outputs=[status_box, output_box])
    btn_stop.click(stop_fn, inputs=None, outputs=[status_box, output_box])


if __name__ == "__main__":
    demo.launch(inbrowser=True)
