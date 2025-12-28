import os
import time
import threading
import binascii
import gradio as gr
import markdown
from transformers import AutoTokenizer
from funasr_onnx import Paraformer
from chatlaw.configuration import config
from chatlaw.client.utils.utils_ms import (
    heartbeat_client_ms,
    stream_from_server_ms,
)
from chatlaw.client.utils.common_utils import (
    recv_exact,
    render_mathml_from_latex,
    connection_acknowledgement,
    speech_to_text
)
from chatlaw.dataloader import download_resources
from launcher import get_resources_path

alive = True
stop_event = threading.Event()


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
        Gradio 的核心回调函数，负责：
        1. 启动与服务器的心跳监控；
        2. 将用户输入封装为模型输入格式（numpy 张量）；
        3. 执行一次短连接握手验证服务器是否在线；
        4. 建立推理连接并通过流式协议持续接收模型输出；
        5. 将增量输出渲染为 Markdown + MathML，并逐步发送到前端；
        6. 处理推理中断（STOP）以及异常情况。

        本函数为一个 Python generator，每次 yield 会推动 Gradio 更新界面。

    Args:
        input_audio : 用户录入语音，将作为问询内容或 prompt。
        input_text (str): 用户在前端输入的自然语言文本。

    Inputs:
        - **input_text**: 用户输入内容。
        - 全局依赖：
            - **alive**: 控制心跳线程继续执行的标志。
            - **stop_event**: 用于推理中断的事件对象。
            - **tokenizer**: 用于生成模型输入的 tokenizer。
            - **heartbeat_client_ms**: 心跳线程函数。
            - **connection_acknowledgement**: 测试短连是否成功的函数。
            - **stream_from_server_ms**: 流式推理接收器。
            - **render_mathml_from_latex**: 将 Markdown 输出中的 LaTeX 转换为 MathML。
            - **markdown.markdown**: 渲染 Markdown。

    Outputs:
        作为一个生成器 (generator)：
            yield 两个值：(status_text, html_output)
            示例：
                - ("🟡 正在建立连接...", "")
                - ("⌛️ 语音处理中...", "")
                - ("⌛️ 知识库检索中...", "")
                - ("🟢 推理中...", "<html>渲染内容</html>")
                - ("🛑 推理已中断。", "<html>最终渲染</html>")
                - ("⚠️ 数据接收异常：xxx", "")
                - ("✅ 推理完成。", "<html>最终渲染</html>")

        这些值会逐步通过 Gradio 输出到界面。

    Raises:
        本函数不向外抛出异常。
        若在连接或推理过程中出现错误，将 yield `"⚠️ 数据接收异常：xxx"` 并结束函数。
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

    # 启动心跳线程
    threading.Thread(
        target=heartbeat_client_ms,
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

        # —— 构建模型输入（np tensor） ——
        messages = [{"role": "user", "content": input_text}]
        templated = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([templated], return_tensors="np")

        # —— 连接确认（短连接） ——
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

        # —— 流式接收 ——
        partial_md = ""

        for chunk in stream_from_server_ms(
            config.SERVER_IP,
            config.DATA_PORT,
            binascii.unhexlify(config.DATA_HANDSHAKE_REQ),
            binascii.unhexlify(config.DATA_HANDSHAKE_RESP),
            recv_exact,
            #alive_flag,      # ⚠️ 当前函数签名里没有这个参数，后面可以一起改
            stop_event,
            model_inputs
        ):
            if stop_event.is_set():
                rendered = markdown.markdown(
                    partial_md, extensions=["fenced_code", "tables"]
                )
                yield "🛑 推理已中断。", rendered
                break

            if chunk == "<END>":
                rendered = markdown.markdown(
                    partial_md, extensions=["fenced_code", "tables"]
                )
                html_math = render_mathml_from_latex(rendered)
                yield "✅ 推理完成。", html_math
                break

            partial_md += chunk
            rendered = markdown.markdown(
                partial_md, extensions=["fenced_code", "tables"]
            )
            html_math = render_mathml_from_latex(rendered)
            yield "🟢 推理中...", html_math

    except Exception as e:
        yield f"⚠️ 数据接收异常：{e}", ""

    finally:
        # global alive
        alive = False
        time.sleep(0.5)


def stop_fn():
    stop_event.set()
    return "🛑 已发送停止信号给服务器", ""


with gr.Blocks(
    title="Qwen 模型客户端（UI + 流式输出）",
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

    gr.Markdown("## 🔗 Qwen 模型客户端（MindNLP版）")
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
    demo.queue()
    demo.launch(inbrowser=True)
