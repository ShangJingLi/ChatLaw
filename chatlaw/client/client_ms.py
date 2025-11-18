import time
import threading
import binascii

import gradio as gr
import markdown
from transformers import AutoTokenizer

from chatlaw.configuration import config

# MindNLP 工具函数
from chatlaw.client.utils.utils_ms import (
    heartbeat_client_ms,
    stream_from_server_ms,
)

from chatlaw.client.utils.common_utils import (
    recv_exact,
    render_mathml_from_latex,
    connection_acknowledgement,
)

# ============================
# 全局状态（供 utils_ms 调用）
# ============================
alive = True
stop_event = threading.Event()


def alive_flag():
    return alive


# ============================
# 加载 tokenizer
# ============================
model_name = "Qwen/Qwen3-4B-Instruct-2507"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# ============================
# Gradio 回调函数
# ============================
def gradio_interface_fn(input_text):
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


# ============================
# 停止回调
# ============================
def stop_fn():
    stop_event.set()
    return "🛑 已发送停止信号给服务器", ""


# ============================
# Gradio UI
# ============================
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

    inp = gr.Textbox(label="输入文本", lines=2, placeholder="请输入内容...")
    status_box = gr.Textbox(label="连接与状态信息", interactive=False)

    with gr.Row():
        btn_send = gr.Button("🚀 发送到服务器")
        btn_stop = gr.Button("🛑 停止推理")

    output_box = gr.HTML(label="模型输出", elem_id="model_output")

    btn_send.click(gradio_interface_fn, inputs=inp, outputs=[status_box, output_box])
    btn_stop.click(stop_fn, inputs=None, outputs=[status_box, output_box])


if __name__ == "__main__":
    demo.queue()
    demo.launch()
