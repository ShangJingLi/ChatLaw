import os
import time
import threading
import binascii
import gradio as gr
import markdown
from transformers import AutoTokenizer
from chatlaw.client.utils.common_utils import (recv_exact,
                                               render_mathml_from_latex,
                                               connection_acknowledgement)
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


# ========== Tokenizer 准备 ==========
resource_path = get_resources_path()
model_name = "Qwen/Qwen3-4B-Instruct-2507"

if not os.path.exists(os.path.join(resource_path, "tokenizer")):
    download_resources(resource_type="tokenizer")

tokenizer_path = os.path.join(resource_path, "tokenizer").replace("\\", "/")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)


# ============================
# Gradio 回调函数
# ============================
def gradio_interface_fn(input_text):
    """
    每次点击“发送”都会进入一次生成器序列。
    """
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

    finally:
        alive = False
        time.sleep(0.5)  # 给心跳线程一点时间退出


# ============================
# 停止按钮
# ============================
def stop_fn():
    stop_event.set()
    return "🛑 已发送停止信号到服务器", ""


# ============================
# Gradio UI
# ============================
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
