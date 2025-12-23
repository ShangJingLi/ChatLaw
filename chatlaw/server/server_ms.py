import os
import socket
import subprocess
import threading
import time
import pickle
import queue
import mindspore as ms
from mindspore import context
import numpy as np
from mindnlp.transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    StoppingCriteriaList,
    StoppingCriteria,
)
from chatlaw.configuration import config
from chatlaw.dataloader import download_resources
from launcher import get_resources_path

device_map = "auto"
USE_ORANGE_PI = False

try:
    user = subprocess.run(["whoami"], capture_output=True, text=True).stdout.strip()
    context.set_context(mode=context.GRAPH_MODE)

    if user == "HwHiAiUser":
        context.set_context(jit_config={"jit_level": "O2"})
        ms.set_device("Ascend")
        device_map = "npu"
        USE_ORANGE_PI = True
    else:
        # 尝试普通 Ascend
        ms.set_device("Ascend")
        device_map = "npu"

except Exception:
    # Ascend 不可用，切换到 GPU
    try:
        context.set_context(mode=context.GRAPH_MODE)
        ms.set_device("GPU")
        device_map = "cuda"
    except Exception:
        raise RuntimeError("No compatible backend (Ascend/GPU) Found")

HEARTBEAT_PORT = config.HEARTBEAT_PORT
DATA_PORT = config.DATA_PORT

stop_flag = threading.Event()
last_heartbeat_time = time.time()
alive = True

print("[Model] Loading Qwen model...")
resource_path  = get_resources_path()
download_resources(resource_type="tokenizer")
download_resources(resource_type="llm")

tokenizer_path = os.path.join(resource_path, "tokenizer").replace("\\", "/")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
model_path = os.path.join(resource_path, "llm").replace("\\", "/")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    dtype="auto",
    device_map=device_map
)
print("[Model] Qwen loaded successfully.")

task_queue = queue.Queue()
result_queue = queue.Queue()


def recv_exact(conn, n, timeout=None):
    """
    功能：
        从指定连接中精确读取 `n` 字节的数据。若未能在读取过程中获取足够字节数，
        或对端提前关闭连接，则抛出异常。该函数保证返回的数据长度严格等于 `n`。

    Args:
        conn: 网络连接对象，需提供 `recv()` 方法用于接收字节数据。
        n (int): 期望接收的字节数。
        timeout (float, None): 接收数据的超时时间（秒）。若为 ``None`` 则不设置超时。默认： ``None`` 。

    Inputs:
        - **conn**: 提供 `recv` 方法的连接实例，通常为 socket 连接。
        - **n** (int): 需要从连接中完整读取的字节数。
        - **timeout** (float, None): 当设置超时时间时，若超时未接收到数据，将由底层抛出超时异常。

    Outputs:
        bytes: 精确长度为 ``n`` 的字节序列。如果连接在读取过程中被关闭，将抛出异常。

    Raises:
        ConnectionError: 当对端在未传输足够数据的情况下关闭连接时抛出该异常。
    """
    if timeout:
        conn.settimeout(timeout)
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return buf


class StopNow(StoppingCriteria):
    """
    功能：
        自定义推理停止条件，用于在 HuggingFace 文本生成过程中动态终止生成。
        当外部的事件对象（Event）被设置为触发状态时，本停止条件返回 True，
        从而令模型立即停止继续生成后续 token，适配服务器端的即时中断需求。

    Args:
        stop_event (Event): 外部传入的线程事件对象。当调用 `stop_event.set()` 时，
                            模型生成过程将被立即终止。

    Inputs:
        - **input_ids** (Tensor): 当前生成序列的 token IDs，由生成器内部自动传入。
        - **scores** (Tensor): 当前步对应的模型分数，同样由生成过程提供。
        - **kwargs**: 生成框架可能传入的额外参数，通常无需使用。

    Outputs:
        bool: 若 `stop_event` 处于已触发状态，则返回 ``True``，表示应立即停止生成；
              否则返回 ``False``，模型继续生成下一步 token。

    Raises:
        本类不主动抛出异常。
    """
    def __init__(self, stop_event):
        super().__init__()
        self.stop_event = stop_event

    def __call__(self, input_ids, scores, **kwargs):
        return self.stop_event.is_set()


def heartbeat_server():
    """
    功能：
        启动心跳检测服务器，持续监听指定心跳端口，用于监控客户端连接是否存活。
        服务器与客户端约定采用简单的心跳协议：客户端发送 `PING`，服务器返回 `PONG`，
        并更新最近心跳时间。若超过指定时间未收到心跳包，则认为客户端连接已丢失。
        若接收到 `STOP` 指令，则触发停止标志 `stop_flag`，用于通知其他线程中断运行。
        本函数持续循环工作，每当会话结束后重新等待下一个客户端连接。

    Inputs:
        本函数没有外部参数，但依赖以下全局变量与常量：
        - **HEARTBEAT_PORT** (int): 心跳服务器监听的端口号。
        - **alive** (bool): 全局标志，指示当前连接是否处于存活状态。
        - **last_heartbeat_time** (float): 最近一次收到心跳包的时间戳。
        - **stop_flag** (Event): 外部事件对象，用于通知系统停止当前推理会话。

    Outputs:
        无显式返回值。本函数以循环方式持续运行，会话结束后自动等待下一次连接。

    Raises:
        异常不会向外抛出。本函数使用 try/except 捕获所有异常，
        如发生错误会打印日志并在短暂延时后自动重试。
    """
    global last_heartbeat_time, alive
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('0.0.0.0', HEARTBEAT_PORT))
            s.listen(1)
            print(f"[Heartbeat] Listening on {HEARTBEAT_PORT}")

            conn, addr = s.accept()
            print(f"[Heartbeat] Connected by {addr}")

            conn.settimeout(2)
            alive = True
            last_heartbeat_time = time.time()

            while True:
                try:
                    msg = conn.recv(16)
                    if not msg:
                        break
                    if msg == b"PING":
                        conn.sendall(b"PONG")
                        last_heartbeat_time = time.time()
                    elif msg == b"STOP":
                        print("[Heartbeat] STOP received")
                        stop_flag.set()
                        break

                except socket.timeout:
                    if time.time() - last_heartbeat_time > 3:
                        print("[Heartbeat] LOST")
                        break

            conn.close()
            s.close()
            print("[Heartbeat] Session closed")

        except Exception as e:
            print(f"[Heartbeat] ERR: {e}")
            time.sleep(1)


def data_server():
    """
    功能：
        启动主数据服务器，监听指定端口以接受客户端的数据连接请求。
        每当有新的客户端连接到达时，服务器会为该连接创建独立的处理线程，
        并交由 `handle_client` 函数处理其推理请求。
        本函数不会阻塞于任何单一客户端，而是持续循环以接受更多连接。

    Inputs:
        虽然函数本身无参数输入，但依赖以下外部变量与资源：
        - **DATA_PORT** (int): 数据服务的监听端口。
        - **handle_client** (callable): 用于处理单个客户端连接的函数。

    Outputs:
        无显式返回值。该服务器长期运行，根据客户端连接情况持续创建会话线程。

    Raises:
        本函数内部未包含显式异常捕获，因此任何异常将由上层调用者处理。
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", DATA_PORT))
    s.listen(5)
    print(f"[Data] Listening on {DATA_PORT}")

    while True:
        conn, addr = s.accept()
        print(f"[Data] Connected by {addr}")
        threading.Thread(target=handle_client, args=(conn,), daemon=True).start()


def handle_client(conn):
    """
    功能：
        处理单个客户端的任务提交请求。函数首先执行固定格式的一字节握手，
        确保客户端合法接入。随后从连接中读取请求头与请求体，解析得到模型输入，
        并将任务提交到全局任务队列 `task_queue`，供后台推理线程处理。
        本函数不负责推理，仅负责接收与排队任务。

    Args:
        conn: 与客户端建立的 socket 连接对象，用于接收数据与发送握手确认。

    Inputs:
        - **conn**: 必须提供 `recv()`、`sendall()`、`close()` 等方法的 socket 连接。
        - 函数内部依赖以下外部变量/对象：
            - **recv_exact** (callable): 精确读取指定字节数的辅助函数。
            - **task_queue** (queue.Queue): 用于存放待推理任务的全局任务队列。
            - **pickle**: 用于对模型输入进行序列化/反序列化。

    Outputs:
        无显式返回值。任务会被放入 `task_queue` 中等待后端处理。

    Raises:
        函数内部捕获所有异常并打印错误，不会向外抛出异常。
        若出现异常，会尝试安全关闭客户端连接。
    """
    try:
        hb = recv_exact(conn, 1, timeout=5)
        if hb != b'\xA5':
            raise ValueError("Handshake failed")
        conn.sendall(b'\x5A')

        header = recv_exact(conn, 8, timeout=30)
        n = int.from_bytes(header, 'big')
        body = recv_exact(conn, n, timeout=30)
        model_inputs = pickle.loads(body)

        task_queue.put((conn, model_inputs))
        print("[Data] Task queued")

    except Exception as e:
        print("[Data] ERR:", e)
        try:
            conn.close()
        except:
            pass


def safe_generate(model, generation_kwargs):
    """
    功能：
        对模型的生成过程进行安全封装。该函数以关键字参数字典的方式调用
        `model.generate()`，并在调用期间捕获可能出现的异常。
        若生成过程中发生错误，会打印错误日志并重新抛出异常，以便调用方
        在上层逻辑中继续处理错误。该封装常用于线程环境中，确保异常不会被隐藏。

    Args:
        model: 具有 `generate()` 方法的模型实例，用于执行文本生成。
        generation_kwargs (dict): 传递给 `model.generate()` 的关键字参数字典。
                                  其中通常包含：
                                  - `input_ids`: 输入 token 张量
                                  - `attention_mask`: 注意力掩码
                                  - `streamer`: 文本流式输出器
                                  - `stopping_criteria`: 停止条件（例如 StopFlagCriteria）
                                  - 以及其他 HuggingFace 生成参数

    Inputs:
        - **model**: 提供文本生成能力的模型对象。
        - **generation_kwargs** (dict): 用于 `model.generate()` 的参数。
          若需支持流式输出与外部停止信号，则相关组件应在此字典中配置。

    Outputs:
        无显式返回值。生成过程的输出由 streamer 或 generate 内部机制处理。

    Raises:
        Exception: 若模型生成过程中发生异常，则打印错误信息并将异常重新抛出。
    """
    try:
        model.generate(**generation_kwargs)
    except Exception as e:
        print("[Generate] ERROR:", e)
        raise


def execute():
    """
    功能：
        从全局任务队列 `task_queue` 中持续获取推理任务，完成以下步骤：
        1. 将客户端发送的 numpy 数组转换为 MindSpore Tensor；
        2. 为推理创建 streamer（流式输出器）和停止条件；
        3. 启动线程以 safe_generate() 调用模型 generate() 执行推理；
        4. 持续监听 streamer 输出的新文本，将增量内容写入 result_queue；
        5. 在推理结束、异常或 STOP 标志触发时终止任务并发送 "<END>"。

    Inputs:
        本函数无外部参数，但依赖以下全局对象：
        - **task_queue** (queue.Queue): 客户端提交任务的队列，每项为 (conn, model_inputs)。
        - **result_queue** (queue.Queue): 用于向客户端发送流式推理输出。
        - **model**: Qwen 模型实例，需提供 generate() 方法。
        - **tokenizer**: 模型配套的 tokenizer，用于 streamer。
        - **stop_flag** (Event): 可由心跳线程或客户端触发，用于停止生成。
        - **StopNow**: 作为停止条件使用的 StoppingCriteria 类。

    Outputs:
        无显式返回值。
        本函数会不断向 result_queue 写入：
            - (conn, text)：模型生成的增量文本；
            - (conn, "<END>")：表示该连接的本次推理任务已结束。

    Raises:
        不会向外抛出异常。
        若任务处理过程中出现错误，会将错误字符串放入 result_queue。
    """
    while True:
        conn, model_inputs = task_queue.get()

        try:
            # ---- Tensor 转换 ----
            fixed = {}
            for k, v in model_inputs.items():
                if isinstance(v, np.ndarray):
                    v = ms.Tensor(v.astype(np.int32))
                fixed[k] = v.to(model.device)

            # ---- Streamer ----
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)

            # ---- 停止条件 ----
            stopping = StoppingCriteriaList([StopNow(stop_flag)])

            # ---- 生成参数 ----
            if USE_ORANGE_PI:
                gen_kwargs = dict(
                    **fixed,
                    streamer=streamer,
                    max_new_tokens=4096,
                    temperature=0.7,
                    do_sample=True,
                    stopping_criteria=stopping,
                )
            else:
                gen_kwargs = dict(
                    **fixed,
                    streamer=streamer,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    stopping_criteria=stopping,
                )
            # ---- generate 线程 ----
            gen_thread = threading.Thread(
                target=safe_generate,       # 已适配新版本签名
                args=(model, gen_kwargs),   # 只传 model 和 kwargs
                daemon=True,
            )
            gen_thread.start()

            # ---- streamer 输出处理 ----
            for text in streamer:
                if stop_flag.is_set():
                    print("[Execute] STOP handled safely.")
                    break
                result_queue.put((conn, text))

            gen_thread.join(timeout=1)

        except Exception as e:
            result_queue.put((conn, f"[Error] {str(e)}"))

        finally:
            stop_flag.clear()  # 状态复位
            result_queue.put((conn, "<END>"))
            print("[Execute] Job done → <END>")


def result_dispatcher():
    """
    功能：
        从全局 result_queue 中持续获取模型生成的结果片段，并将这些片段按协议
        推送给对应的客户端连接。数据发送格式为：
            1. 先发送 8 字节长度头（大端序）
            2. 再发送序列化后的内容 data
        若收到内容为 "<END>"，则表示该连接的本次推理会话已经结束，
        函数会关闭该客户端连接并加入 closed 集合，避免后续重复发送。

    Inputs:
        本函数无外部参数，但依赖以下全局对象：
        - **result_queue** (queue.Queue):
            由 execute() 写入 (conn, chunk) 的推理输出队列。
            conn：客户端 socket
            chunk：模型生成的文本片段或 "<END>"
        - **pickle**:
            用于序列化 chunk 内容。
        - **socket**:
            用于 shutdown/close 客户端连接。

    Outputs:
        无显式返回值。
        函数不断向各客户端 socket 推送数据，直到客户端会话结束。
        若 chunk 为 "<END>"，则关闭对应连接。

    Raises:
        函数内部捕获所有异常，不向外抛出。
        若发送失败，会自动关闭连接并将其标记为 closed。
    """
    closed = set()

    while True:
        conn, chunk = result_queue.get()

        if conn in closed:
            continue

        try:
            data = pickle.dumps(chunk)
            conn.send(len(data).to_bytes(8, "big"))
            conn.sendall(data)

            if isinstance(chunk, str) and chunk.strip() == "<END>":
                conn.shutdown(socket.SHUT_RDWR)
                conn.close()
                closed.add(conn)

        except Exception:
            closed.add(conn)
            try:
                conn.close()
            except:
                pass
            continue

if __name__ == "__main__":
    threading.Thread(target=heartbeat_server, daemon=True).start()
    threading.Thread(target=execute, daemon=True).start()
    threading.Thread(target=result_dispatcher, daemon=True).start()
    data_server()
