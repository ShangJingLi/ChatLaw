import os
import socket
import threading
import time
import pickle
import traceback
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
from chatlaw.dataloader import download_resources
from chatlaw.configuration import config
from launcher import get_resources_path

# ======== 基本配置 ========
HEARTBEAT_PORT = config.HEARTBEAT_PORT
DATA_PORT = config.DATA_PORT
stop_flag = threading.Event()  # 全局停止标志

last_heartbeat_time = time.time()
alive = True

class StopFlagCriteria(StoppingCriteria):
    def __init__(self, flag):
        self.flag = flag
    def __call__(self, input_ids, scores, **kwargs):
        # 返回 True 表示应当停止
        return self.flag.is_set()


print("[Backend] Checking available backend hardware...")

device = None
device_map = "auto"
use_npu = False

# 检查是否有NPU环境
if hasattr(torch, "npu"):
    try:
        if torch.npu.is_available():
            use_npu = True
            print("[Backend] Ascend NPU available.")
    except Exception:
        pass

if use_npu:
    try:
        import torch_npu
        device = "npu"
        device_map = {"": torch.npu.current_device()}
        print("[Backend] Using Ascend NPU backend.")
    except ImportError:
        raise RuntimeError(
            "Ascend NPU detected but torch_npu package is missing."
        )
# 使用GPU
elif torch.cuda.is_available():
    device = "cuda"
    device_map = "auto"
    print("[Backend] Using NVIDIA GPU.")
# 若不含NPU和GPU环境，抛出错误
else:
    raise RuntimeError("No compatible backend (Ascend/GPU) Found")


# 加载模型
print("[Model] Loading Qwen model...")
resource_path = get_resources_path()
download_resources(resource_type="tokenizer")
download_resources(resource_type="llm")

tokenizer_path = os.path.join(resource_path, "tokenizer").replace("\\", "/")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

model_path = os.path.join(resource_path, "llm").replace("\\", "/")
if device_map == "npu":
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype="auto",
        device_map=device_map
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        device_map="auto",  # ★ 保留 auto
        dtype="auto",
        attn_implementation="sdpa",
    )
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
print("[Model] Qwen loaded successfully.")


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
    if timeout is not None:
        conn.settimeout(timeout)
    buf = b''
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed while receiving")
        buf += chunk
    return buf


# ======== 心跳检测线程 ========
def heartbeat_server():
    """
    功能：
        启动一个心跳监测服务器，用于维持与客户端的连接状态检测。
        服务器监听指定心跳端口，当客户端发送 `PING` 时返回 `PONG`，
        并更新最近一次心跳时间，用于判断对端是否存活。
        当收到 `STOP` 指令时，函数会触发停止标志并终止当前会话。
        若在指定时间内未收到心跳包，则认为连接已丢失并结束本次会话。

    Inputs:
        本函数不接收外部输入参数。需要使用到以下全局变量与常量：
        - **HEARTBEAT_PORT** (int): 心跳监听端口号。
        - **last_heartbeat_time** (float): 全局变量，记录最近一次收到心跳包的时间戳。
        - **alive** (bool): 全局变量，表示当前连接是否存活。
        - **stop_flag** (Event): 全局事件对象，用于向推理线程发送停止信号。

    Outputs:
        无显式返回值。该函数以循环方式持续运行，直到进程结束或异常发生。

    Raises:
        本函数内部通过 try/except 捕获异常，不会主动抛出异常。
        所有异常均会被打印并在短暂延时后重试启动服务器。
    """
    global last_heartbeat_time, alive
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('0.0.0.0', HEARTBEAT_PORT))
            s.listen(1)
            print(f"[Heartbeat] Listening on port {HEARTBEAT_PORT}")
            conn, addr = s.accept()
            print(f"[Heartbeat] Connected by {addr}")
            conn.settimeout(2)

            last_heartbeat_time = time.time()
            alive = True

            session_active = True
            while session_active:
                try:
                    msg = conn.recv(16)
                    if not msg:
                        break
                    if msg == b'PING':
                        conn.sendall(b'PONG')
                        last_heartbeat_time = time.time()
                    elif msg == b'STOP':
                        print("[Heartbeat] STOP signal received!")
                        stop_flag.set()      # 通知推理线程停止
                        alive = False        # 终止心跳
                        break
                except socket.timeout:
                    if time.time() - last_heartbeat_time > 3:
                        print("[Heartbeat] Connection lost.")
                        alive = False
                        break
            conn.close()
            s.close()
            print("[Heartbeat] Connection closed, waiting for next client...")
        except Exception as e:
            print(f"[Heartbeat] Error: {e}")
            time.sleep(1)


# ======== 数据服务器 ========
def data_server():
    """
    功能：
        启动主数据监听服务器，用于接收客户端的数据连接请求。
        服务器在指定端口上持续监听，每当有新客户端连接时，
        会为该客户端创建独立的处理线程并委托给 `handle_client` 进行处理。
        该函数本身不阻塞于单一客户端，而是负责持续接受新连接。

    Inputs:
        本函数不接收外部输入参数，但依赖以下外部资源：
        - **DATA_PORT** (int): 数据服务监听端口号。
        - **handle_client** (callable): 客户端连接处理函数。
        - **alive** (bool): 全局变量，用于反映当前系统存活状态。

    Outputs:
        无显式返回值。该函数在循环中持续运行并不断接受新的客户端连接。

    Raises:
        本函数内部未进行异常捕获。如发生异常，将由上层调用者负责处理。
    """
    global alive  # pylint: disable=global-variable-not-assigned
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('0.0.0.0', DATA_PORT))
    s.listen(5)
    print(f"[Data] Listening on port {DATA_PORT}")

    while True:
        conn, addr = s.accept()
        print(f"[Data] Connected by {addr}")
        client_thread = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
        client_thread.start()


# ======== 单客户端处理逻辑 ========
def handle_client(conn, addr):
    """
    功能：
        处理单个客户端的数据推理请求。该函数首先与客户端进行固定格式的握手，
        随后在循环中接收数据头与数据体，解析得到模型输入，并以流式方式调用
        `stream_generate` 执行推理，将推理结果分片序列化后逐块发送给客户端。
        若推理正常结束或发生错误，均会发送流式结束标识，确保客户端安全退出。
        在网络异常、超时或连接关闭时，本函数会终止当前会话并关闭连接。

    Args:
        conn: 客户端连接的 socket 对象，用于收发数据。
        addr (tuple): 客户端的地址信息，一般为 ``(ip, port)``。

    Inputs:
        - **conn**: 可执行 `recv()`、`send()`、`sendall()` 等方法的连接对象。
        - **addr** (tuple): 客户端地址，仅用于日志打印或调试。
        - 函数内部依赖以下外部资源：
            - **alive** (bool): 全局变量，指示当前系统是否保持存活状态。
            - **recv_exact** (callable): 从连接中精确读取指定字节数的函数。
            - **stream_generate** (callable): 执行模型推理并按分片返回数据的生成器函数。

    Outputs:
        无显式返回值。本函数通过 socket 与客户端进行交互，并在结束时自动关闭连接。

    Raises:
        本函数内部使用多层 try/except 捕获所有异常，因此不会将错误向外抛出。
        任何错误均会记录日志，并确保连接在最终阶段被安全关闭。
    """
    global alive  # pylint: disable=global-variable-not-assigned
    try:
        # === 握手 ===
        first = recv_exact(conn, 1, timeout=2.0)
        if first != b'\xA5':
            raise ConnectionError(f"Invalid handshake byte: {first!r}")
        conn.sendall(b'\x5A')
        print(f"[Data] 1-byte handshake OK with {addr}")

        # === 循环接收 ===
        while alive:
            try:
                header = recv_exact(conn, 8, timeout=10)
                data_len = int.from_bytes(header, 'big')
                data = recv_exact(conn, data_len, timeout=10)
                model_inputs = pickle.loads(data)
                print(f"[Data] Got model input tensor from {addr}")

                # === 模型推理（流式） ===
                try:
                    for chunk in stream_generate(model_inputs):
                        serialized = pickle.dumps(chunk)
                        conn.send(len(serialized).to_bytes(8, 'big'))
                        conn.sendall(serialized)

                    # 发送结束标识
                    end_flag = pickle.dumps("<END>")
                    conn.send(len(end_flag).to_bytes(8, 'big'))
                    conn.sendall(end_flag)
                    print(f"[Data] Stream finished for {addr}")


                except Exception as e:
                    err_msg = f"[ServerError] {type(e).__name__}: {e}"
                    print(err_msg)
                    serialized = pickle.dumps(err_msg)
                    conn.send(len(serialized).to_bytes(8, 'big'))
                    conn.sendall(serialized)
                    # 发送结束标志，让客户端安全退出流式循环
                    end_flag = pickle.dumps("<END>")
                    conn.send(len(end_flag).to_bytes(8, 'big'))
                    conn.sendall(end_flag)


            except socket.timeout:
                print(f"[Data] Timeout waiting data from {addr}")
                continue
            except ConnectionError as e:
                print(f"[Data] ConnectionError {addr}: {e}")
                break
            except Exception as e:
                traceback.print_exc()
                print(f"[Data] Unexpected error from {addr}: {e}")
                break

    except Exception as e:
        print(f"[Data] Error during handshake/init: {e}")
    finally:
        conn.close()
        print(f"[Data] Connection closed for {addr}")


# ======== Qwen 流式生成函数 ========
def stream_generate(model_inputs):
    """
    功能：
        将客户端传入的模型输入张量送入 Qwen 模型进行流式文本生成。
        本函数使用 `TextIteratorStreamer` 实现“增量文本”输出，即模型每生成一段新的文本
        即可被即时捕获并通过 `yield` 返回，便于服务器向客户端实时推送推理结果。
        同时支持外部停止标志 `stop_flag`，可在推理过程中随时终止生成并安全退出。

    Args:
        model_inputs (dict): 由客户端发送的模型输入字典，通常包含：
            - ``"input_ids"``: 输入 token 序列张量。
            - ``"attention_mask"``: 注意力掩码张量。

    Inputs:
        - **model_inputs** (dict): 必须包含两个键：
            - ``input_ids`` (Tensor): 用于生成文本的输入 token 序列。
            - ``attention_mask`` (Tensor): 指定哪些位置参与 attention 计算。
        - 函数依赖全局变量与外部对象：
            - **model**: Qwen 模型实例，需提供 ``generate`` 方法。
            - **tokenizer**: 对应的分词器，用于 streamer 文本处理。
            - **stop_flag** (Event): 外部停止信号，用于提前终止生成。

    Outputs:
        这是一个 Python 生成器（generator）。
        每次调用 `yield` 返回一段模型新生成的 **新增文本字符串（str）**，
        不包含已经生成过的内容，适用于流式输出场景。

    Raises:
        本函数内部不主动抛出异常。模型生成中的异常会在上层捕获。
    """
    input_ids = model_inputs["input_ids"].to(model.device)
    attention_mask = model_inputs["attention_mask"].to(model.device)

    # 创建流式输出器
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )
    generation_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        streamer=streamer,
        max_new_tokens=4096,
        temperature=0.7,
        do_sample=True,
        stopping_criteria=StoppingCriteriaList([StopFlagCriteria(stop_flag)]),  # ★ 加上这一行
    )

    # 生成线程（异步）
    def _run_generate():
        if device == "cuda":
            with torch.inference_mode(), torch.autocast("cuda"):
                model.generate(**generation_kwargs)
        else:
            with torch.inference_mode():
                model.generate(**generation_kwargs)

    thread = threading.Thread(target=_run_generate)
    thread.start()
    # thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    # thread.start()


    # 正确的流式返回：只发送“新增文本”
    for new_text in streamer:
        if stop_flag.is_set():  # 用户发了STOP
            print("[Data] Generation stopped by client request.")
            break
        if new_text.strip():
            yield new_text
    stop_flag.clear()  # 复位标志

# ======== 主入口 ========
if __name__ == "__main__":
    hb_thread = threading.Thread(target=heartbeat_server, daemon=True)
    hb_thread.start()
    data_server()
