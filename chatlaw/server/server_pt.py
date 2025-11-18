import os
import socket
import threading
import time
import pickle
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
from chatlaw.dataloader import download_resources
from launcher import get_resources_path

# ======== 基本配置 ========
HEARTBEAT_PORT = 6005
DATA_PORT = 6006
stop_flag = threading.Event()  # 全局停止标志

last_heartbeat_time = time.time()
alive = True

class StopFlagCriteria(StoppingCriteria):
    def __init__(self, flag):
        self.flag = flag
    def __call__(self, input_ids, scores, **kwargs):
        # 返回 True 表示应当停止
        return self.flag.is_set()


# ======== 模型加载 ========
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
    torch_dtype="auto",
    device_map="auto"
)
print("[Model] Qwen loaded successfully.")


# ======== 通用函数 ========
def recv_exact(conn, n, timeout=None):
    """确保接收到指定字节数"""
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
    """主数据监听"""
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
    """处理客户端推理请求"""
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
    """将客户端发来的张量送入Qwen模型进行流式推理（仅发送增量文本）"""
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
        max_new_tokens=16384,
        temperature=0.7,
        do_sample=True,
        stopping_criteria=StoppingCriteriaList([StopFlagCriteria(stop_flag)]),  # ★ 加上这一行
    )

    # 生成线程（异步）
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()


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
