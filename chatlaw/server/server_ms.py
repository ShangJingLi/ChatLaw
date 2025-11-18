import socket
import threading
import time
import pickle
import queue
import mindspore as ms
import numpy as np

from mindnlp.transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    StoppingCriteriaList,
    StoppingCriteria,
)

# ==============================
# 基本配置
# ==============================
HEARTBEAT_PORT = 6005
DATA_PORT = 6006

stop_flag = threading.Event()
last_heartbeat_time = time.time()
alive = True

# ==============================
# MindSpore 初始化
# ==============================
ms.set_context(device_target="GPU")

# ==============================
# 模型加载
# ==============================
print("[Model] Loading Qwen model on GPU ...")
model_name = "Qwen/Qwen3-4B-Instruct-2507"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="cuda"
)

print("[Model] Qwen loaded successfully on GPU ✅")

# ==============================
# 队列
# ==============================
task_queue = queue.Queue()
result_queue = queue.Queue()

# ==============================
# 工具函数
# ==============================
def recv_exact(conn, n, timeout=None):
    if timeout:
        conn.settimeout(timeout)
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return buf


# ===========================================
#   停止条件（正确终止 generate）
# ===========================================
class StopNow(StoppingCriteria):
    def __init__(self, stop_event):
        super().__init__()
        self.stop_event = stop_event

    def __call__(self, input_ids, scores, **kwargs):
        return self.stop_event.is_set()


# ===========================================
#   心跳线程（PING/PONG）
# ===========================================
def heartbeat_server():
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


# ===========================================
#   数据监听线程
# ===========================================
def data_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", DATA_PORT))
    s.listen(5)
    print(f"[Data] Listening on {DATA_PORT}")

    while True:
        conn, addr = s.accept()
        print(f"[Data] Connected by {addr}")
        threading.Thread(target=handle_client, args=(conn,), daemon=True).start()


# ===========================================
#   客户端处理（一次请求）
# ===========================================
def handle_client(conn):
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


def safe_generate(model, kwargs, streamer, stop_event):
    try:
        model.generate(**kwargs)
    except Exception as e:
        print("[Generate] ERROR:", e)
        raise

def gpu_worker():
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

            gen_kwargs = dict(
                **fixed,
                streamer=streamer,
                max_new_tokens=4096,
                temperature=0.7,
                do_sample=True,
                stopping_criteria=stopping,
            )

            # ---- generate 线程 ----
            gen_thread = threading.Thread(
                target=safe_generate,
                args=(model, gen_kwargs, streamer, stop_flag),
                daemon=True,
            )
            gen_thread.start()

            # ---- 从 streamer 读 ----
            for text in streamer:
                if stop_flag.is_set():
                    print("[GPUWorker] STOP handled safely.")
                    break
                result_queue.put((conn, text))

            gen_thread.join(timeout=1)

        except Exception as e:
            result_queue.put((conn, f"[Error] {str(e)}"))

        finally:
            # ---- 清理 ----
            stop_flag.clear()
            result_queue.put((conn, "<END>"))
            print("[GPUWorker] Job done → <END>")


# ===========================================
#   dispatcher（发送给客户端）
# ===========================================
def result_dispatcher():
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
    threading.Thread(target=gpu_worker, daemon=True).start()
    threading.Thread(target=result_dispatcher, daemon=True).start()
    data_server()
