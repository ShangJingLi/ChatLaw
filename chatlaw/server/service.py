"""服务端网络服务层：心跳、数据连接、单次咨询编排。

重构后的数据流（单条数据连接内顺序完成）：
    1. 客户端握手（A5 -> 5A）；
    2. 客户端发送一条咨询请求：{"kind": "text"|"audio", ...}；
    3. 服务端：语音先转文字 -> 知识库检索 -> 先把 {识别文本, 检索结果} 发回客户端；
    4. 服务端：构造 prompt + chat template 分词 -> 流式推理 -> 逐段发回 chunk；
    5. 服务端发送 {"type": "end"} 结束。

设计约束：本模块**不依赖任何推理引擎（vllm / transformers）**。真正的生成能力
通过 ``generate_fn`` 注入，因此可以用一个假的 generate_fn 在无 GPU 机器上端到端测试。

线路帧格式（沿用原协议）：8 字节大端长度前缀 + pickle 负载。
"""
import os
import pickle
import socket
import threading
import time
import traceback

from launcher import get_resources_path
from chatlaw.server.utils.asr_utils import speech_to_text
from chatlaw.server.utils.rag_utils import retrieve_laws, build_prompt, doc_to_dict


HANDSHAKE_REQ = b"\xA5"
HANDSHAKE_RESP = b"\x5A"


def recv_exact(conn, n, timeout=None):
    """
    从连接中精确读取 n 字节；对端提前关闭则抛出 ConnectionError。
    timeout 为 None 时不改变当前 socket 超时设置。
    """
    if timeout is not None:
        conn.settimeout(timeout)
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed while receiving")
        buf += chunk
    return buf


def send_msg(conn, obj):
    """发送一条 pickle 消息（8 字节长度前缀 + 负载）。"""
    body = pickle.dumps(obj)
    conn.sendall(len(body).to_bytes(8, "big"))
    conn.sendall(body)


def recv_msg(conn, timeout=None):
    """接收一条 pickle 消息。"""
    header = recv_exact(conn, 8, timeout=timeout)
    data_len = int.from_bytes(header, "big")
    data = recv_exact(conn, data_len, timeout=timeout)
    return pickle.loads(data)


# ======== 心跳服务器（当前仍为单会话；多用户在后续步骤重构） ========
def heartbeat_server(hb_port, stop_flag):
    """
    监听心跳端口。收到 PING 回 PONG；收到 STOP 置位 stop_flag（用于中断当前生成）；
    若长时间收不到心跳则认为连接丢失并等待下一个客户端。
    """
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", hb_port))
            s.listen(1)
            print(f"[Heartbeat] Listening on port {hb_port}")
            conn, addr = s.accept()
            print(f"[Heartbeat] Connected by {addr}")
            conn.settimeout(2)

            last_heartbeat_time = time.time()
            session_active = True
            while session_active:
                try:
                    msg = conn.recv(16)
                    if not msg:
                        break
                    if msg == b"PING":
                        conn.sendall(b"PONG")
                        last_heartbeat_time = time.time()
                    elif msg == b"STOP":
                        print("[Heartbeat] STOP signal received!")
                        stop_flag.set()  # 通知推理线程停止
                        break
                except socket.timeout:
                    if time.time() - last_heartbeat_time > 3:
                        print("[Heartbeat] Connection lost.")
                        break
            conn.close()
            s.close()
            print("[Heartbeat] Connection closed, waiting for next client...")
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"[Heartbeat] Error: {e}")
            time.sleep(1)


# ======== 数据服务器：每条连接一个处理线程 ========
def data_server(data_port, handle_client):
    """
    监听数据端口，为每个连接启动一个 handle_client 线程。
    handle_client 由调用方提供（通常是 InferenceService.handle_client）。
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", data_port))
    s.listen(5)
    print(f"[Data] Listening on port {data_port}")

    while True:
        conn, addr = s.accept()
        print(f"[Data] Connected by {addr}")
        threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()


# ======== 咨询编排 ========
class InferenceService:
    """
    持有服务端全部推理相关资源，并把「一次咨询」的完整流程编排起来。

    资源在构造时注入，真正的文本生成通过 ``generate_fn`` 注入：
        generate_fn(prompt_token_ids) -> 生成器，逐段 yield 新增文本(str)

    这样 service 层与具体引擎（vllm / transformers / mock）完全解耦。
    """

    def __init__(self, tokenizer, audio_model, vectorstore, exact_index,
                 law_name_candidates, generate_fn,
                 target_sr=16000, audio_cache_dir=None):
        self.tokenizer = tokenizer
        self.audio_model = audio_model
        self.vectorstore = vectorstore
        self.exact_index = exact_index
        self.law_name_candidates = law_name_candidates
        self.generate_fn = generate_fn
        self.target_sr = target_sr

        if audio_cache_dir is None:
            audio_cache_dir = os.path.join(get_resources_path(), "_asr_cache")
        os.makedirs(audio_cache_dir, exist_ok=True)
        self.audio_cache_dir = audio_cache_dir

    def _resolve_query(self, request):
        """把请求解析成查询文本；语音请求先转文字。"""
        kind = request.get("kind")
        if kind == "audio":
            return speech_to_text(
                request.get("audio"),
                target_sr=self.target_sr,
                audio_cache_dir=self.audio_cache_dir,
                audio_model=self.audio_model,
            )
        if kind == "text":
            return (request.get("text") or "").strip()
        raise ValueError(f"未知请求类型: {kind!r}")

    def _run_consultation(self, conn, addr, request):
        # 1. 得到查询文本（语音先转文字）
        query = self._resolve_query(request)

        # 2. 知识库检索 -> 归一化为纯 dict（客户端无需 langchain 即可解析）
        raw_docs = retrieve_laws(
            vectorstore=self.vectorstore,
            query=query,
            exact_index=self.exact_index,
            law_name_candidates=self.law_name_candidates,
        )
        docs = [doc_to_dict(d) for d in raw_docs]

        # 3. 先把识别文本 + 检索结果发回客户端（客户端据此显示知识库面板）
        send_msg(conn, {"type": "retrieval", "query": query, "docs": docs})
        print(f"[Data] Retrieval sent to {addr}: query={query!r}, {len(docs)} docs")

        # 4. 构造 prompt 并分词（chat template 也在服务端完成）
        prompt = build_prompt(query, docs)
        messages = [{"role": "user", "content": prompt}]
        prompt_token_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )

        # 5. 流式推理
        for chunk in self.generate_fn(prompt_token_ids):
            send_msg(conn, {"type": "chunk", "data": chunk})

        send_msg(conn, {"type": "end"})
        print(f"[Data] Stream finished for {addr}")

    def handle_client(self, conn, addr):
        try:
            # === 握手 ===
            first = recv_exact(conn, 1, timeout=5.0)
            if first != HANDSHAKE_REQ:
                raise ConnectionError(f"Invalid handshake byte: {first!r}")
            conn.sendall(HANDSHAKE_RESP)
            print(f"[Data] 1-byte handshake OK with {addr}")

            # 握手后转为阻塞等待请求（客户端每次咨询新开一条连接，用完即关）
            conn.settimeout(None)

            while True:
                try:
                    request = recv_msg(conn, timeout=None)
                except (ConnectionError, socket.timeout):
                    break

                try:
                    self._run_consultation(conn, addr, request)
                except Exception as e:  # pylint: disable=broad-exception-caught
                    err_msg = f"[ServerError] {type(e).__name__}: {e}"
                    traceback.print_exc()
                    try:
                        send_msg(conn, {"type": "error", "message": err_msg})
                        send_msg(conn, {"type": "end"})
                    except Exception:  # pylint: disable=broad-exception-caught
                        break

        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"[Data] Error during handshake/init for {addr}: {e}")
        finally:
            conn.close()
            print(f"[Data] Connection closed for {addr}")


__all__ = [
    "HANDSHAKE_REQ",
    "HANDSHAKE_RESP",
    "recv_exact",
    "send_msg",
    "recv_msg",
    "heartbeat_server",
    "data_server",
    "InferenceService",
]
