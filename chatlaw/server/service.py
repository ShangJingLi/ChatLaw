"""服务端网络服务层：心跳、数据连接、单次咨询编排（多客户端并发版）。

数据流（单条数据连接内顺序完成，多条连接可并发）：
    1. 客户端握手（A5 -> 5A）；
    2. 客户端发送一条咨询请求：{"session_id": ..., "kind": "text"|"audio", ...}；
    3. 服务端：并发准入 -> 语音转文字 -> 知识库检索 -> 先把 {识别文本, 检索结果} 发回；
    4. 服务端：构造 prompt + chat template 分词 -> 流式推理 -> 逐段发回 chunk；
    5. 服务端发送 {"type": "end"} 结束。

多客户端要点：
    - 每条数据连接一个 handle_client 线程；vLLM 引擎在自己的事件循环里对多请求做
      continuous batching，并按 request_id 严格隔离，参数只加载一份。
    - 每次咨询有独立的 session_id，服务端用 SessionRegistry 维护 session_id ->
      SessionHandle(内含专属 stop_event)。心跳通道收到该 session 的 STOP、或探测到
      客户端掉线时，只置位该会话的 stop_event，互不影响其他客户端。
    - transformers 回退后端不支持并发（model.generate 被串行化），由 BusyGate 保证
      同一时刻只处理一个会话，其余请求被显式拒绝（{"type": "reject"}）。

设计约束：本模块**不依赖任何推理引擎（vllm / transformers）**。真正的生成能力
通过 ``generate_fn(prompt_token_ids, stop_event)`` 注入，因此可用假的 generate_fn
在无 GPU 机器上端到端测试。

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

# 心跳通道会话注册帧：客户端连接后先发送固定 32 字节的 session_id（uuid4().hex）
SESSION_ID_LEN = 32


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


# ======== 会话状态管理 ========
class SessionHandle:
    """一次咨询会话的共享控制块，被同一 session_id 的心跳/数据两条通道共享。"""

    def __init__(self, session_id):
        self.session_id = session_id
        self.stop_event = threading.Event()  # 该会话专属的停止/中断信号


class SessionRegistry:
    """线程安全的 session_id -> SessionHandle 注册表（带引用计数防泄漏）。

    心跳通道与数据通道谁先到都可 get_or_create，各自在结束时 release；引用计数
    归零才真正删除，避免"某条通道未到达/异常退出"造成句柄泄漏。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._sessions = {}  # session_id -> [SessionHandle, refcount]

    def get_or_create(self, session_id):
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                entry = [SessionHandle(session_id), 0]
                self._sessions[session_id] = entry
            entry[1] += 1
            return entry[0]

    def get(self, session_id):
        with self._lock:
            entry = self._sessions.get(session_id)
            return entry[0] if entry else None

    def release(self, session_id):
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                return
            entry[1] -= 1
            if entry[1] <= 0:
                del self._sessions[session_id]


class BusyGate:
    """并发准入闸门。

    - concurrent=True（vLLM）：恒放行，靠 vLLM 的 continuous batching 支撑并发。
    - concurrent=False（transformers 回退）：同一时刻只允许一个会话，其余 try_enter
      返回 False，由调用方向客户端回发拒绝信号。
    """

    def __init__(self, concurrent):
        self.concurrent = concurrent
        self._lock = threading.Lock()
        self._active = 0

    def try_enter(self):
        if self.concurrent:
            return True
        with self._lock:
            if self._active > 0:
                return False
            self._active += 1
            return True

    def leave(self):
        if self.concurrent:
            return
        with self._lock:
            if self._active > 0:
                self._active -= 1


# ======== 心跳服务器（并发：每条心跳连接一个线程） ========
def heartbeat_server(hb_port, registry):
    """
    监听心跳端口，为每个心跳连接启动一个 _heartbeat_handler 线程。
    每个连接先上报 32 字节 session_id，之后 PING/PONG 保活，收到 STOP 或探测到掉线
    时置位对应会话的 stop_event（只影响该会话）。
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", hb_port))
    s.listen(16)
    print(f"[Heartbeat] Listening on port {hb_port}")

    while True:
        try:
            conn, addr = s.accept()
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"[Heartbeat] Accept error: {e}")
            time.sleep(0.5)
            continue
        threading.Thread(
            target=_heartbeat_handler, args=(conn, addr, registry), daemon=True
        ).start()


def _heartbeat_handler(conn, addr, registry):
    session_id = None
    try:
        session_id = recv_exact(conn, SESSION_ID_LEN, timeout=5.0).decode("ascii")
        handle = registry.get_or_create(session_id)
        print(f"[Heartbeat] Connected by {addr}, session={session_id}")
        conn.settimeout(2)

        last_heartbeat_time = time.time()
        while True:
            try:
                msg = conn.recv(16)
                if not msg:
                    break  # 客户端正常关闭
                if msg == b"PING":
                    conn.sendall(b"PONG")
                    last_heartbeat_time = time.time()
                elif msg == b"STOP":
                    print(f"[Heartbeat] STOP for session={session_id}")
                    handle.stop_event.set()  # 只中断本会话生成
                    break
            except socket.timeout:
                if time.time() - last_heartbeat_time > 3:
                    print(f"[Heartbeat] Lost session={session_id}, aborting its generation")
                    handle.stop_event.set()  # 掉线 -> abort，释放 batch 槽位
                    break
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"[Heartbeat] Error for {addr}: {e}")
    finally:
        try:
            conn.close()
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        if session_id is not None:
            registry.release(session_id)
        print(f"[Heartbeat] Connection closed for {addr}, session={session_id}")


# ======== 数据服务器：每条连接一个处理线程 ========
def data_server(data_port, handle_client):
    """
    监听数据端口，为每个连接启动一个 handle_client 线程。
    handle_client 由调用方提供（通常是 InferenceService.handle_client）。
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", data_port))
    s.listen(16)
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
        generate_fn(prompt_token_ids, stop_event) -> 生成器，逐段 yield 新增文本(str)

    并发控制通过 ``registry``（每会话 stop_event）与 ``gate``（transformers 单请求
    准入）注入，因此 service 层与具体引擎（vllm / transformers / mock）完全解耦。
    """

    def __init__(self, tokenizer, audio_model, vectorstore, exact_index,
                 law_name_candidates, generate_fn, registry, gate,
                 target_sr=16000, audio_cache_dir=None):
        self.tokenizer = tokenizer
        self.audio_model = audio_model
        self.vectorstore = vectorstore
        self.exact_index = exact_index
        self.law_name_candidates = law_name_candidates
        self.generate_fn = generate_fn
        self.registry = registry
        self.gate = gate
        self.target_sr = target_sr

        # ASR(funasr-onnx) 封装的并发安全性存疑，且 STT 很快，用锁串行化，代价小。
        self._asr_lock = threading.Lock()

        if audio_cache_dir is None:
            audio_cache_dir = os.path.join(get_resources_path(), "_asr_cache")
        os.makedirs(audio_cache_dir, exist_ok=True)
        self.audio_cache_dir = audio_cache_dir

    def _resolve_query(self, request):
        """把请求解析成查询文本；语音请求先转文字（ASR 串行化）。"""
        kind = request.get("kind")
        if kind == "audio":
            with self._asr_lock:
                return speech_to_text(
                    request.get("audio"),
                    target_sr=self.target_sr,
                    audio_cache_dir=self.audio_cache_dir,
                    audio_model=self.audio_model,
                )
        if kind == "text":
            return (request.get("text") or "").strip()
        raise ValueError(f"未知请求类型: {kind!r}")

    def _run_consultation(self, conn, addr, request, stop_event):
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
        # 注意：transformers 5.x 起 apply_chat_template(tokenize=True) 默认
        # return_dict=True，会返回 BatchEncoding（dict）而非 List[int]；下游引擎
        # 需要的是扁平 token id 列表，故显式 return_dict=False 保持跨版本一致。
        prompt_token_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False
        )

        # 5. 流式推理（使用本会话专属 stop_event）
        for chunk in self.generate_fn(prompt_token_ids, stop_event):
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

                session_id = request.get("session_id")

                # 并发准入：transformers 单请求模式下，已有会话在跑则直接拒绝，
                # 在 STT/RAG 之前拦截，不浪费算力。vLLM 模式恒放行。
                if not self.gate.try_enter():
                    print(f"[Data] Rejecting {addr} (backend busy, single-request mode)")
                    try:
                        send_msg(conn, {
                            "type": "reject",
                            "reason": "服务端当前为 transformers 单请求模式，已有会话进行中，请稍后重试。",
                        })
                        send_msg(conn, {"type": "end"})
                    except Exception:  # pylint: disable=broad-exception-caught
                        break
                    continue

                handle = self.registry.get_or_create(session_id) if session_id else None
                stop_event = handle.stop_event if handle else threading.Event()

                try:
                    self._run_consultation(conn, addr, request, stop_event)
                except Exception as e:  # pylint: disable=broad-exception-caught
                    err_msg = f"[ServerError] {type(e).__name__}: {e}"
                    traceback.print_exc()
                    try:
                        send_msg(conn, {"type": "error", "message": err_msg})
                        send_msg(conn, {"type": "end"})
                    except Exception:  # pylint: disable=broad-exception-caught
                        break
                finally:
                    if session_id:
                        self.registry.release(session_id)
                    self.gate.leave()

        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"[Data] Error during handshake/init for {addr}: {e}")
        finally:
            conn.close()
            print(f"[Data] Connection closed for {addr}")


__all__ = [
    "HANDSHAKE_REQ",
    "HANDSHAKE_RESP",
    "SESSION_ID_LEN",
    "recv_exact",
    "send_msg",
    "recv_msg",
    "SessionHandle",
    "SessionRegistry",
    "BusyGate",
    "heartbeat_server",
    "data_server",
    "InferenceService",
]
