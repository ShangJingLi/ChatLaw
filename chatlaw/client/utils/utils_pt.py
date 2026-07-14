# utils_pt.py
"""客户端与服务端的两条通道逻辑：
    - heartbeat_client   : 心跳 / STOP 控制通道（不变）
    - stream_consultation: 数据通道。发送一条咨询请求（文本或音频），
                           依次接收「检索结果」与「流式推理输出」。

线路帧格式（与服务端一致）：8 字节大端长度前缀 + pickle 负载。
"""
import pickle
import socket
import time


def heartbeat_client(server_ip, hb_port, session_id, hb_interval, hb_timeout,
                     alive_flag, stop_event, recv_exact_fn):
    """
    客户端心跳线程：
        1. 连接后先上报本次会话的 session_id（32 字节），供服务端把心跳/数据/生成关联；
        2. 周期性发送 PING，服务端回 PONG；
        3. 超过 hb_timeout 收不到 PONG 视为服务端掉线，置位 stop_event；
        4. UI 触发 stop_event 时，向服务端发送 STOP 并退出（只中断本会话生成）。
    只负责连接存活性检测与停止信号，不参与数据接收。
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3.0)
    s.connect((server_ip, hb_port))
    # 会话注册帧：固定 32 字节的 session_id（uuid4().hex），必须先于任何 PING 发送
    s.sendall(session_id.encode("ascii"))
    print(f"[HB] Connected, session={session_id}.", flush=True)
    s.settimeout(1.0)

    last_ok = time.time()
    try:
        while alive_flag():
            # UI 主动停止：只负责告诉服务端 STOP，然后退出心跳线程
            if stop_event.is_set():
                try:
                    s.sendall(b"STOP")
                    print("[HB] STOP sent.")
                except Exception:  # pylint: disable=broad-exception-caught
                    pass
                break

            try:
                s.sendall(b"PING")
                pong = recv_exact_fn(s, 4, timeout=1.0)
                if pong != b"PONG":
                    raise ConnectionError("bad PONG response")

                last_ok = time.time()
                time.sleep(hb_interval)

            except socket.timeout:
                # 超时但还没到阈值，先再等等
                if time.time() - last_ok > hb_timeout:
                    print("[HB] timeout, marking dead")
                    stop_event.set()
                    break

    finally:
        s.close()
        print("[HB] Closed.")


def _send_msg(conn, obj):
    body = pickle.dumps(obj)
    conn.sendall(len(body).to_bytes(8, "big"))
    conn.sendall(body)


def stream_consultation(server_ip, data_port,
                        handshake_req, handshake_resp,
                        recv_exact_fn,
                        alive_flag, stop_event,
                        request):
    """
    与服务端建立一条数据连接，完成一次完整咨询：

        握手 -> 发送 request -> 依次接收服务端消息，逐条以事件形式 yield 出去。

    request（客户端构造）：
        {"kind": "text",  "text": "..."}                     或
        {"kind": "audio", "audio": (sample_rate, np.ndarray)}

    产出的事件（供 UI 消费）：
        ("retrieval", query:str, docs:list[dict])  —— 识别文本 + 检索到的法条
        ("chunk", text:str)                        —— 模型新增文本
        ("end",)                                   —— 推理结束
        ("error", message:str)                     —— 出错信息
        ("reject", reason:str)                     —— 服务端拒绝（如 transformers 单请求繁忙）

    停止语义：stop_event 置位后，仍继续接收 chunk（保证与服务端状态一致），
    但不再把 chunk 透出给 UI，直到收到 end。
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5.0)
    s.connect((server_ip, data_port))
    s.settimeout(2.0)

    timeout_count = 0
    # STT + RAG + 首 token 可能较慢，容忍更长的静默（3s * 10 = 30s）
    max_timeout = 10

    try:
        # --- 握手 ---
        s.sendall(handshake_req)
        resp = recv_exact_fn(s, 1, timeout=3.0)
        if resp != handshake_resp:
            raise ConnectionError("data handshake failed")

        # --- 发送咨询请求 ---
        _send_msg(s, request)

        # --- 接收消息流 ---
        while alive_flag():
            suppress_output = stop_event.is_set()

            try:
                header = recv_exact_fn(s, 8, timeout=3.0)
                data_len = int.from_bytes(header, "big")
                data = recv_exact_fn(s, data_len, timeout=3.0)
                timeout_count = 0
            except socket.timeout:
                timeout_count += 1
                if timeout_count >= max_timeout:
                    yield ("error", "Connection timed out")
                    break
                continue

            msg = pickle.loads(data)
            mtype = msg.get("type")

            if mtype == "retrieval":
                yield ("retrieval", msg.get("query", ""), msg.get("docs", []))
            elif mtype == "chunk":
                if not suppress_output:
                    yield ("chunk", msg.get("data", ""))
            elif mtype == "reject":
                # 服务端拒绝本次请求（如 transformers 单请求模式繁忙），随后会补发 end
                yield ("reject", msg.get("reason", "服务端拒绝了本次请求"))
            elif mtype == "error":
                yield ("error", msg.get("message", ""))
                # 服务端出错后仍会补发 end，这里继续读直到 end
            elif mtype == "end":
                yield ("end",)
                break

    except Exception as e:  # pylint: disable=broad-exception-caught
        yield ("error", str(e))

    finally:
        try:
            s.close()
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        print("[DATA] Closed.")


__all__ = ["heartbeat_client", "stream_consultation"]
