# utils_pt.py

import pickle
import socket
import time


# -----------------------------------------------------------
# 心跳线程
# -----------------------------------------------------------
def heartbeat_client(server_ip, hb_port, hb_interval, hb_timeout,
                     alive_flag, stop_event, recv_exact_fn):
    """
    心跳线程：
    - 周期性向服务器发送 PING
    - 收到 PONG 刷新 last_ok
    - 若超时 hb_timeout，则认为服务器掉线，设置 stop_event
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3.0)
    s.connect((server_ip, hb_port))
    print("[HB] Connected.", flush=True)
    s.settimeout(1.0)

    last_ok = time.time()
    try:
        while alive_flag():
            # UI 主动停止：只负责告诉服务器“STOP”，然后退出心跳线程
            if stop_event.is_set():
                try:
                    s.sendall(b"STOP")
                    print("[HB] STOP sent.")
                except Exception:
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
                    # 告诉数据线程：别等了，退出吧
                    stop_event.set()
                    break

    finally:
        s.close()
        print("[HB] Closed.")


# -----------------------------------------------------------
# 数据流式接收
# -----------------------------------------------------------
def stream_from_server(server_ip, data_port,
                       handshake_req, handshake_resp,
                       recv_exact_fn,
                       alive_flag, stop_event,
                       input_tensor):
    """
    与服务器建立数据连接、发送输入、并流式接收模型输出。

    设计要点：
    - STOP 时：不再向 UI 渲染 chunk，但仍继续读取，直到 <END> 或连接中断。
    - 服务器掉线 / 网络断开：连续多次 timeout 判定连接挂掉，返回 ClientError。
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5.0)
    s.connect((server_ip, data_port))
    s.settimeout(2.0)

    timeout_count = 0
    MAX_TIMEOUT = 3   # 连续 3 次 timeout 就认为连接挂了

    try:
        # --- handshake ---
        s.sendall(handshake_req)
        resp = recv_exact_fn(s, 1, timeout=2.0)
        if resp != handshake_resp:
            raise ConnectionError("data handshake failed")

        # --- send input ---
        body = pickle.dumps(input_tensor)
        s.send(len(body).to_bytes(8, "big"))
        s.sendall(body)

        # --- receive stream ---
        while alive_flag():
            # 是否屏蔽输出（STOP 按钮已按下）
            suppress_output = stop_event.is_set()

            try:
                hdr = s.recv(8)
                if not hdr:
                    print("[DATA] Server closed connection.")
                    break

                if len(hdr) < 8:
                    hdr += recv_exact_fn(s, 8 - len(hdr), timeout=2.0)

                data_len = int.from_bytes(hdr, "big")
                data = recv_exact_fn(s, data_len, timeout=2.0)

                timeout_count = 0  # 收到数据就清零

            except socket.timeout:
                timeout_count += 1
                if timeout_count >= MAX_TIMEOUT:
                    # 彻底认定连接挂了
                    yield "[ClientError] Connection timed out"
                    break
                continue

            chunk = pickle.loads(data)
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8", "ignore")

            # 服务器结束信号
            if str(chunk).strip() == "<END>":
                yield "<END>"
                break

            # STOP 状态下继续读，但不往 UI 渲染
            if not suppress_output:
                yield chunk

    except Exception as e:
        yield f"[ClientError] {e}"

    finally:
        try:
            s.close()
        except Exception:
            pass
        print("[DATA] Closed.")
