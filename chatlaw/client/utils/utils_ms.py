# utils_ms.py

import socket
import pickle
import time


def heartbeat_client_ms(server_ip, hb_port, hb_interval, hb_timeout,
                        alive_flag, stop_event, recv_exact_fn):
    """
    客户端心跳线程，用于维持与服务器的 PING/PONG 保活连接。
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3.0)
    s.connect((server_ip, hb_port))
    print("[HB] Connected.", flush=True)

    s.settimeout(1.0)
    last_ok = time.time()

    try:
        while alive_flag():
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
                    raise ConnectionError("Bad PONG response")

                last_ok = time.time()
                time.sleep(hb_interval)

            except socket.timeout:
                if time.time() - last_ok > hb_timeout:
                    print("[HB] Timeout, marking dead")
                    break

    finally:
        try:
            s.close()
        except Exception:
            pass
        print("[HB] Closed.")


# -----------------------------------------------------------
# MindNLP 版流式接收
# -----------------------------------------------------------
def stream_from_server_ms(
        server_ip, data_port,
        handshake_req, handshake_resp,
        recv_exact_fn,
        stop_event,
        input_tensor):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5.0)
    s.connect((server_ip, data_port))
    s.settimeout(2.0)

    try:
        # --- 握手 ---
        s.sendall(handshake_req)
        resp = recv_exact_fn(s, 1, timeout=2.0)
        if resp != handshake_resp:
            raise ConnectionError("Handshake failed")

        # --- 发送输入 ---
        body = pickle.dumps(input_tensor)
        s.send(len(body).to_bytes(8, "big"))
        s.sendall(body)

        # 使用死循环，直到读到 <END>
        while True:

            # STOP 只表示“不渲染内容”，但仍继续读取服务器数据
            if stop_event.is_set():
                pass  # 不 break，让服务器把 <END> 发完

            try:
                hdr = s.recv(8)
                if not hdr:
                    break
                if len(hdr) < 8:
                    hdr += recv_exact_fn(s, 8 - len(hdr), timeout=2.0)

                data_len = int.from_bytes(hdr, "big")
                data = recv_exact_fn(s, data_len, timeout=2.0)

            except socket.timeout:
                continue

            chunk = pickle.loads(data)

            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8", "ignore")

            # 服务端结束信号
            if str(chunk).strip() == "<END>":
                yield "<END>"
                break

            # STOP 期间不渲染，但仍然正常 read chunk
            if not stop_event.is_set():
                yield chunk

    except Exception as e:
        yield f"[ClientError] {e}"

    finally:
        try:
            s.close()
        except Exception:
            pass
        print("[DATA] Closed.")
