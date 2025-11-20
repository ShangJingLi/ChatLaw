import socket
import pickle
import time


def heartbeat_client_ms(server_ip, hb_port, hb_interval, hb_timeout,
                        alive_flag, stop_event, recv_exact_fn):
    """
    功能：
        该函数作为心跳客户端，与服务器的心跳端口保持持续通信，
        用于检测服务器连接是否保持存活。
        客户端周期性发送 `PING`，并期望收到服务器返回的 `PONG`。
        若服务器在指定超时时间内未响应，则视为连接中断。
        当本地触发 stop_event 时，心跳线程会发送 `STOP` 通知服务器，
        以请求中断当前推理任务。

    Args:
        server_ip (str): 服务器地址。
        hb_port (int): 心跳端口号。
        hb_interval (float): 发送心跳包（PING）之间的时间间隔（秒）。
        hb_timeout (float): 若超过此时间未收到 `PONG`，视为心跳超时。
        alive_flag (callable): 返回布尔值的函数，用于指示客户端是否继续运行。
        stop_event (Event): 若被触发，则向服务器发送 STOP 并终止心跳。
        recv_exact_fn (callable): 从 socket 精确读取指定字节数的函数。

    Inputs:
        - **server_ip**: 服务器 IP，用于建立心跳连接。
        - **hb_port**: 心跳端口，应与服务器 heartbeat_server 监听的一致。
        - **hb_interval**: 心跳发送周期。
        - **hb_timeout**: 超时判定时间。
        - **alive_flag()**: 决定心跳线程是否继续运行。
        - **stop_event**: 外部触发以终止推理与心跳。
        - **recv_exact_fn**: 读取服务器心跳响应的函数。

    Outputs:
        无显式返回值。
        本函数在内部维持 socket 心跳连接，并在会话结束时自动关闭连接。

    Raises:
        本函数不会向外抛出异常。
        所有网络错误均会在内部捕获并导致心跳线程安全退出。
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

def stream_from_server_ms(
        server_ip, data_port,
        handshake_req, handshake_resp,
        recv_exact_fn,
        stop_event,
        input_tensor):
    """
    功能：
        与服务器的数据端口建立连接，完成一次推理会话。
        客户端将输入张量序列化后发送给服务器，随后进入循环读取服务器返回的
        流式推理结果。每次接收到的片段经反序列化后作为生成器输出，从而实现
        流式推理。

        当服务器返回 `<END>` 或客户端触发 stop_event 时，会话结束并关闭连接。
        在 STOP 模式下，客户端仍会持续接收服务器发来的数据，但不会向上层渲染，
        以确保服务器能够完整发送 `<END>`，避免两端状态不一致。

    Args:
        server_ip (str): 服务器 IP 地址。
        data_port (int): 数据端口号，应与服务器的 data_server() 一致。
        handshake_req (bytes): 客户端发送的握手请求字节。
        handshake_resp (bytes): 期望从服务器收到的握手响应字节。
        recv_exact_fn (callable): 用于从 socket 精确读取指定字节数的函数。
        stop_event (Event): 客户端侧停止推理事件，通过 UI “停止推理”按钮触发。
        input_tensor (Any): 需发送给服务器的输入（通常包含 numpy 张量的字典）。

    Inputs:
        - **server_ip**、**data_port**：用于建立 socket 连接。
        - **handshake_req / handshake_resp**：与服务器保持一致的握手协议。
        - **recv_exact_fn**：必须满足签名 `fn(conn, n, timeout=None)`。
        - **stop_event**：用于在推理过程中临时停止渲染但不立即中断 socket。
        - **input_tensor**：通过 pickle 序列化后发送给服务器的模型输入。

    Outputs:
        这是一个生成器（generator）。
        yield 返回值包括：
            - 文本片段（str）：模型逐段生成的内容
            - "<END>"：服务器推理会话结束标志
            - "[ClientError] xxx"：出现异常时的错误信息

    Raises:
        本函数不会向外抛出异常。
        任何 socket 或反序列化错误都会在 except 块中转化为
        `"[ClientError] ..."` 并以 yield 的方式返回给调用方。
    """
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
