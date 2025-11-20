# utils_pt.py

import pickle
import socket
import time


def heartbeat_client(server_ip, hb_port, hb_interval, hb_timeout,
                     alive_flag, stop_event, recv_exact_fn):
    """
    功能：
        实现客户端侧的心跳检测线程，用于维持与服务器的长连接活性检测。
        主要行为如下：
        1. 周期性向服务器发送 `PING`；
        2. 服务器返回 `PONG` 表示连接正常；
        3. 若超过 hb_timeout 未收到有效 PONG，则认为服务器已掉线；
        4. 若 UI 或上层逻辑触发 stop_event，则向服务器发送 `STOP` 请求终止推理；
        5. 心跳线程始终保持轻量级，不负责推理或数据接收，只负责连接存活性检测。

    Args:
        server_ip (str): 服务器 IP 地址。
        hb_port (int): 心跳端口号，需与服务器 heartbeat_server 保持一致。
        hb_interval (float): 两次发送 `PING` 之间的间隔（秒）。
        hb_timeout (float): 若此时间内未收到 `PONG`，视为心跳超时。
        alive_flag (callable): 返回布尔值的函数，用于决定心跳线程是否继续运行。
        stop_event (Event): 如果被触发，向服务器发送 STOP 并终止心跳线程。
        recv_exact_fn (callable): 精确从 socket 读取指定字节数的函数。

    Inputs:
        - **server_ip**、**hb_port**：连接服务器所需的信息。
        - **hb_interval / hb_timeout**：控制心跳频率与超时策略。
        - **alive_flag()**：来自 UI 或主线程，用于终止心跳。
        - **stop_event**：用于中断推理，会被双方共享。
        - **recv_exact_fn**：必须为一个函数，签名 `(conn, n, timeout=None)`，用于精确读取 PONG。

    Outputs:
        无显式返回值。
        该函数在内部维持心跳连接，随状态变化自动退出。
        当检测到服务器掉线时，会触发 stop_event 供其他线程感知。

    Raises:
        无向外抛出的异常。
        所有 socket 和心跳过程中的异常均被内部捕获，最终心跳线程只会安全退出。
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


def stream_from_server(server_ip, data_port,
                       handshake_req, handshake_resp,
                       recv_exact_fn,
                       alive_flag, stop_event,
                       input_tensor):
    """
    功能：
        与服务器建立数据连接，发送模型输入，并通过流式协议逐段接收服务器生成的
        模型输出内容。本函数实现了流式推理机制，支持推理中断、
        停止信号处理和断线自动检测，确保客户端在各种异常情况下保持健壮性。

        特殊设计要点：
        1. 当用户触发 STOP（stop_event.set()）后，不再向 UI 输出新的内容，
           但仍继续接收服务器数据，直到收到 `<END>`，保持客户端与服务器状态一致；
        2. 若连续多次超时（MAX_TIMEOUT 次）未收到任何数据，则视为服务器掉线，
           以 `[ClientError]` 格式返回错误；
        3. 当服务器返回 `<END>` 时，表示会话结束，生成器退出。

    Args:
        server_ip (str): 服务器的 IP 地址。
        data_port (int): 数据端口号，需与服务器端 data_server 保持一致。
        handshake_req (bytes): 客户端握手请求字节。
        handshake_resp (bytes): 期望从服务器获得的握手响应字节。
        recv_exact_fn (callable): 精确读取指定字节数的函数。
        alive_flag (callable): 返回布尔值，用于决定是否继续读取服务器数据。
        stop_event (Event): 客户端停止推理事件，可由 UI 或心跳机制设置。
        input_tensor (Any): 由 tokenizer 生成的 numpy 张量或输入字典，将发送给服务器。

    Inputs:
        - **socket连接参数**：server_ip、data_port。
        - **握手协议**：handshake_req 与 handshake_resp。
        - **recv_exact_fn**：签名为 `(conn, n, timeout=None)`。
        - **alive_flag()**：控制流式会话继续执行或终止。
        - **stop_event**：控制 UI 停止渲染但保持读取。
        - **input_tensor**：经过 pickle 序列化后发送给服务器的输入数据。

    Outputs:
        本函数为生成器，逐段输出：
            - chunk（str）：模型生成的部分文本；
            - "<END>"：服务器推理结束标识；
            - "[ClientError] xxx"：发生异常时的错误信息。

        注意：stop_event 被触发后，仍会继续接收 chunk（但不 yield 给 UI），直至 `<END>`。

    Raises:
        本函数不会向外抛出异常。
        所有错误均转换成：
            yield "[ClientError] xxx"
        的形式返回给调用端。
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
