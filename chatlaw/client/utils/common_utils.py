import os
import re
import socket
import time
import uuid
import numpy as np
import soundfile as sf
import librosa
from latex2mathml.converter import convert as latex_to_mathml


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
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed while receiving")
        buf += chunk
    return buf


def render_mathml_from_latex(md_text: str) -> str:
    """
    功能：
        将 Markdown 文本中的 LaTeX 数学公式（包括行内公式 `$...$` 与块级公式
        `$$...$$`）转换为对应的 MathML 表达形式，便于在前端浏览器中渲染数学内容。
        若转换失败，则以 `<code>` 或 `<pre>` 标签形式回退显示原始 LaTeX 代码，
        以保证渲染过程的健壮性。

    Args:
        md_text (str): 输入的 Markdown 文本，其中可能包含 LaTeX 行内公式与块级公式。

    Inputs:
        - **md_text** (str): 原始 Markdown 文本，可能含有：
              - `$ ... $`    行内公式
              - `$$ ... $$`  块级公式
          函数内部依赖：
              - **latex_to_mathml**: 将 LaTeX 文本转换为 MathML 的转换函数。
              - **re**: 用于匹配并替换公式的正则表达式模块。

    Outputs:
        str: 返回替换完成的 HTML + MathML 字符串。其中：
             - 行内公式替换为 `<math>...</math>`（通过 latex_to_mathml 转换）
             - 块级公式替换为 `<div>...</div>`
             - 若转换失败，则保留原有公式并以 `<code>` 或 `<pre>` 包裹

    Raises:
        本函数不向外抛出异常。任何转换错误都会在内部捕获并采用回退方案。
    """
    def repl_inline(m):
        try:
            return latex_to_mathml(m.group(1))
        except:
            return f"<code>{m.group(1)}</code>"

    def repl_block(m):
        try:
            return f"<div>{latex_to_mathml(m.group(1))}</div>"
        except:
            return f"<pre>{m.group(1)}</pre>"

    text = re.sub(r"\$\$(.+?)\$\$", repl_block, md_text, flags=re.S)
    text = re.sub(r"\$(.+?)\$", repl_inline, text)
    return text


def connection_acknowledgement(server_ip, data_port,
                                  handshake_req, handshake_resp,
                                  recv_exact_fn, start_time):
    """
    功能：
        建立一次短连接，用于快速检测服务器的数据端口是否正常工作。
        函数会向服务器发送握手请求字节，并验证服务器返回的响应是否正确。
        若握手成功，则说明服务器运行正常；否则视为连接失败。
        该检测通常用于在正式推理前确认服务器在线状态。

    Args:
        server_ip (str): 服务器的 IP 地址。
        data_port (int): 数据服务监听端口。
        handshake_req (bytes): 客户端发送给服务器的握手请求字节。
        handshake_resp (bytes): 客户端期望从服务器收到的握手响应字节。
        recv_exact_fn (callable): 准确读取指定字节数的函数，用于接收握手响应。
        start_time (float): 调用开始时间戳，用于计算连接耗时。

    Inputs:
        - **server_ip**: 服务器主机地址。
        - **data_port**: 用于测试连接的数据端口。
        - **handshake_req**: 握手机制中发送的标记字节。
        - **handshake_resp**: 握手机制中预期收到的字节。
        - **recv_exact_fn**: 必须具有签名 `fn(conn, n, timeout=None)`。
        - **start_time**: 用于计算连接耗时（秒）。

    Outputs:
        tuple(str, bool):
            - 第一个元素为状态信息字符串（成功或失败信息）；
            - 第二个元素为布尔值：
                - True：连接验证成功；
                - False：连接失败。

    Raises:
        函数内部捕获所有异常，不会向外抛出。
        若连接或握手过程中出现错误，会返回 (错误信息, False)。
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3.0)
        s.connect((server_ip, data_port))
        s.settimeout(1.0)

        s.sendall(handshake_req)
        resp = recv_exact_fn(s, 1, timeout=2.0)

        if resp != handshake_resp:
            raise ConnectionError("Handshake failed")

        cost = time.time() - start_time
        return f"✅ 建立连接成功，用时 {cost:.2f}s", True

    except Exception as e:
        return f"⚠️ 通信失败：{e}", False

    finally:
        try:
            s.close()
        except:
            pass


def speech_to_text(audio, target_sr, audio_cache_dir, audio_model):
    """
    audio: (sample_rate, numpy_array)
    return: 中文文本
    """
    if audio is None:
        return "未检测到语音输入"

    sr, data = audio

    # ---------- 1. 转单声道 ----------
    if data.ndim > 1:
        data = data.mean(axis=1)

    data = data.astype(np.float32)

    # ---------- 2. 重采样到 16k ----------
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)

    # ---------- 3. 幅值归一化（防止过小） ----------
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val * 0.9

    # ---------- 4. 时长校验 ----------
    duration = len(data) / target_sr
    if duration < 0.8:
        return "语音过短，请说完整一句话"

    # ---------- 5. 写入固定目录 wav（避免 tempfile 坑） ----------
    wav_name = f"{uuid.uuid4().hex}.wav"
    wav_path = os.path.join(audio_cache_dir, wav_name)

    sf.write(wav_path, data, target_sr, subtype="PCM_16")

    # 确保文件完全落盘（Windows 必须）
    time.sleep(0.05)

    # ---------- 6. 调用 Paraformer ONNX ----------
    try:
        result = audio_model([wav_path])
    except Exception as e:
        return f"ASR 推理异常: {e}"
    finally:
        try:
            os.remove(wav_path)
        except Exception:
            pass

    # ---------- 7. 正确解析 funasr-onnx 返回 ----------
    # 返回格式示例：
    # [{'preds': ('如何理解等差数列', ['如','何','理','解','等','差','数','列'])}]
    if not result or not isinstance(result, list):
        return "未识别到有效内容"

    preds = result[0].get("preds", None)
    if not preds or not isinstance(preds, tuple):
        return "未识别到有效内容"

    text = preds[0].strip()
    if not text:
        raise ValueError("语音未识别到有效内容！")

    return text


__all__ = ["recv_exact",
           "render_mathml_from_latex",
           "connection_acknowledgement"]
