import re
import socket
import time

from latex2mathml.converter import convert as latex_to_mathml


def recv_exact(conn, n, timeout=None):
    """
    从 socket 连接中精确读取 n 字节数据。
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
    """Markdown 中的 $ 和 $$ 公式转换为 MathML"""

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
    建立一次短连接，用于验证服务器是否正常运行。
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


__all__ = ["recv_exact",
           "render_mathml_from_latex",
           "connection_acknowledgement"]
