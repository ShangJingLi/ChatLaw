"""客户端通用工具（精简版）。

重构后客户端只负责：采集音频 / 文本、通过 socket 与服务端交互、渲染服务端返回结果。
因此本模块只保留：
    - recv_exact          : 精确读取指定字节数（协议用）
    - render_mathml_from_latex : 把 LaTeX 公式渲染成 MathML（展示用）

语音转文字、知识库检索、prompt 构造等全部已迁移到服务端。
"""
import re

from latex2mathml.converter import convert as latex_to_mathml


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


def render_mathml_from_latex(md_text: str) -> str:
    """
    把文本中的 LaTeX 行内公式 `$...$` 与块级公式 `$$...$$` 转成 MathML，
    便于浏览器渲染。转换失败时回退为 <code>/<pre> 原样显示。
    """
    def repl_inline(m):
        try:
            return latex_to_mathml(m.group(1))
        except Exception:  # pylint: disable=broad-exception-caught
            return f"<code>{m.group(1)}</code>"

    def repl_block(m):
        try:
            return f"<div>{latex_to_mathml(m.group(1))}</div>"
        except Exception:  # pylint: disable=broad-exception-caught
            return f"<pre>{m.group(1)}</pre>"

    text = re.sub(r"\$\$(.+?)\$\$", repl_block, md_text, flags=re.S)
    text = re.sub(r"\$(.+?)\$", repl_inline, text)
    return text


__all__ = ["recv_exact", "render_mathml_from_latex"]
