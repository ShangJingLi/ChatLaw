import os
import re
import json
import socket
import time
import uuid
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import soundfile as sf
import librosa
from latex2mathml.converter import convert as latex_to_mathml
from launcher import get_resources_path


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


def normalize_law_name(law_name: str) -> str:
    """
    统一法律名，便于精确匹配。
    例如：
    - 中华人民共和国公司法_20231229 -> 公司法
    - 公司法 -> 公司法
    """
    if not law_name:
        return ""

    s = law_name.strip()
    s = s.replace("《", "").replace("》", "")
    s = s.replace("中华人民共和国", "")
    s = re.sub(r"_[0-9]{8}$", "", s)
    s = re.sub(r"\s+", "", s)
    return s


def normalize_article_number(article_number: str) -> str:
    """
    统一条号格式。
    """
    if not article_number:
        return ""
    return re.sub(r"\s+", "", article_number.strip())


def match_law_name_from_query(query: str, law_name_candidates):
    """
    不再靠正则硬截法律名，而是：
    - 先标准化 query
    - 再在所有已知 law_name_norm 中找“最长子串匹配”
    """
    q = re.sub(r"\s+", "", query.strip())
    q = q.replace("《", "").replace("》", "")
    q = q.replace("中华人民共和国", "")

    for law_name in law_name_candidates:
        if law_name and law_name in q:
            return law_name

    return None


def parse_law_article_query(query: str, law_name_candidates=None):
    """
    尝试从 query 中解析：
    - 法律名
    - 条号

    适用于类似：
    - 公司法第五十八条的内容是什么？
    - 民法商法公司法第五十八条的内容是什么？
    """
    q = re.sub(r"\s+", "", query.strip())

    # 先提取条号
    m_article = re.search(r"(第[一二三四五六七八九十百千万零〇两\d]+条)", q)
    article = normalize_article_number(m_article.group(1)) if m_article else None

    # 再匹配法律名
    law_name = None
    if law_name_candidates:
        law_name = match_law_name_from_query(q, law_name_candidates)

    # 如果候选集合匹配不到，再兜底走旧正则
    if law_name is None:
        m = re.search(r"(?:民法商法)?(.+?法)(第[一二三四五六七八九十百千万零〇两\d]+条)", q)
        if m:
            law_name = normalize_law_name(m.group(1))

    return law_name, article


def load_vectorstore(path):
    embeddings = HuggingFaceEmbeddings(
        model_name=os.path.join(
            get_resources_path(),
            "vectorstore",
            "embedding_model"
        ),
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


def load_exact_index(index_path=None):
    """
    加载精确索引。
    默认路径：
    resources/vectorstore/law_article_index.json
    """
    if index_path is None:
        index_path = os.path.join(
            get_resources_path(),
            "vectorstore",
            "law_article_index.json"
        )

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"精确索引不存在: {index_path}")

    with open(index_path, "r", encoding="utf-8") as f:
        exact_index = json.load(f)

    return exact_index


def build_law_name_candidates(exact_index):
    """
    从精确索引中提取全部 law_name_norm，并按长度从长到短排序。
    这样做“最长子串匹配”更稳。
    """
    law_names = set()

    for _, item in exact_index.items():
        law_name_norm = item.get("law_name_norm", "")
        if law_name_norm:
            law_names.add(law_name_norm)

    return sorted(law_names, key=len, reverse=True)


def retrieve_laws_faiss(vectorstore, query, k=5):
    """
    走 FAISS 语义检索，返回 LangChain Document 列表
    """
    return vectorstore.similarity_search(query, k=k)


def retrieve_laws_exact(exact_index, query, law_name_candidates=None):
    """
    走精确索引检索。
    当前主要面向“某法第几条”的查询。
    返回统一格式的 dict 列表。
    """
    law_name, article = parse_law_article_query(
        query,
        law_name_candidates=law_name_candidates
    )

    if not law_name or not article:
        return []

    key = f"{law_name}::{article}"

    if key in exact_index:
        item = exact_index[key]
        return [item]

    return []


def _doc_unique_key(item):
    """
    用于 exact/faiss 混合后去重
    """
    if hasattr(item, "metadata"):
        law_name = (
            item.metadata.get("law_name_raw")
            or item.metadata.get("law_name")
            or item.metadata.get("law_name_norm")
            or ""
        )
        article = item.metadata.get("article", "")
        return f"{law_name}::{article}"

    if isinstance(item, dict):
        law_name = (
            item.get("law_name_raw")
            or item.get("law_name")
            or item.get("law_name_norm")
            or ""
        )
        article = item.get("article", "")
        return f"{law_name}::{article}"

    return str(item)


def merge_docs_keep_order(*doc_lists):
    """
    按顺序合并多个结果列表，并去重
    """
    merged = []
    seen = set()

    for docs in doc_lists:
        for doc in docs:
            key = _doc_unique_key(doc)
            if key not in seen:
                merged.append(doc)
                seen.add(key)

    return merged


def retrieve_laws(vectorstore,
                  query,
                  exact_index=None,
                  law_name_candidates=None,
                  exact_faiss_k=1,
                  faiss_only_k=5):
    """
    混合检索策略：

    1. 先尝试 exact
    2. 若 exact 命中：
       - 返回 1 条 exact
       - 再补 1 条 faiss 作为参考
    3. 若 exact 未命中：
       - 直接走 faiss，取 5 条

    这样可以兼顾：
    - 法条精确定位
    - 开放场景语义召回

    返回统一的 docs 列表，可直接传给 build_prompt。
    """
    exact_docs = []
    if exact_index is not None:
        exact_docs = retrieve_laws_exact(
            exact_index=exact_index,
            query=query,
            law_name_candidates=law_name_candidates
        )

    # exact 命中：1条 exact + 1条 faiss
    if exact_docs:
        faiss_docs = retrieve_laws_faiss(vectorstore, query, k=exact_faiss_k)
        return merge_docs_keep_order(exact_docs, faiss_docs)

    # exact 未命中：直接 faiss 5条
    return retrieve_laws_faiss(vectorstore, query, k=faiss_only_k)


def _format_doc_block(item, idx: int):
    """
    兼容两种输入：
    1. FAISS 返回的 LangChain Document
    2. 精确索引返回的 dict
    """
    # FAISS Document
    if hasattr(item, "metadata"):
        law_name = (
            item.metadata.get("law_name_raw")
            or item.metadata.get("law_name")
            or item.metadata.get("law_name_norm")
            or ""
        )
        article = item.metadata.get("article", "")
        content = item.metadata.get("content", "") or item.page_content
        return f"{idx}. 《{law_name}》{article}：\n{content}"

    # exact index dict
    if isinstance(item, dict):
        law_name = (
            item.get("law_name_raw")
            or item.get("law_name")
            or item.get("law_name_norm")
            or ""
        )
        article = item.get("article", "")
        content = item.get("content", "")
        return f"{idx}. 《{law_name}》{article}：\n{content}"

    # fallback
    return f"{idx}. {str(item)}"


def build_prompt(question, docs):
    blocks = []

    for i, doc in enumerate(docs, 1):
        blocks.append(_format_doc_block(doc, i))

    context = "\n\n".join(blocks) if blocks else "（未检索到可用法律条文）"

    return f"""你是一个法律咨询专用模型。
你会遇到以下两种情况：
1. 用户的问题与法律条文无关。
2. 用户的问题与法律条文有关。

如果是情况1，则忽略法律条文，直接回答用户问题。
如果是情况2，请结合法律条文回答，并在回答结尾说明你的判断依据，具体到哪部法律及第几条。
如果当前没有检索到合适法条，不要编造法条内容，应基于常识谨慎回答，并明确说明“未检索到直接对应法条”。

【法律条文】（用户不可见）
{context}

【问题】
{question}

【回答】
"""


__all__ = [
    "recv_exact",
    "render_mathml_from_latex",
    "connection_acknowledgement",
    "build_prompt",
    "retrieve_laws",
    "retrieve_laws_faiss",
    "retrieve_laws_exact",
    "merge_docs_keep_order",
    "load_vectorstore",
    "load_exact_index",
    "build_law_name_candidates",
    "normalize_law_name",
    "normalize_article_number",
    "parse_law_article_query",
    "speech_to_text",
]
