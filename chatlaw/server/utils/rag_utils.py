"""服务端知识库检索（RAG）与 prompt 构造工具。

本模块从原客户端 ``common_utils.py`` 迁移而来。重构后，tokenizer、语音转文字、
知识库检索全部下沉到服务端，客户端不再加载任何资源。

设计约束：本模块**不依赖任何推理引擎（vllm / transformers）**，因此可以在没有
GPU / 没有 vllm 的机器上单独导入并测试。
"""
import os
import re
import json

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from launcher import get_resources_path


# ======== 法律名 / 条号标准化 ========
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


# ======== 索引加载 ========
def load_vectorstore(path):
    # 单卡部署：embedding 模型固定跑 CPU，把整块 GPU 显存让给 vLLM 的 LLM。
    # 它只用于给用户的短查询编码做 FAISS 检索（单用户、毫秒级），CPU 足够；
    # 若放 GPU 会与 vLLM 抢显存，导致引擎初始化 CUDA out of memory。
    embeddings = HuggingFaceEmbeddings(
        model_name=os.path.join(
            get_resources_path(),
            "vectorstore",
            "embedding_model"
        ),
        model_kwargs={"device": "cpu"},
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


# ======== 检索 ========
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


# ======== 结果结构统一 ========
def doc_to_dict(item):
    """
    将检索结果（FAISS 的 LangChain Document 或精确索引的 dict）统一成
    可安全通过网络发送、且客户端可直接渲染的纯 dict：

        {"law_name": ..., "article": ..., "content": ...}

    这样做的目的：
    1. 解耦客户端——客户端不再需要 langchain 才能解析 Document；
    2. 网络传输只携带展示所需字段。
    """
    if hasattr(item, "metadata"):
        metadata = item.metadata or {}
        law_name = (
            metadata.get("law_name_raw")
            or metadata.get("law_name")
            or metadata.get("law_name_norm")
            or "未知法律"
        )
        article = metadata.get("article", "")
        content = metadata.get("content", "") or getattr(item, "page_content", "")
    elif isinstance(item, dict):
        law_name = (
            item.get("law_name_raw")
            or item.get("law_name")
            or item.get("law_name_norm")
            or "未知法律"
        )
        article = item.get("article", "")
        content = item.get("content", "")
    else:
        law_name, article, content = "未知法律", "", str(item)

    return {"law_name": law_name, "article": article, "content": content}


# ======== prompt 构造 ========
def _format_doc_block(item, idx: int):
    """
    兼容三种输入：
    1. FAISS 返回的 LangChain Document
    2. 精确索引返回的 dict
    3. doc_to_dict 归一化后的 dict
    """
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
    "normalize_law_name",
    "normalize_article_number",
    "match_law_name_from_query",
    "parse_law_article_query",
    "load_vectorstore",
    "load_exact_index",
    "build_law_name_candidates",
    "retrieve_laws_faiss",
    "retrieve_laws_exact",
    "merge_docs_keep_order",
    "retrieve_laws",
    "doc_to_dict",
    "build_prompt",
]
