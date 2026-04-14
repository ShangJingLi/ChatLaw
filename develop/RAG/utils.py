import os
import re
import json
from typing import List, Dict, Tuple, Optional

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from launcher import get_resources_path


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
    s = re.sub(r"_[0-9]{8}$", "", s)  # 去掉 _20231229
    s = re.sub(r"\s+", "", s)
    return s


def normalize_article_number(article_number: str) -> str:
    """
    统一条号格式。
    """
    if not article_number:
        return ""
    return re.sub(r"\s+", "", article_number.strip())


def load_law_json_dir(json_dir: str) -> Tuple[List[Document], Dict[str, dict], Dict[str, List[dict]]]:
    """
    读取清洗后的法律 JSON 目录，同时构造：

    1. LangChain Document 列表（给 FAISS 用）
    2. 精确索引：
       key = "{law_name_norm}::{article}"
    3. 按法律名分组索引：
       key = law_name_norm

    返回：
    - docs
    - exact_index
    - law_name_index
    """
    docs: List[Document] = []
    exact_index: Dict[str, dict] = {}
    law_name_index: Dict[str, List[dict]] = {}

    if not os.path.isdir(json_dir):
        raise FileNotFoundError(f"json_dir 不存在或不是目录: {json_dir}")

    for filename in sorted(os.listdir(json_dir)):
        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(json_dir, filename)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠️ 跳过读取失败文件: {file_path} | {e}")
            continue

        law_name_raw = str(data.get("law_name", filename.replace(".json", ""))).strip()
        law_name_norm = normalize_law_name(law_name_raw)
        articles = data.get("articles", [])

        if not isinstance(articles, list):
            print(f"⚠️ 跳过结构异常文件: {file_path} | articles 不是列表")
            continue

        for art in articles:
            if not isinstance(art, dict):
                continue

            article_number = normalize_article_number(str(art.get("article_number", "")))
            content = str(art.get("content", "")).strip()

            if not article_number or not content:
                continue

            # =========================
            # 1) 给向量检索的文本
            # =========================
            text_for_embedding = (
                f"法律名称：{law_name_raw}\n"
                f"法律简称：{law_name_norm}\n"
                f"条号：{article_number}\n"
                f"正文：{content}"
            )

            docs.append(
                Document(
                    page_content=text_for_embedding,
                    metadata={
                        "law_name_raw": law_name_raw,
                        "law_name_norm": law_name_norm,
                        "article": article_number,
                        "content": content,
                    }
                )
            )

            # =========================
            # 2) 精确索引
            # =========================
            exact_key = f"{law_name_norm}::{article_number}"
            item = {
                "law_name_raw": law_name_raw,
                "law_name_norm": law_name_norm,
                "article": article_number,
                "content": content,
            }
            exact_index[exact_key] = item

            # =========================
            # 3) 按法律名分组索引
            # =========================
            law_name_index.setdefault(law_name_norm, []).append(item)

    return docs, exact_index, law_name_index


def build_embeddings() -> HuggingFaceEmbeddings:
    """
    构建 embedding 模型。
    """
    model_path = os.path.join(
        get_resources_path(),
        "vectorstore",
        "embedding_model"
    )

    return HuggingFaceEmbeddings(
        model_name=model_path,
        encode_kwargs={"normalize_embeddings": True}
    )


def build_and_save_knowledge_base(json_dir: str, save_dir: str) -> None:
    """
    一次性构建并保存：
    1. FAISS 向量库
    2. 精确索引 exact index
    3. 法律名分组索引 law_name index
    """
    docs, exact_index, law_name_index = load_law_json_dir(json_dir)
    print(f"共加载 {len(docs)} 条法律条文")

    if not docs:
        raise ValueError("未加载到任何有效法条，无法构建知识库。")

    os.makedirs(save_dir, exist_ok=True)

    # =========================
    # 保存 FAISS
    # =========================
    embeddings = build_embeddings()
    vectorstore = FAISS.from_documents(
        documents=docs,
        embedding=embeddings
    )
    vectorstore_path = os.path.join(save_dir, "law_faiss")
    os.makedirs(vectorstore_path, exist_ok=True)
    vectorstore.save_local(vectorstore_path)
    print(f"✅ 向量库已保存至：{vectorstore_path}")

    # =========================
    # 保存精确索引
    # =========================
    exact_index_path = os.path.join(save_dir, "law_article_index.json")
    with open(exact_index_path, "w", encoding="utf-8") as f:
        json.dump(exact_index, f, ensure_ascii=False, indent=2)
    print(f"✅ 精确索引已保存至：{exact_index_path}")

    # =========================
    # 保存法律名分组索引
    # =========================
    law_name_index_path = os.path.join(save_dir, "law_name_index.json")
    with open(law_name_index_path, "w", encoding="utf-8") as f:
        json.dump(law_name_index, f, ensure_ascii=False, indent=2)
    print(f"✅ 法律名索引已保存至：{law_name_index_path}")