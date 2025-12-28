import os
import json
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_law_json_dir(json_dir):
    docs = []

    for filename in os.listdir(json_dir):
        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(json_dir, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        law_name = data.get("law_name", filename.replace(".json", ""))
        articles = data.get("articles", [])

        for art in articles:
            article_number = art.get("article_number", "").strip()
            content = art.get("content", "").strip()

            if not content:
                continue

            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "law_name": law_name,
                        "article": article_number
                    }
                )
            )

    return docs


def build_embeddings():
    """
    embeddings 本质是：text -> vector 的 callable
    """
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-zh-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )


def build_and_save_vectorstore(
    json_dir: str,
    save_dir: str
):
    docs = load_law_json_dir(json_dir)
    print(f"共加载 {len(docs)} 条法律条文")

    embeddings = build_embeddings()

    vectorstore = FAISS.from_documents(
        documents=docs,
        embedding=embeddings
    )

    vectorstore.save_local(save_dir)
    print(f"向量库已保存至：{save_dir}")
