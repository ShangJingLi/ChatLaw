"""
法律 RAG 推理脚本
"""
import os.path
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM
from launcher import get_project_root


# ========== 1. 加载向量库 ==========
def load_vectorstore(path):
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-zh-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


# ========== 2. 检索法律条文 ==========
def retrieve_laws(vectorstore, query, k=5):
    return vectorstore.similarity_search(query, k=k)


# ========== 3. 构造 Prompt ==========
def build_prompt(question, docs):
    blocks = []

    for i, doc in enumerate(docs, 1):
        blocks.append(
            f"{i}. 《{doc.metadata['law_name']}》"
            f"{doc.metadata['article']}：\n"
            f"{doc.page_content}"
        )

    context = "\n\n".join(blocks)

    return f"""如果问题与下列法律问题无关则直接回答；如果有关请结合法律条文回答，并在回答结尾说明具体哪部法律以及第几条

【法律条文】
{context}

【问题】
{question}

【回答】
"""


# ========== 4. 调用 Qwen ==========
def qwen_generate(prompt):
    model_name = "Qwen/Qwen3-4B-Instruct-2507"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=2048
    )

    result = outputs[0][len(inputs.input_ids[0]):]
    return tokenizer.decode(result, skip_special_tokens=True)


# ========== 5. 主流程 ==========
if __name__ == "__main__":
    vectorstore = load_vectorstore(os.path.join(get_project_root(), "resources", "vectorstore"))

    question = "离婚后未成年子女的抚养权如何确定？"
    start = time.time()
    docs = retrieve_laws(vectorstore, question)
    end = time.time()
    print("检索时长：" + str(round(end - start, 2)) + "秒")

    prompt = build_prompt(question, docs)
    answer = qwen_generate(prompt)

    print(answer)
