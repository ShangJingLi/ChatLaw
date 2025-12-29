"""
法律 RAG 推理脚本
"""
import os.path
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM
from launcher import get_project_root

model_name = "Qwen/Qwen3-4B-Instruct-2507"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto"
)
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
def retrieve_laws(vectorstore, query, k=10):
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

    return f"""你会遇到以下两种情况：1.用户的问题与法律问题无关。 2.用户的问题与法律问题有关
               如果是情况1，则忽略法律条文，直接回答用户问题
               如果是情况2，请结合法律条文回答，并在回答结尾说明具体哪部法律及第几条

【法律条文】
{context}

【问题】
{question}

【回答】
"""


# ========== 4. 调用 Qwen ==========
def qwen_generate(prompt):
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
    start = time.time()
    vectorstore = load_vectorstore(os.path.join(get_project_root(), "resources", "vectorstore"))
    end = time.time()
    print("向量库加载时长:" + str(round(end - start, 2)) + "秒")

    question = "你好！"
    start = time.time()
    docs = retrieve_laws(vectorstore, question)
    end = time.time()
    print("检索时长：" + str(round(end - start, 2)) + "秒")

    prompt = build_prompt(question, docs)
    print(prompt)
    answer = qwen_generate(prompt)
    print(answer)

    question = "小明操纵期货交易需要承担什么法律责任？"
    start = time.time()
    docs = retrieve_laws(vectorstore, question)
    end = time.time()
    print("检索时长：" + str(round(end - start, 2)) + "秒")

    prompt = build_prompt(question, docs)
    print(prompt)
    answer = qwen_generate(prompt)
    print(answer)
