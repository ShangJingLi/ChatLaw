# -*- coding: utf-8 -*-

"""
评估脚本：支持 4 种实验
1. pretrain       预训练模型
2. finetune       微调模型
3. pretrain_rag   预训练模型 + RAG
4. finetune_rag   微调模型 + RAG

使用方式：
- 每台机器只需要修改 EXPERIMENTS 列表，保留自己要跑的实验
- 例如 A 机器跑 pretrain / pretrain_rag
- B 机器跑 finetune / finetune_rag

输出内容：
1. 每个实验一个 results.json，保存逐条结果：
   {
      "question": "...",
      "prediction": "...",
      "gold": "...",
      "em": ...,
      "f1": ...,
      "rouge_l": ...,
      "citation": ...,
      "hallucination": ...,
      ...
   }

2. 一个总 summary_metrics.json，格式类似：
   {
      "pretrain": {...},
      "finetune": {...},
      "pretrain_rag": {...},
      "finetune_rag": {...}
   }
"""

from __future__ import annotations

import os
import re
import json
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# =========================================================
# 需要你自己填写/修改的参数：全部放这里
# =========================================================

# 测试集
TEST_FILE = r"develop/resources/finetuning_dataset/test.json"

# 预训练模型目录
BASE_MODEL_PATH = r"chatlaw/resources/llm"

# tokenizer 目录
TOKENIZER_PATH = r"chatlaw/resources/tokenizer"

# LoRA 目录
LORA_ADAPTER_PATH = r"chatlaw/resources/llm/lora_adapter"

# 知识库目录
VECTORSTORE_PATH = r"chatlaw/resources/vectorstore"

# embedding 模型目录
EMBEDDING_MODEL_PATH = os.path.join(VECTORSTORE_PATH, "embedding_model")

# 精确索引
EXACT_INDEX_PATH = os.path.join(VECTORSTORE_PATH, "law_article_index.json")

# 输出根目录
OUTPUT_ROOT = r"develop/resources/eval_outputs_rebuilt"

# 混合检索参数
# exact 命中时：1 条 exact + 1 条 faiss
EXACT_FAISS_K = 1

# exact 未命中时：faiss 取 5 条
FAISS_ONLY_K = 5

# 生成参数
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.0
TOP_P = 1.0
DO_SAMPLE = False

# batch size
BATCH_SIZE = 4

# 已存在结果是否跳过
SKIP_IF_EXISTS = False

# 随机种子
SEED = 42

# 设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# dtype
TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


# =========================================================
# 实验配置
# 每台机器只保留自己要跑的
# =========================================================

@dataclass
class ExperimentConfig:
    system_name: str
    use_rag: bool
    use_lora: bool


EXPERIMENTS = [
    # 预训练模型
    ExperimentConfig(
        system_name="pretrain",
        use_rag=False,
        use_lora=False,
    ),

    # # 微调模型
    # ExperimentConfig(
    #     system_name="finetune",
    #     use_rag=False,
    #     use_lora=True,
    # ),
    #
    # # 预训练模型 + RAG
    # ExperimentConfig(
    #     system_name="pretrain_rag",
    #     use_rag=True,
    #     use_lora=False,
    # ),
    #
    # # 微调模型 + RAG
    # ExperimentConfig(
    #     system_name="finetune_rag",
    #     use_rag=True,
    #     use_lora=True,
    # ),
]


# =========================================================
# 基础工具
# =========================================================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =========================================================
# 数据集读取
# =========================================================

def load_dataset(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("测试集应为 list[dict] 格式")

    valid_data = []
    for item in data:
        if not isinstance(item, dict):
            continue

        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        reference = item.get("reference", [])

        if not question or not answer:
            continue

        if reference is None:
            reference = []
        elif not isinstance(reference, list):
            reference = [reference]

        valid_data.append({
            "question": question,
            "answer": answer,
            "reference": reference,
        })

    return valid_data


# =========================================================
# 检索逻辑
# =========================================================

def normalize_law_name(law_name: str) -> str:
    if not law_name:
        return ""

    s = law_name.strip()
    s = s.replace("《", "").replace("》", "")
    s = s.replace("中华人民共和国", "")
    s = re.sub(r"_[0-9]{8}$", "", s)
    s = re.sub(r"\s+", "", s)
    return s


def normalize_article_number(article_number: str) -> str:
    if not article_number:
        return ""
    return re.sub(r"\s+", "", article_number.strip())


def match_law_name_from_query(query: str, law_name_candidates):
    q = re.sub(r"\s+", "", query.strip())
    q = q.replace("《", "").replace("》", "")
    q = q.replace("中华人民共和国", "")

    for law_name in law_name_candidates:
        if law_name and law_name in q:
            return law_name

    return None


def parse_law_article_query(query: str, law_name_candidates=None):
    q = re.sub(r"\s+", "", query.strip())

    m_article = re.search(r"(第[一二三四五六七八九十百千万零〇两\d]+条)", q)
    article = normalize_article_number(m_article.group(1)) if m_article else None

    law_name = None
    if law_name_candidates:
        law_name = match_law_name_from_query(q, law_name_candidates)

    if law_name is None:
        m = re.search(r"(?:民法商法)?(.+?法)(第[一二三四五六七八九十百千万零〇两\d]+条)", q)
        if m:
            law_name = normalize_law_name(m.group(1))

    return law_name, article


def load_vectorstore(path: str):
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        encode_kwargs={"normalize_embeddings": True},
    )

    faiss_dir = os.path.join(path, "law_faiss")
    load_path = faiss_dir if os.path.isdir(faiss_dir) else path

    vectorstore = FAISS.load_local(
        load_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


def load_exact_index(index_path=None):
    if index_path is None:
        index_path = EXACT_INDEX_PATH

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"精确索引不存在: {index_path}")

    with open(index_path, "r", encoding="utf-8") as f:
        exact_index = json.load(f)

    return exact_index


def build_law_name_candidates(exact_index):
    law_names = set()

    for _, item in exact_index.items():
        law_name_norm = item.get("law_name_norm", "")
        if law_name_norm:
            law_names.add(law_name_norm)

    return sorted(law_names, key=len, reverse=True)


def retrieve_laws_faiss(vectorstore, query, k=5):
    return vectorstore.similarity_search(query, k=k)


def retrieve_laws_exact(exact_index, query, law_name_candidates=None):
    law_name, article = parse_law_article_query(
        query,
        law_name_candidates=law_name_candidates
    )

    if not law_name or not article:
        return []

    key = f"{law_name}::{article}"

    if key in exact_index:
        return [exact_index[key]]

    return []


def _doc_unique_key(item):
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
    merged = []
    seen = set()

    for docs in doc_lists:
        for doc in docs:
            key = _doc_unique_key(doc)
            if key not in seen:
                merged.append(doc)
                seen.add(key)

    return merged


def retrieve_laws(
    vectorstore,
    query,
    exact_index=None,
    law_name_candidates=None,
    exact_faiss_k=1,
    faiss_only_k=5
):
    """
    混合检索策略：
    1. 先尝试 exact
    2. 若 exact 命中：1 条 exact + 1 条 faiss
    3. 若 exact 未命中：faiss 取 5 条
    """
    exact_docs = []
    if exact_index is not None:
        exact_docs = retrieve_laws_exact(
            exact_index=exact_index,
            query=query,
            law_name_candidates=law_name_candidates
        )

    if exact_docs:
        faiss_docs = retrieve_laws_faiss(vectorstore, query, k=exact_faiss_k)
        return merge_docs_keep_order(exact_docs, faiss_docs), "exact+faiss"

    return retrieve_laws_faiss(vectorstore, query, k=faiss_only_k), "faiss_only"


def _format_doc_block(item, idx: int):
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


def build_prompt(question: str, docs: Optional[List[Any]] = None) -> str:
    if docs:
        blocks = []
        for i, doc in enumerate(docs, 1):
            blocks.append(_format_doc_block(doc, i))
        context = "\n\n".join(blocks)

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
""".strip()

    return f"""你是一个专业法律助手。请直接回答用户问题。
如果涉及法律依据，尽量给出具体法律及条文；如果无法确认，不要编造。

【问题】
{question}

【回答】
""".strip()


def serialize_docs(docs: List[Any]) -> List[Dict[str, Any]]:
    out = []

    for doc in docs:
        if hasattr(doc, "metadata"):
            out.append({
                "source_type": "faiss",
                "law_name": (
                    doc.metadata.get("law_name_raw")
                    or doc.metadata.get("law_name")
                    or doc.metadata.get("law_name_norm")
                    or ""
                ),
                "article": doc.metadata.get("article", ""),
                "content": doc.metadata.get("content", "") or getattr(doc, "page_content", ""),
            })
        elif isinstance(doc, dict):
            out.append({
                "source_type": "exact",
                "law_name": (
                    doc.get("law_name_raw")
                    or doc.get("law_name")
                    or doc.get("law_name_norm")
                    or ""
                ),
                "article": doc.get("article", ""),
                "content": doc.get("content", ""),
            })
        else:
            out.append({
                "source_type": "unknown",
                "text": str(doc),
            })

    return out


# =========================================================
# 模型封装
# =========================================================

class LocalQwenGenerator:
    def __init__(
        self,
        tokenizer_path: str,
        base_model_path: str,
        lora_adapter_path: Optional[str] = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,
            trust_remote_code=True,
            padding_side="left",
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=TORCH_DTYPE,
        ).to(DEVICE)

        if lora_adapter_path is not None:
            print(f"正在加载 LoRA: {lora_adapter_path}")
            peft_model = PeftModel.from_pretrained(
                base_model,
                lora_adapter_path,
                local_files_only=True,
            )
            print("正在合并 LoRA 权重到基座模型...")
            self.model = peft_model.merge_and_unload()
            print("LoRA 权重合并完成。")
        else:
            self.model = base_model

        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def generate_batch(self, prompts: List[str]) -> List[str]:
        texts = []

        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            texts.append(text)

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        input_len = inputs.input_ids.shape[1]
        outputs = []
        for i in range(len(generated_ids)):
            output_ids = generated_ids[i][input_len:]
            text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            outputs.append(text)

        return outputs


# =========================================================
# 指标
# =========================================================

def normalize_text(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "", s)
    s = s.replace("，", ",").replace("。", ".").replace("：", ":")
    s = s.replace("；", ";").replace("？", "?").replace("！", "!")
    return s


def tokenize_for_f1(s: str) -> List[str]:
    s = normalize_text(s)
    return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+", s)


def exact_match_score(pred: str, gold: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = tokenize_for_f1(pred)
    gold_tokens = tokenize_for_f1(gold)

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    from collections import Counter
    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    common = pred_counter & gold_counter
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def lcs_len(x: List[str], y: List[str]) -> int:
    if not x or not y:
        return 0
    if len(y) > len(x):
        x, y = y, x

    prev = [0] * (len(y) + 1)
    for i in range(1, len(x) + 1):
        curr = [0] * (len(y) + 1)
        for j in range(1, len(y) + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def rouge_l_score(pred: str, gold: str) -> float:
    pred_tokens = tokenize_for_f1(pred)
    gold_tokens = tokenize_for_f1(gold)

    if not pred_tokens or not gold_tokens:
        return 0.0

    lcs = lcs_len(pred_tokens, gold_tokens)
    prec = lcs / len(pred_tokens)
    rec = lcs / len(gold_tokens)

    if prec + rec == 0:
        return 0.0

    beta = 1.2
    return ((1 + beta ** 2) * prec * rec) / (rec + beta ** 2 * prec)


def extract_article_mentions(text: str) -> set:
    return set(re.findall(r"第[一二三四五六七八九十百千万零〇两\d]+条", str(text)))


def extract_law_mentions(text: str) -> set:
    text = str(text)
    laws_1 = set(re.findall(r"《([^》]{1,50}?法)》", text))
    laws_2 = set(re.findall(r"(?<![《\w])([\u4e00-\u9fff]{2,50}?法)(?![\u4e00-\u9fff])", text))
    laws = set()
    for x in laws_1 | laws_2:
        laws.add(normalize_law_name(x))
    return laws


def build_supported_citations_from_references(references: List[str]) -> Tuple[set, set]:
    supported_laws = set()
    supported_articles = set()

    for ref in references:
        ref = str(ref)
        supported_laws |= extract_law_mentions(ref)
        supported_articles |= extract_article_mentions(ref)

    return supported_laws, supported_articles


def citation_accuracy_and_hallucination(pred: str, references: List[str]) -> Tuple[float, float]:
    pred_laws = extract_law_mentions(pred)
    pred_articles = extract_article_mentions(pred)

    if len(pred_laws) == 0 and len(pred_articles) == 0:
        return 0.0, 0.0

    supported_laws, supported_articles = build_supported_citations_from_references(references)

    total = len(pred_laws) + len(pred_articles)
    hit = 0

    for x in pred_laws:
        if x in supported_laws:
            hit += 1

    for x in pred_articles:
        if x in supported_articles:
            hit += 1

    citation_acc = hit / total if total > 0 else 0.0
    hallucination = 1.0 - citation_acc
    return citation_acc, hallucination


def retrieval_recall_at_k(references: List[str], retrieved_docs: List[Any]) -> Optional[float]:
    if not retrieved_docs:
        return 0.0

    gold_laws, gold_articles = build_supported_citations_from_references(references)
    if not gold_laws and not gold_articles:
        return None

    retrieved_text = "\n".join(
        _format_doc_block(doc, i)
        for i, doc in enumerate(retrieved_docs, start=1)
    )

    hit = 0
    total = len(gold_laws) + len(gold_articles)

    normalized_retrieved = normalize_text(retrieved_text)

    for law in gold_laws:
        if law and law in normalized_retrieved:
            hit += 1

    for article in gold_articles:
        if article and article in retrieved_text:
            hit += 1

    if total == 0:
        return None

    return hit / total


# =========================================================
# 单实验运行
# =========================================================

def evaluate_one_experiment(
    exp_cfg: ExperimentConfig,
    dataset: List[Dict[str, Any]],
    vectorstore=None,
    exact_index=None,
    law_name_candidates=None,
):
    print("\n" + "=" * 80)
    print(f"开始运行实验: {exp_cfg.system_name}")
    print(f"use_rag = {exp_cfg.use_rag}")
    print(f"use_lora = {exp_cfg.use_lora}")
    print("=" * 80)

    out_dir = ensure_dir(os.path.join(OUTPUT_ROOT, exp_cfg.system_name))
    results_path = os.path.join(out_dir, "results.json")
    summary_path = os.path.join(out_dir, "summary_metrics.json")

    if SKIP_IF_EXISTS and os.path.exists(results_path):
        print(f"跳过已有结果: {results_path}")
        if os.path.exists(summary_path):
            return load_json(summary_path)
        return None

    lora_path = LORA_ADAPTER_PATH if exp_cfg.use_lora else None

    generator = LocalQwenGenerator(
        tokenizer_path=TOKENIZER_PATH,
        base_model_path=BASE_MODEL_PATH,
        lora_adapter_path=lora_path,
    )

    results = []
    batch_prompts = []
    batch_meta = []

    iterator = tqdm(dataset, desc=exp_cfg.system_name, ncols=100)

    start_time = time.time()

    for idx, sample in enumerate(iterator):
        question = sample["question"]
        gold = sample["answer"]
        references = sample["reference"]

        retrieved_docs = []
        retrieval_mode = None

        if exp_cfg.use_rag:
            retrieved_docs, retrieval_mode = retrieve_laws(
                vectorstore=vectorstore,
                query=question,
                exact_index=exact_index,
                law_name_candidates=law_name_candidates,
                exact_faiss_k=EXACT_FAISS_K,
                faiss_only_k=FAISS_ONLY_K
            )

        origin_prompt = build_prompt(question, retrieved_docs if exp_cfg.use_rag else None)

        batch_prompts.append(origin_prompt)
        batch_meta.append({
            "question": question,
            "gold": gold,
            "reference": references,
            "retrieved_docs": retrieved_docs,
            "retrieval_mode": retrieval_mode,
        })

        if len(batch_prompts) == BATCH_SIZE or idx == len(dataset) - 1:
            predictions = generator.generate_batch(batch_prompts)

            for pred, meta in zip(predictions, batch_meta):
                em = exact_match_score(pred, meta["gold"])
                f1 = f1_score(pred, meta["gold"])
                rouge_l = rouge_l_score(pred, meta["gold"])
                citation, hallucination = citation_accuracy_and_hallucination(pred, meta["reference"])

                rr = None
                if exp_cfg.use_rag:
                    rr = retrieval_recall_at_k(meta["reference"], meta["retrieved_docs"])

                item = {
                    "question": meta["question"],
                    "prediction": pred,
                    "gold": meta["gold"],
                    "em": em,
                    "f1": f1,
                    "rouge_l": rouge_l,
                    "citation": citation,
                    "hallucination": hallucination,
                    "reference": meta["reference"],
                }

                if exp_cfg.use_rag:
                    item["retrieval_mode"] = meta["retrieval_mode"]
                    item["retrieval_recall_at_k"] = rr
                    item["retrieved_docs"] = serialize_docs(meta["retrieved_docs"])

                results.append(item)

            batch_prompts.clear()
            batch_meta.clear()

    cost = time.time() - start_time

    summary = {
        "exact_match": sum(x["em"] for x in results) / len(results) if results else 0.0,
        "f1": sum(x["f1"] for x in results) / len(results) if results else 0.0,
        "rouge_l": sum(x["rouge_l"] for x in results) / len(results) if results else 0.0,
        "citation_accuracy": sum(x["citation"] for x in results) / len(results) if results else 0.0,
        "hallucination_rate": sum(x["hallucination"] for x in results) / len(results) if results else 0.0,
        "retrieval_recall_at_k": (
            sum(x["retrieval_recall_at_k"] for x in results if x.get("retrieval_recall_at_k") is not None)
            / max(1, sum(1 for x in results if x.get("retrieval_recall_at_k") is not None))
            if exp_cfg.use_rag else None
        ),
        "sample_count": len(results),
        "elapsed_seconds": cost,
    }

    save_json(results, results_path)
    save_json(summary, summary_path)

    print(f"结果已保存到: {results_path}")
    print(f"汇总已保存到: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    del generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


# =========================================================
# 主函数
# =========================================================

def main():
    set_seed(SEED)

    dataset = load_dataset(TEST_FILE)
    print(f"测试集样本数: {len(dataset)}")

    need_rag = any(exp.use_rag for exp in EXPERIMENTS)

    vectorstore = None
    exact_index = None
    law_name_candidates = None

    if need_rag:
        print("正在加载向量库...")
        vectorstore = load_vectorstore(VECTORSTORE_PATH)

        print("正在加载精确索引...")
        exact_index = load_exact_index(EXACT_INDEX_PATH)

        print("正在构建法律名候选集合...")
        law_name_candidates = build_law_name_candidates(exact_index)
        print(f"法律名候选数: {len(law_name_candidates)}")

    overall_summary = {}

    for exp_cfg in EXPERIMENTS:
        summary = evaluate_one_experiment(
            exp_cfg=exp_cfg,
            dataset=dataset,
            vectorstore=vectorstore,
            exact_index=exact_index,
            law_name_candidates=law_name_candidates,
        )
        if summary is not None:
            overall_summary[exp_cfg.system_name] = summary

    overall_summary_path = os.path.join(OUTPUT_ROOT, "summary_metrics.json")
    save_json(overall_summary, overall_summary_path)

    print("\n全部实验完成。")
    print(f"总汇总已保存到: {overall_summary_path}")
    print(json.dumps(overall_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()