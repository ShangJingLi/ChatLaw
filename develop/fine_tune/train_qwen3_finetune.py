#!/usr/bin/env python3
"""使用 PyTorch + Transformers 微调 Qwen/Qwen3-4B-Instruct-2507。

按用户要求：
1) 直接使用 `python train_qwen3_finetune.py` 启动（不使用命令行参数）。
2) 数据集从本脚本同目录自动查找。
3) 训练输出保存在本脚本同目录下新建的子目录中。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

# =========================
# 训练配置（按需直接修改此处）
# =========================
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
SEED = 42

MAX_LENGTH = 2048
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
LOGGING_STEPS = 10
SAVE_STEPS = 200
USE_BF16 = True
GRADIENT_CHECKPOINTING = True

# 数据文件优先按下列名称在脚本目录查找；若都不存在，会自动挑选目录内第一个 .json/.jsonl
PREFERRED_DATA_FILES = ["train.json", "train.jsonl", "dataset.json", "dataset.jsonl"]

# 输出目录（会在脚本目录下创建）
OUTPUT_SUBDIR = "qwen3_4b_finetuned"


def find_train_file(script_dir: Path) -> Path:
    for name in PREFERRED_DATA_FILES:
        candidate = script_dir / name
        if candidate.exists() and candidate.is_file():
            return candidate

    json_candidates = sorted(script_dir.glob("*.json")) + sorted(script_dir.glob("*.jsonl"))
    if json_candidates:
        return json_candidates[0]

    raise FileNotFoundError(
        f"未在 {script_dir} 找到训练数据文件。"
        f"请放置 {PREFERRED_DATA_FILES} 之一，或任意 .json/.jsonl 文件。"
    )


def load_records(train_file: Path) -> list[dict[str, Any]]:
    if train_file.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        with train_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    with train_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON 文件应为数组格式：[{...}, {...}]")
    return data


def build_user_content(example: dict[str, Any]) -> str:
    references = example.get("reference", [])
    if not isinstance(references, list):
        references = [str(references)]
    references_text = "\n".join(str(x).strip() for x in references if str(x).strip())

    question = str(example.get("question", "")).strip()
    if not question:
        raise ValueError("样本缺少 question 字段或内容为空")

    return (
        "你是一名专业法律助手。请严格依据参考法条回答问题。\n\n"
        f"【参考法条】\n{references_text}\n\n"
        f"【用户问题】\n{question}\n"
    )


def preprocess_function(example: dict[str, Any], tokenizer: AutoTokenizer) -> dict[str, Any]:
    answer = str(example.get("answer", "")).strip()
    if not answer:
        raise ValueError("样本缺少 answer 字段或内容为空")

    user_content = build_user_content(example)
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": answer},
    ]

    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)

    full = tokenizer(full_text, truncation=True, max_length=MAX_LENGTH, add_special_tokens=False)
    prompt = tokenizer(prompt_text, truncation=True, max_length=MAX_LENGTH, add_special_tokens=False)

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]

    prompt_len = min(len(prompt["input_ids"]), len(input_ids))
    labels = [-100] * prompt_len + input_ids[prompt_len:]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class SupervisedDataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids = []
        attention_mask = []
        labels = []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    train_file = find_train_file(script_dir)
    output_dir = script_dir / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if USE_BF16 else None,
    )

    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()

    records = load_records(train_file)
    dataset = Dataset.from_list(records)
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer=tokenizer),
        remove_columns=dataset.column_names,
        desc="Tokenizing train dataset",
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        bf16=USE_BF16,
        fp16=(not USE_BF16 and torch.cuda.is_available()),
        report_to="none",
        dataloader_num_workers=2,
    )

    collator = SupervisedDataCollator(pad_token_id=tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    print(f"[INFO] 使用数据集: {train_file}")
    print(f"[INFO] 输出目录: {output_dir}")

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("[INFO] 训练完成")


if __name__ == "__main__":
    main()
