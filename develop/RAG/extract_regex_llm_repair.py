"""
法律条文提取脚本（正则切分 + LLM 结构化版）

功能：
1. 使用正则快速切分法律条文
2. 使用 Qwen2.5-7B-Instruct-AWQ 进行结构化清洗
3. 支持 Batch 处理，跑满 GPU

优势：
- 速度极快（无需滑动窗口）
- 准确率高（正则定位边界）
- 结构化好（LLM 专注清洗）
"""

import os
import json
import re
import argparse
import time
from pathlib import Path
from typing import List, Dict

import fitz  # PyMuPDF
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==================== 配置 ====================

# 模型配置
# 普通版 + 4bit 量化加载（相对路径，基于脚本所在目录）
MODEL_NAME = str(Path(__file__).parent / "resources" / "Qwen2.5-7B-Instruct")

# 生成配置
BATCH_SIZE = 8          # 4bit动态量化显存占用稍高，先用8，如果显存有空余再加到16
MAX_NEW_TOKENS = 1024   # 增加长度防止截断

# ==================== 文本处理 ====================

def extract_text_from_pdf(pdf_path: str) -> str:
    """从 PDF 提取全文"""
    doc = fitz.open(pdf_path)
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n".join(text_parts)

def split_by_articles(text: str) -> List[Dict]:
    """
    使用正则切分法律条文
    返回: [{"raw_text": "第一条...", "article_id": "第一条", "start": 0, "end": 100}, ...]
    """
    # 匹配 "第X条"（严格模式）
    pattern = re.compile(r'(?:\n|^)\s*(第\s*[一二三四五六七八九十百千零0-9\s]+条)(?![一二三四五六七八九十])')

    matches = list(pattern.finditer(text))
    articles = []

    for i, match in enumerate(matches):
        start = match.start()
        article_id = re.sub(r'\s+', '', match.group(1))

        if i < len(matches) - 1:
            end = matches[i+1].start()
        else:
            end = len(text)

        raw_content = text[start:end].strip()

        articles.append({
            "id": article_id,
            "raw_text": raw_content,
            "start": start,
            "end": end
        })

    return articles

# ==================== LLM 处理 ====================

def load_model(model_path: str):
    print(f"正在加载模型: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # 使用 bitsandbytes 4bit 量化加载
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quantization_config
        )
        print("模型加载完成 (4-bit 量化)")
        return model, tokenizer
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None, None

def process_batch(model, tokenizer, batch_articles: List[Dict]) -> List[Dict]:
    """批量处理条文结构化"""

    prompts = []
    for art in batch_articles:
        raw = art['raw_text']
        # 构造 Prompt
        prompt = f"""你是一个法律助手。请将以下法律条文文本转换为 JSON 格式。
        
要求：
1. 提取 "article_number"（如"第一条"）
2. 提取 "content"（正文内容，去除换行符和多余空格）
3. 提取 "title"（如果条文第一句显然是标题/定义，则提取，否则为空字符串）
4. 直接输出 JSON，不要 Markdown 标记。

待处理文本：
{raw}

JSON 输出："""

        messages = [
            {"role": "system", "content": "你是一个精确的法律条文解析器。"},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(text)

    # 批量编码
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").to(model.device)

    # 批量生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # 解码
    # 只取生成部分
    input_len = inputs.input_ids.shape[1]
    generated_ids = outputs[:, input_len:]
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    results = []
    for i, response in enumerate(responses):
        try:
            # 尝试解析 JSON
            # 简单清理
            json_str = response.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:-3]
            elif json_str.startswith("```"):
                json_str = json_str[3:-3]

            data = json.loads(json_str)
            
            # 如果 LLM 返回的是列表，取第一个元素
            if isinstance(data, list):
                if len(data) > 0:
                    data = data[0]
                else:
                    raise ValueError("Empty list returned")
            
            results.append(data)
        except:
            # 解析失败，回退到原始文本
            results.append({
                "article_number": batch_articles[i]['id'],
                "content": batch_articles[i]['raw_text'],
                "title": "",
                "error": "json_parse_fail"
            })

    return results


def flatten_articles(articles: List) -> List[Dict]:
    """展平可能嵌套的 articles 列表"""
    result = []
    for item in articles:
        if isinstance(item, dict):
            result.append(item)
        elif isinstance(item, list):
            # 递归展平
            result.extend(flatten_articles(item))
    return result


def is_incomplete(article: Dict) -> bool:
    """检测条目是否不完整"""
    # article_number 为空或不符合格式
    num = article.get('article_number', '')
    if not num or not re.match(r'^第[一二三四五六七八九十百千零0-9]+条$', num):
        return True
    # content 过短（可能截断）
    content = article.get('content', '')
    if len(content) < 10:
        return True
    return False


def repair_single(model, tokenizer, full_text: str, raw_article: Dict, window_size: int = 2000) -> Dict:
    """
    用大窗口重新提取单条条文
    """
    # 获取原位置，扩展上下文
    start = raw_article.get('start', 0)
    end = raw_article.get('end', len(full_text))

    # 扩展窗口
    ctx_start = max(0, start - window_size // 2)
    ctx_end = min(len(full_text), end + window_size // 2)
    context = full_text[ctx_start:ctx_end]

    article_id = raw_article['id']

    prompt = f"""你是一个法律助手。请从以下文本中提取条文 "{article_id}" 的完整内容。

要求：
1. 提取 "article_number"（即"{article_id}"）
2. 提取 "content"（该条的完整正文，去除换行符和多余空格）
3. 提取 "title"（如果条文有标题则提取，否则为空字符串）
4. 只输出这一条的 JSON，不要其他内容

文本上下文：
{context}

JSON 输出："""

    messages = [
        {"role": "system", "content": "你是一个精确的法律条文解析器。"},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    input_len = inputs.input_ids.shape[1]
    generated_ids = outputs[:, input_len:]
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    try:
        json_str = response.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:-3]
        elif json_str.startswith("```"):
            json_str = json_str[3:-3]
        data = json.loads(json_str)
        data['repaired'] = True
        return data
    except:
        return {
            "article_number": article_id,
            "content": raw_article['raw_text'],
            "title": "",
            "repair_failed": True
        }

# ==================== 主程序 ====================

def main():
    parser = argparse.ArgumentParser(description="正则+LLM 法律条文提取")
    parser.add_argument("--pdf", "-p", required=True, help="PDF文件路径")
    parser.add_argument("--output", "-o", default=None, help="输出文件路径")
    parser.add_argument("--model", "-m", default=MODEL_NAME, help="模型路径")
    parser.add_argument("--batch", "-b", type=int, default=BATCH_SIZE, help="Batch Size")

    args = parser.parse_args()

    # 1. 提取文本
    print(f"正在读取 PDF: {args.pdf}")
    full_text = extract_text_from_pdf(args.pdf)

    # 2. 正则切分
    print("正在进行正则切分...")
    raw_articles = split_by_articles(full_text)
    print(f"共切分出 {len(raw_articles)} 条条文")

    if not raw_articles:
        print("未找到条文，请检查文件内容或正则匹配规则。")
        return

    # 3. 加载模型
    if not os.path.exists(args.model):
        print(f"错误: 模型路径不存在 {args.model}")
        print("请先下载 Qwen2.5-7B-Instruct-AWQ")
        return

    model, tokenizer = load_model(args.model)
    if not model:
        return

    # 4. 批量处理（第一轮）
    print("正在进行 LLM 结构化处理（第一轮）...")
    final_articles = []

    for i in tqdm(range(0, len(raw_articles), args.batch), desc="第一轮"):
        batch = raw_articles[i : i + args.batch]
        processed = process_batch(model, tokenizer, batch)
        final_articles.extend(processed)

    # 展平可能的嵌套列表
    final_articles = flatten_articles(final_articles)

    # 5. 检测不完整条目并修复（第二轮）
    incomplete_indices = [i for i, art in enumerate(final_articles) if is_incomplete(art)]

    if incomplete_indices:
        print(f"\n检测到 {len(incomplete_indices)} 条不完整，启动大窗口修复...")
        for idx in tqdm(incomplete_indices, desc="修复中"):
            repaired = repair_single(model, tokenizer, full_text, raw_articles[idx])
            final_articles[idx] = repaired
        print(f"修复完成")
    else:
        print("所有条目完整，无需修复")

    # 6. 保存结果
    if args.output is None:
        pdf_name = Path(args.pdf).stem
        # 修改保存路径为 resources/extracted_v2
        output_dir = Path(__file__).parent / "resources" / "extracted_v2"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(output_dir / f"{pdf_name}_struct.json")
        
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({
            "source": args.pdf,
            "total": len(final_articles),
            "articles": final_articles
        }, f, ensure_ascii=False, indent=2)
        
    print(f"处理完成，结果已保存至: {args.output}")

if __name__ == "__main__":
    main()
