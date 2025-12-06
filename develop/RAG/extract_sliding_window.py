"""
法律条文提取脚本

功能：使用滑动窗口 + LLM 从 PDF 文档中提取法律条文，以 JSON 格式保存
模型：Qwen2.5-3B-Instruct（约3B参数，显存友好）
"""

import os
import json
import re
import argparse
from pathlib import Path
from typing import Generator

import fitz  # PyMuPDF
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ==================== 配置 ====================

# 模型配置
# 模型路径（相对路径，基于脚本所在目录）
MODEL_NAME = str(Path(__file__).parent / "resources" / "Qwen2.5-3B-Instruct")

# 滑动窗口配置
WINDOW_SIZE = 2000      # 窗口大小（字符数）
OVERLAP_SIZE = 500      # 重叠大小（字符数）

# 生成配置
MAX_NEW_TOKENS = 4096   # 最大生成 token 数
TEMPERATURE = 0.1       # 低温度，更确定性的输出


# ==================== PDF 解析 ====================

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    从 PDF 文件中提取全部文本
    
    Args:
        pdf_path: PDF 文件路径
        
    Returns:
        提取的文本内容
    """
    doc = fitz.open(pdf_path)
    text_parts = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        text_parts.append(text)
    
    doc.close()
    
    # 合并所有页面文本，清理多余空白
    full_text = "\n".join(text_parts)
    # 清理多余的空行
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)
    
    return full_text


# ==================== 滑动窗口 ====================

def sliding_window(text: str, window_size: int = WINDOW_SIZE, 
                   overlap: int = OVERLAP_SIZE) -> Generator[tuple[int, str], None, None]:
    """
    滑动窗口生成器
    
    Args:
        text: 输入文本
        window_size: 窗口大小
        overlap: 重叠大小
        
    Yields:
        (窗口索引, 窗口文本)
    """
    step = window_size - overlap
    start = 0
    window_idx = 0
    
    while start < len(text):
        end = min(start + window_size, len(text))
        window_text = text[start:end]
        
        yield window_idx, window_text
        
        window_idx += 1
        start += step
        
        # 如果已经到达末尾，退出
        if end == len(text):
            break


# ==================== LLM 提取 ====================

def load_model(model_name: str = MODEL_NAME, device: str = "auto"):
    """
    加载模型和分词器
    
    Args:
        model_name: 模型名称或路径
        device: 设备，"auto" 自动选择
        
    Returns:
        (model, tokenizer)
    """
    print(f"正在加载模型: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=device,
        trust_remote_code=True
    )
    
    print("模型加载完成")
    return model, tokenizer


def extract_articles_from_window(model, tokenizer, window_text: str) -> list[dict]:
    """
    使用 LLM 从窗口文本中提取法律条文
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        window_text: 窗口文本
        
    Returns:
        提取的条文列表，每个条文是一个字典
    """
    # 构造提示词
    system_prompt = """你是一个专业的法律文本分析助手。你的任务是从给定的文本中提取完整的法律条文。

提取规则：
1. 识别以"第X条"开头的法律条文
2. 提取条文的完整内容，包括所有款项
3. 如果条文不完整（被截断），标记为不完整
4. 严格按照原文提取，不要修改或总结

输出格式要求（JSON数组）：
[
  {
    "article_number": "第一条",
    "title": "条文标题（如果有）",
    "content": "条文完整内容",
    "is_complete": true
  }
]

如果文本中没有法律条文，返回空数组 []"""

    user_prompt = f"""请从以下文本中提取所有法律条文：

---
{window_text}
---

请以JSON格式输出提取结果："""

    # 构造消息
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # 使用 chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 编码输入
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # 生成
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # 解码输出（只取新生成的部分）
    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # 解析 JSON
    articles = parse_json_response(response)
    
    return articles


def parse_json_response(response: str) -> list[dict]:
    """
    解析 LLM 输出的 JSON 响应
    
    Args:
        response: LLM 的原始输出
        
    Returns:
        解析后的条文列表
    """
    # 尝试直接解析
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # 尝试提取 JSON 数组
    json_pattern = r'\[[\s\S]*?\]'
    matches = re.findall(json_pattern, response)
    
    for match in matches:
        try:
            result = json.loads(match)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            continue
    
    # 尝试提取 ```json 代码块
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    code_matches = re.findall(code_block_pattern, response)
    
    for match in code_matches:
        try:
            result = json.loads(match)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            continue
    
    print(f"警告: 无法解析 JSON 响应，原文: {response[:200]}...")
    return []


# ==================== 去重与合并 ====================

def merge_articles(all_articles: list[dict]) -> list[dict]:
    """
    合并和去重提取的条文
    
    Args:
        all_articles: 所有提取的条文列表
        
    Returns:
        去重合并后的条文列表
    """
    # 按条文编号分组
    article_map = {}
    
    for article in all_articles:
        article_num = article.get("article_number", "")
        if not article_num:
            continue
            
        # 标准化条文编号（去除空格）
        article_num = article_num.strip()
        
        if article_num not in article_map:
            article_map[article_num] = article
        else:
            # 如果已存在，选择更完整的版本
            existing = article_map[article_num]
            existing_content = existing.get("content", "")
            new_content = article.get("content", "")
            
            # 选择内容更长且完整的版本
            if len(new_content) > len(existing_content):
                article_map[article_num] = article
            elif article.get("is_complete", False) and not existing.get("is_complete", False):
                article_map[article_num] = article
    
    # 按条文编号排序
    def sort_key(item):
        article_num = item[0]
        # 提取数字进行排序
        numbers = re.findall(r'\d+', article_num)
        if numbers:
            return int(numbers[0])
        return 0
    
    sorted_articles = sorted(article_map.items(), key=sort_key)
    
    return [article for _, article in sorted_articles]


# ==================== 主流程 ====================

def extract_law_articles(pdf_path: str, output_path: str, 
                         model_path: str = MODEL_NAME,
                         window_size: int = WINDOW_SIZE,
                         overlap: int = OVERLAP_SIZE):
    """
    主提取函数
    
    Args:
        pdf_path: PDF 文件路径
        output_path: 输出 JSON 文件路径
        model_path: 模型路径或名称
        window_size: 滑动窗口大小
        overlap: 窗口重叠大小
    """
    print("=" * 50)
    print("法律条文提取工具")
    print("=" * 50)
    
    # 1. 提取 PDF 文本
    print(f"\n[1/4] 正在读取 PDF: {pdf_path}")
    full_text = extract_text_from_pdf(pdf_path)
    print(f"提取文本长度: {len(full_text)} 字符")
    
    # 2. 加载模型
    print(f"\n[2/4] 正在加载模型...")
    model, tokenizer = load_model(model_path)
    
    # 3. 滑动窗口提取
    print(f"\n[3/4] 正在使用滑动窗口提取条文...")
    print(f"窗口大小: {window_size}, 重叠: {overlap}")
    
    all_articles = []
    windows = list(sliding_window(full_text, window_size, overlap))
    
    for window_idx, window_text in tqdm(windows, desc="处理窗口"):
        try:
            articles = extract_articles_from_window(model, tokenizer, window_text)
            all_articles.extend(articles)
        except Exception as e:
            print(f"\n窗口 {window_idx} 处理失败: {e}")
            continue
    
    print(f"\n共提取到 {len(all_articles)} 条原始记录")
    
    # 4. 合并去重
    print(f"\n[4/4] 正在合并去重...")
    merged_articles = merge_articles(all_articles)
    print(f"去重后剩余 {len(merged_articles)} 条")
    
    # 5. 保存结果
    output_data = {
        "source_file": os.path.basename(pdf_path),
        "total_articles": len(merged_articles),
        "extraction_config": {
            "model": model_path,
            "window_size": window_size,
            "overlap": overlap
        },
        "articles": merged_articles
    }
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_path}")
    print("=" * 50)
    print("提取完成！")
    print("=" * 50)
    
    return merged_articles


# ==================== 命令行入口 ====================

def main():
    parser = argparse.ArgumentParser(
        description="从 PDF 文档中提取法律条文"
    )
    parser.add_argument(
        "--pdf", "-p",
        type=str,
        required=True,
        help="输入的 PDF 文件路径"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出的 JSON 文件路径（默认保存在 resources 目录）"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=MODEL_NAME,
        help=f"模型名称或路径（默认: {MODEL_NAME}）"
    )
    parser.add_argument(
        "--window-size", "-w",
        type=int,
        default=WINDOW_SIZE,
        help=f"滑动窗口大小（默认: {WINDOW_SIZE}）"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=OVERLAP_SIZE,
        help=f"窗口重叠大小（默认: {OVERLAP_SIZE}）"
    )
    
    args = parser.parse_args()
    
    # 默认输出路径
    if args.output is None:
        script_dir = Path(__file__).parent
        resources_dir = script_dir / "resources"
        resources_dir.mkdir(exist_ok=True)
        
        pdf_name = Path(args.pdf).stem
        args.output = str(resources_dir / f"{pdf_name}_articles.json")
    
    # 执行提取
    extract_law_articles(
        pdf_path=args.pdf,
        output_path=args.output,
        model_path=args.model,
        window_size=args.window_size,
        overlap=args.overlap
    )


if __name__ == "__main__":
    main()
