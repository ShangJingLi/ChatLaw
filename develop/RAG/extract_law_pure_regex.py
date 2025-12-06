"""
法律条文提取脚本（纯正则版）

功能：使用正则精准切分法律条文，不依赖 LLM，速度极快

用法：python extract_law_pure_regex.py -p <PDF文件路径>
"""

import os
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict

import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path: str) -> str:
    """从 PDF 提取全文"""
    doc = fitz.open(pdf_path)
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n".join(text_parts)


def clean_content(text: str) -> str:
    """清洗条文内容"""
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()


def split_by_articles(text: str, law_name: str) -> List[Dict]:
    """使用正则切分法律条文"""
    # 严格匹配 "第X条"，排除"章"、"节"、"编"
    pattern = re.compile(r'(?:\n|^)\s*(第\s*[一二三四五六七八九十百千零0-9\s]+条)(?![一二三四五六七八九十])')

    matches = list(pattern.finditer(text))
    articles = []

    for i, match in enumerate(matches):
        start = match.start()
        article_id = re.sub(r'\s+', '', match.group(1))

        if i < len(matches) - 1:
            end = matches[i + 1].start()
        else:
            end = len(text)

        raw_content = text[start:end].strip()
        # 去掉开头的条文编号
        content = re.sub(r'^第[一二三四五六七八九十百千零0-9\s]+条\s*', '', raw_content)
        content = clean_content(content)

        articles.append({
            "article_number": article_id,
            "content": content,
            "title": f"{law_name}{article_id}" if law_name else ""
        })

    return articles


def main():
    parser = argparse.ArgumentParser(description="法律条文提取（纯正则版）")
    parser.add_argument("--pdf", "-p", required=True, help="PDF 文件路径")
    parser.add_argument("--output", "-o", help="输出 JSON 路径")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        print(f"错误: 文件不存在 {args.pdf}")
        return

    # 从文件名提取法律名称
    pdf_name = Path(args.pdf).stem
    law_name = re.sub(r'_\d+$', '', pdf_name)

    print(f"正在读取 PDF: {args.pdf}")
    text = extract_text_from_pdf(args.pdf)

    print("正在切分条文...")
    articles = split_by_articles(text, law_name)
    print(f"共提取 {len(articles)} 条")

    # 保存
    if args.output is None:
        output_dir = Path(__file__).parent / "resources" / "extracted_pure"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(output_dir / f"{pdf_name}_struct.json")

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({
            "source": args.pdf,
            "total": len(articles),
            "articles": articles
        }, f, ensure_ascii=False, indent=2)

    print(f"已保存至: {args.output}")


if __name__ == "__main__":
    main()
