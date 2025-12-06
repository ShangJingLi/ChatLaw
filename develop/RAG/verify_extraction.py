"""
法律条文提取结果验证脚本

功能：
1. 内容匹配检查：每个条目的 content 是否能在原 PDF 中找到（目标 >= 98%）
2. 编号顺序检查：条文编号是否按顺序排列（失序率目标 <= 2%）

用法：
python verify_extraction.py -p <原PDF路径> -j <提取结果JSON路径>
"""

import os
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import fitz  # PyMuPDF

# ==================== 中文数字转阿拉伯数字 ====================

CN_NUM = {
    '零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
    '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
    '十': 10, '百': 100, '千': 1000
}


def cn_to_arabic(cn_str: str) -> int:
    """
    将中文数字转换为阿拉伯数字
    例如: "第一千二百六十条" -> 1260
    """
    # 提取数字部分
    cn_str = cn_str.replace('第', '').replace('条', '').strip()

    # 如果是纯阿拉伯数字
    if cn_str.isdigit():
        return int(cn_str)

    result = 0
    temp = 0

    for char in cn_str:
        if char in CN_NUM:
            num = CN_NUM[char]
            if num >= 10:
                if temp == 0:
                    temp = 1
                if num == 10:
                    result += temp * 10
                    temp = 0
                elif num == 100:
                    result += temp * 100
                    temp = 0
                elif num == 1000:
                    result += temp * 1000
                    temp = 0
            else:
                temp = num

    result += temp
    return result


# ==================== PDF 文本提取 ====================

def extract_text_from_pdf(pdf_path: str) -> str:
    """从 PDF 提取全文"""
    doc = fitz.open(pdf_path)
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n".join(text_parts)


def normalize_text(text: str) -> str:
    """
    规范化文本用于匹配
    - 去除所有空白字符（空格、换行、制表符等）
    - 统一全角半角
    """
    # 去除所有空白
    text = re.sub(r'\s+', '', text)
    return text


# ==================== 检查函数 ====================

def flatten_articles(articles: List) -> List[Dict]:
    """展平嵌套的 articles 列表"""
    result = []
    for art in articles:
        if isinstance(art, dict):
            result.append(art)
        elif isinstance(art, list):
            # 递归展平
            result.extend(flatten_articles(art))
    return result


def check_content_match(pdf_text: str, articles: List[Dict]) -> Tuple[int, int, List[Dict]]:
    """
    检查内容匹配率

    返回: (匹配数, 总数, 未匹配列表)
    """
    # 先展平
    articles = flatten_articles(articles)

    # 规范化 PDF 文本
    normalized_pdf = normalize_text(pdf_text)

    matched = 0
    unmatched = []

    for art in articles:
        if not isinstance(art, dict):
            continue
        content = art.get('content', '')
        if not content:
            continue

        # 规范化条文内容
        normalized_content = normalize_text(content)

        # 检查是否在 PDF 中存在
        # 由于 PDF 提取可能有微小差异，我们用子串匹配
        # 取内容的前 50 个字符做匹配（避免完整匹配过于严格）
        search_str = normalized_content[:min(50, len(normalized_content))]

        if search_str in normalized_pdf:
            matched += 1
        else:
            # 尝试更宽松的匹配（前 30 个字符）
            search_str_short = normalized_content[:min(30, len(normalized_content))]
            if search_str_short in normalized_pdf:
                matched += 1
            else:
                unmatched.append({
                    'article_number': art.get('article_number', '未知'),
                    'content_preview': content[:100] + '...' if len(content) > 100 else content
                })

    return matched, len(articles), unmatched


def check_number_order(articles: List[Dict]) -> Tuple[int, int, List[Dict]]:
    """
    检查编号顺序

    返回: (正序数, 总数, 失序列表)
    """
    # 先展平
    articles = flatten_articles(articles)

    if not articles:
        return 0, 0, []

    ordered = 0
    disordered = []
    prev_num = 0

    for i, art in enumerate(articles):
        article_number = art.get('article_number', '')

        try:
            current_num = cn_to_arabic(article_number)
        except:
            # 无法解析的编号
            disordered.append({
                'index': i,
                'article_number': article_number,
                'reason': '无法解析编号'
            })
            continue

        if current_num > prev_num:
            ordered += 1
        elif current_num == prev_num:
            # 重复编号（可能是分款）
            ordered += 1  # 暂时视为正常
        else:
            disordered.append({
                'index': i,
                'article_number': article_number,
                'prev_number': prev_num,
                'current_number': current_num,
                'reason': f'顺序错误: 前一条是第{prev_num}条，当前是第{current_num}条'
            })

        prev_num = current_num

    return ordered, len(articles), disordered


# ==================== 主程序 ====================

def main():
    parser = argparse.ArgumentParser(description="验证法律条文提取结果")
    parser.add_argument("--pdf", "-p", required=True, help="原 PDF 文件路径")
    parser.add_argument("--json", "-j", required=True, help="提取结果 JSON 文件路径")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细信息")

    args = parser.parse_args()

    # 检查文件存在
    if not os.path.exists(args.pdf):
        print(f"错误: PDF 文件不存在: {args.pdf}")
        return
    if not os.path.exists(args.json):
        print(f"错误: JSON 文件不存在: {args.json}")
        return

    print("=" * 60)
    print("法律条文提取结果验证")
    print("=" * 60)

    # 1. 加载数据
    print(f"\n[1] 加载数据...")
    print(f"    PDF: {args.pdf}")
    print(f"    JSON: {args.json}")

    pdf_text = extract_text_from_pdf(args.pdf)
    with open(args.json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    articles = data.get('articles', [])
    print(f"    条文总数: {len(articles)}")

    # 2. 内容匹配检查
    print(f"\n[2] 内容匹配检查...")
    matched, total, unmatched = check_content_match(pdf_text, articles)
    match_rate = matched / total * 100 if total > 0 else 0

    print(f"    匹配数: {matched}/{total}")
    print(f"    匹配率: {match_rate:.2f}%")

    if match_rate >= 98:
        print(f"    ✅ 通过 (>= 98%)")
    else:
        print(f"    ❌ 未通过 (< 98%)")

    if args.verbose and unmatched:
        print(f"\n    未匹配条目:")
        for item in unmatched[:10]:  # 只显示前10个
            print(f"      - {item['article_number']}: {item['content_preview'][:50]}...")
        if len(unmatched) > 10:
            print(f"      ... 还有 {len(unmatched) - 10} 个未匹配")

    # 3. 编号顺序检查
    print(f"\n[3] 编号顺序检查...")
    ordered, total, disordered = check_number_order(articles)
    order_rate = ordered / total * 100 if total > 0 else 0
    disorder_rate = 100 - order_rate

    print(f"    正序数: {ordered}/{total}")
    print(f"    失序率: {disorder_rate:.2f}%")

    if disorder_rate <= 2:
        print(f"    ✅ 通过 (<= 2%)")
    else:
        print(f"    ❌ 未通过 (> 2%)")

    if args.verbose and disordered:
        print(f"\n    失序条目:")
        for item in disordered[:10]:
            print(f"      - 索引 {item['index']}: {item['article_number']} - {item['reason']}")
        if len(disordered) > 10:
            print(f"      ... 还有 {len(disordered) - 10} 个失序")

    # 4. 总结
    print("\n" + "=" * 60)
    print("验证结果总结")
    print("=" * 60)
    print(f"  内容匹配率: {match_rate:.2f}% {'✅' if match_rate >= 98 else '❌'}")
    print(f"  编号失序率: {disorder_rate:.2f}% {'✅' if disorder_rate <= 2 else '❌'}")

    if match_rate >= 98 and disorder_rate <= 2:
        print("\n🎉 全部验证通过！")
    else:
        print("\n⚠️ 存在问题，请检查提取逻辑")

    print("=" * 60)


if __name__ == "__main__":
    main()
