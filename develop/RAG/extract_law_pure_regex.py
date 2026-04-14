import os
import re
import json
from pathlib import Path

import fitz  # PyMuPDF
from tqdm import tqdm

# ================= 配置 =================
PDF_DIR = os.path.join("develop", "resources", "output_law_pdf")
OUTPUT_DIR = "./output_json_full"

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]

# 这里只是保留批次结构，规则清洗时不涉及模型 batch 推理
BATCH_SIZE = 16

# 是否跳过已经生成 final json 的文件
SKIP_DONE_FILES = True


# ================= 工具函数 =================
def extract_pdf_text_keep_layout(pdf_path: Path) -> str:
    """
    尽量保留 PDF 的行结构，不在粗分割阶段随意合并空格。
    只做最小必要清理：
    1. 统一换行符
    2. 去掉每行结尾多余空白
    3. 保留每行行首空格，因为条首判定要依赖它
    """
    doc = fitz.open(pdf_path)
    pages = []

    for page in doc:
        text = page.get_text("text")
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        lines = []
        for line in text.split("\n"):
            lines.append(line.rstrip())

        pages.append("\n".join(lines))

    doc.close()
    return "\n".join(pages)


# ================= 正则：核心规则 =================
# 严格法条起始：只认“行首空格 + 第xxx条 + 空格 + 正文”
STRICT_ARTICLE_START_PATTERN = re.compile(
    r"^([ ]*)(第\s*[一二三四五六七八九十百千零〇0-9]+\s*条)([ ]+)(?=\S)",
    re.MULTILINE
)

ARTICLE_NUMBER_PATTERN = re.compile(
    r"(第\s*[一二三四五六七八九十百千零〇0-9]+\s*条)"
)

# 尾部混入结构标题时的裁剪规则
STRUCTURE_TITLE_PATTERN = re.compile(
    r"\n\s*(第\s*[一二三四五六七八九十百千零〇0-9]+\s*[章节编]|附则|附录)\s*.*"
)

# 整行结构标题规则：第xx编/章/节 + 标题
STRUCTURE_HEADING_LINE_PATTERN = re.compile(
    r"^\s*"
    r"(第\s*[一二三四五六七八九十百千零〇0-9]+\s*[编章节])"
    r"(?:\s+[\u4e00-\u9fffA-Za-z0-9《》、，,（）()·\-—\s]+)?"
    r"\s*$"
)

APPENDIX_HEADING_LINE_PATTERN = re.compile(
    r"^\s*(附则|附录)\s*$"
)


def is_structure_heading_line(line: str) -> bool:
    """
    判断一整行是否属于结构标题：
    - 第xx编 ...
    - 第xx章 ...
    - 第xx节 ...
    - 附则
    - 附录
    不包括“第xx条 ...”
    """
    s = line.strip()
    if not s:
        return False

    # 排除法条正文行
    if re.match(r"^第\s*[一二三四五六七八九十百千零〇0-9]+\s*条", s):
        return False

    if STRUCTURE_HEADING_LINE_PATTERN.match(s):
        return True

    if APPENDIX_HEADING_LINE_PATTERN.match(s):
        return True

    return False


def remove_structure_heading_lines(text: str) -> str:
    """
    删除全文或候选文本中单独成行的结构标题：
    - 第xx编 ...
    - 第xx章 ...
    - 第xx节 ...
    - 附则
    - 附录
    """
    kept_lines = []
    for line in text.split("\n"):
        if is_structure_heading_line(line):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


def preprocess_for_cut(text: str) -> str:
    """
    粗分割前预处理：
    1. 删除整行结构标题（第xx编/章/节、附则、附录）
    2. 压缩连续空行为最多两个
    3. 不做“第X条”前插换行
    4. 不删除行首空格
    """
    text = remove_structure_heading_lines(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def find_article_starts(text: str):
    return list(STRICT_ARTICLE_START_PATTERN.finditer(text))


def cut_articles_by_strict_rule(text: str):
    """
    按严格条首规则切分全文：
    - 只认“行首空格 + 第xxx条 + 空格 + 正文”
    """
    matches = find_article_starts(text)
    articles = []

    if not matches:
        return articles

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        raw_text = text[start:end].strip("\n")
        articles.append(raw_text)

    return articles


def trim_trailing_structure_title(raw_text: str) -> str:
    """
    对单条候选法条做规则层去尾噪：
    如果尾部混入了“第X章 / 第X节 / 第X编 / 附则 / 附录”这类结构标题，
    则从该标题开始截断。
    """
    m = STRUCTURE_TITLE_PATTERN.search(raw_text)
    if not m:
        return raw_text

    if m.start() >= int(len(raw_text) * 0.4):
        return raw_text[:m.start()].rstrip()

    return raw_text


def extract_article_number(raw_text: str) -> str:
    m = ARTICLE_NUMBER_PATTERN.search(raw_text)
    return m.group(1) if m else "未知条"


# ================= 规则清洗 =================
def clean_article_by_rules(raw_text: str, article_number: str) -> str:
    """
    纯规则清洗法条正文：
    1. 删除页码
    2. 删除开头条号
    3. 删除残留的整行结构标题（第xx编/章/节、附则、附录）
    4. 合并 PDF 断行
    5. 压缩多余空白
    """
    text = raw_text.strip()

    # 删除页码，如 －5－
    text = re.sub(r"\n?\s*－\s*\d+\s*－\s*\n?", "\n", text)

    # 删除整行结构标题
    text = remove_structure_heading_lines(text)

    # 删除开头条号
    pattern = r"^\s*" + re.escape(article_number) + r"\s*"
    text = re.sub(pattern, "", text)

    # 合并单个断行
    text = re.sub(r"(?<!\n)\n(?!\n)", "", text)

    # 压缩空白
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)

    return text.strip()


def post_clean_basic(text: str, article_number: str | None = None) -> str:
    """
    通用后处理兜底
    """
    if not text:
        return ""

    text = text.strip()

    # 删除 markdown 包裹
    text = re.sub(r"^```(?:json|text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    # 删除常见前缀
    text = re.sub(
        r"^(Assistant|assistant|法条正文|清洗结果|输出结果|结果)\s*[:：]?\s*",
        "",
        text
    )

    # 若输出成 JSON 字符串，尽量提取正文
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                if "法条正文" in obj and isinstance(obj["法条正文"], str):
                    text = obj["法条正文"].strip()
                elif "content" in obj and isinstance(obj["content"], str):
                    text = obj["content"].strip()
                elif "正文" in obj and isinstance(obj["正文"], str):
                    text = obj["正文"].strip()
        except Exception:
            pass

    # 删页码
    text = re.sub(r"－\s*\d+\s*－", "", text)

    # 再删一次整行结构标题
    text = remove_structure_heading_lines(text)

    # 删除重复条号
    if article_number:
        pattern = r"^\s*" + re.escape(article_number) + r"\s*"
        text = re.sub(pattern, "", text)

    # 删除开头错误编号
    text = re.sub(r"^\s*\d+\s+", "", text)

    # 删除残留字段名前缀
    text = re.sub(r"^(条号|正文|法条正文)\s*[:：]?\s*", "", text)

    # 压缩空白
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = text.strip()

    # 空结果归一化
    if re.fullmatch(r"[（(]?\s*(无|空|无正文|暂无)\s*[）)]?", text):
        return ""

    return text


def batched(iterable, batch_size: int):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


# ================= 单文件处理 =================
def process_single_pdf(pdf_path: Path, final_json_root: Path, debug_root: Path):
    law_name = pdf_path.stem
    tqdm.write(f"\n📘 正在处理：{law_name}")

    raw_full_text = extract_pdf_text_keep_layout(pdf_path)
    full_text = preprocess_for_cut(raw_full_text)

    article_blocks = cut_articles_by_strict_rule(full_text)
    tqdm.write(f"🔍 严格规则识别到候选条文数：{len(article_blocks)}")

    # ===== 每个文件单独一个 debug 目录 =====
    file_debug_dir = debug_root / law_name
    raw_dir = file_debug_dir / "raw_candidates"
    cleaned_dir = file_debug_dir / "cleaned_outputs"

    raw_dir.mkdir(parents=True, exist_ok=True)
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    pending_items = []
    for idx, raw_text in enumerate(article_blocks, start=1):
        raw_text = trim_trailing_structure_title(raw_text)
        raw_text = remove_structure_heading_lines(raw_text)

        article_number = extract_article_number(raw_text)

        raw_path = raw_dir / f"step_{idx:03d}_{article_number}.txt"
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(raw_text)

        pending_items.append({
            "idx": idx,
            "law_name": law_name,
            "article_number": article_number,
            "raw_text": raw_text,
            "raw_path": str(raw_path)
        })

    articles = []
    debug_records = []

    inner_batches = list(batched(pending_items, BATCH_SIZE))

    for batch_idx, batch_items in enumerate(inner_batches, start=1):
        tqdm.write(
            f"🚀 {law_name}：批次 {batch_idx}/{len(inner_batches)}，"
            f"本批 {len(batch_items)} 条"
        )

        raw_outputs = [
            clean_article_by_rules(
                item["raw_text"],
                article_number=item["article_number"]
            )
            for item in batch_items
        ]

        for item, raw_cleaned in zip(batch_items, raw_outputs):
            cleaned = post_clean_basic(
                raw_cleaned,
                article_number=item["article_number"]
            )

            cleaned_path = cleaned_dir / f"step_{item['idx']:03d}_{item['article_number']}.json"
            with open(cleaned_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "law_name": item["law_name"],
                        "article_number": item["article_number"],
                        "raw_input_path": item["raw_path"],
                        "raw_input": item["raw_text"],
                        "clean_mode": "rules",
                        "rule_output": raw_cleaned,
                        "post_clean_output": cleaned
                    },
                    f,
                    ensure_ascii=False,
                    indent=2
                )

            debug_records.append({
                "law_name": item["law_name"],
                "article_number": item["article_number"],
                "raw_input_path": item["raw_path"],
                "cleaned_output_path": str(cleaned_path),
                "raw_input": item["raw_text"],
                "clean_mode": "rules",
                "rule_output": raw_cleaned,
                "post_clean_output": cleaned
            })

            if not cleaned:
                tqdm.write(f"⚠️ 空结果：{item['article_number']}")
                continue

            articles.append({
                "law_name": item["law_name"],
                "article_number": item["article_number"],
                "content": cleaned
            })

    # ===== 最终 json 全部统一放 final_json 目录 =====
    final_json_root.mkdir(parents=True, exist_ok=True)
    final_json_path = final_json_root / f"{law_name}.json"
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "law_name": law_name,
                "total_articles": len(articles),
                "articles": articles
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    # ===== 每个文件自己的 debug 汇总 =====
    debug_summary_path = file_debug_dir / f"{law_name}_debug_summary.json"
    with open(debug_summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "law_name": law_name,
                "total_candidates": len(pending_items),
                "total_cleaned_articles": len(articles),
                "clean_mode": "rules",
                "records": debug_records
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    tqdm.write(f"✅ 完成：{law_name}")
    tqdm.write(f"📁 本文件 debug 目录：{file_debug_dir}")
    tqdm.write(f"📁 最终 JSON：{final_json_path}")


# ================= 全量主流程 =================
def main():
    pdf_files = sorted(Path(PDF_DIR).glob("*.pdf"))
    print(f"📊 共检测到 {len(pdf_files)} 个 PDF 文件")

    if not pdf_files:
        print("❌ 未找到 PDF 文件，程序结束。")
        return

    final_json_root = Path(OUTPUT_DIR) / "final_json"
    debug_root = Path(OUTPUT_DIR) / "debug_by_file"

    final_json_root.mkdir(parents=True, exist_ok=True)
    debug_root.mkdir(parents=True, exist_ok=True)

    for pdf_file in tqdm(pdf_files, desc="📂 全量切分进度"):
        law_name = pdf_file.stem
        final_json_path = final_json_root / f"{law_name}.json"

        if SKIP_DONE_FILES and final_json_path.exists():
            tqdm.write(f"⏭️ 跳过已完成：{law_name}")
            continue

        process_single_pdf(
            pdf_path=pdf_file,
            final_json_root=final_json_root,
            debug_root=debug_root
        )

    print("✅ 全部文件处理完成。")


if __name__ == "__main__":
    main()