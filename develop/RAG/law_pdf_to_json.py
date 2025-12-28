import os
import re
import json
from pathlib import Path
import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModelForCausalLM

# ================= 配置 =================
PDF_DIR = "/home/stu1/li/ChatLaw/develop/RAG/output/Legal Documents"
OUTPUT_DIR = "./output_json"

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]

MODEL_PATH = PROJECT_ROOT / "chatlaw" / "resources" / "llm"
TOKENIZER_PATH = PROJECT_ROOT / "chatlaw" / "resources" / "tokenizer"

MAX_INPUT_CHARS = 2000

# ================= 工具函数 =================
def extract_pdf_text(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    pages = []

    for page in doc:
        text = page.get_text()
        # 行内断行 → 空格
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        # 多空行压缩
        text = re.sub(r"\n{2,}", "\n", text)
        pages.append(text)

    doc.close()
    return "\n".join(pages)


def normalize_article_starts(text: str) -> str:
    """
    让真正的“第X条 ……”尽量出现在新行开头，恢复结构锚点。
    注意：不会清洗内容，只是插入换行用于分段。
    """
    # 在“第X条”前插入换行，但避免把“……第十二条规定/至……”这种句中引用当成新条
    text = re.sub(
        r"(?<!\n)(?<![一-龥])"                 # 前面不是换行、也不是汉字（尽量避免句中引用）
        r"(第\s*[一二三四五六七八九十百千0-9]+\s*条)"
        r"(?!\s*(至|的|规定))",                # 后面不是“至/的/规定”
        r"\n\1",
        text
    )
    return text


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        local_files_only=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        device_map="auto",
        dtype="auto",
        attn_implementation="sdpa"
    )

    return model, tokenizer


# ================= LLM：清洗 =================

def llm_clean_article(model, tokenizer, raw_text: str) -> str:
    messages = [
        {
            "role": "user",
            "content": f"""下面是一条法律条文的原始文本，可能包含断行、多余空格、页码等噪声。
                请在【不改变法律含义、不新增内容】的前提下：
                1. 合并断行
                2. 删除页码（如“－2－”）
                3. 删除明显多余的空格
                4. 输出一条干净、连续、适合存入 JSON 的条文文本
                
                只输出清洗后的条文正文，不要解释。
                
                原文如下：
                {raw_text}
                """
        }
    ]

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer([chat_text], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()


# ================= 正则：核心规则 =================
# ① 严格：只用于“条文起始判定”
ARTICLE_START_PATTERN = re.compile(
    r"^\s*(第\s*[一二三四五六七八九十百千0-9]+\s*条)",
    re.MULTILINE
)


# ② 宽松：只用于“条号抽取”
ARTICLE_NUMBER_PATTERN = re.compile(
    r"(第\s*[一二三四五六七八九十百千0-9]+\s*条)"
)


def cut_first_article_from_text(text: str) -> tuple[str, str] | None:
    matches = list(ARTICLE_START_PATTERN.finditer(text))
    if not matches:
        return None

    start = matches[0].start()

    if len(matches) > 1:
        end = matches[1].start()
    else:
        end = len(text)

    return text[start:end], text[end:]


def is_valid_cut(raw_text: str) -> bool:
    s = raw_text.strip()

    # 1) 太短的一律当噪声（比如“第三十八条至”）
    if len(s) < 20:
        return False

    # 2) 如果整段几乎就是“第X条至第Y条...”这种范围引用（且很短），才过滤
    #    注意：这里只在“开头紧跟至”才认为是“引用型切片”
    if re.match(r"^第\s*[一二三四五六七八九十百千0-9]+\s*条\s*至\s*第", s):
        return False

    # 3) 如果开头就是“第X条规定...”且文本很短，才认为是引用碎片
    #    （正文条文一般不会以“第X条规定”开头）
    if re.match(r"^第\s*[一二三四五六七八九十百千0-9]+\s*条\s*规定", s) and len(s) < 60:
        return False

    return True

# ================= 主流程 =================
def process_single_pdf(pdf_path: Path, model, tokenizer):
    law_name = pdf_path.stem
    print(f"\n📘 正在处理：{law_name}")

    remaining_text = normalize_article_starts(extract_pdf_text(pdf_path))
    articles = []

    debug_dir = Path(__file__).parent / "debug_trim" / law_name
    debug_dir.mkdir(parents=True, exist_ok=True)
    step = 0

    while remaining_text.strip():
        prev_len = len(remaining_text)

        cut = cut_first_article_from_text(remaining_text)
        if not cut:
            break

        raw_text, rest = cut

        # ✅ 调试打印：看看“被当成条文起始”的到底切出了什么
        print("DEBUG CUT:", raw_text[:80].replace("\n", "\\n"))

        # 如果切出来明显不是条文（例如“第三十八条至”），跳过它，但必须让文本前进
        if not is_valid_cut(raw_text):
            # 让 remaining_text 前进到 rest，避免死循环
            remaining_text = rest
            continue

        remaining_text = rest


        # 抽取条号（一定成功）
        m = ARTICLE_NUMBER_PATTERN.search(raw_text)
        article_number = m.group(1) if m else "未知条"

        cleaned = llm_clean_article(model, tokenizer, raw_text)

        articles.append({
            "law_name": law_name,
            "article_number": article_number,
            "content": cleaned
        })

        step += 1
        debug_path = debug_dir / f"step_{step:03d}.txt"
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(remaining_text)

        print(f"📝 已保存裁剪后文本：{debug_path}")

        if len(remaining_text) >= prev_len:
            print("❌ remaining_text 未缩短，终止防死循环")
            break

    # ================= 输出 =================

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = Path(OUTPUT_DIR) / f"{law_name}.json"

    with open(out_path, "w", encoding="utf-8") as f:
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

    print(f"✅ 完成 {law_name}，共提取 {len(articles)} 条")


def main():
    model, tokenizer = load_model()
    for pdf_file in Path(PDF_DIR).glob("*.pdf"):
        process_single_pdf(pdf_file, model, tokenizer)


if __name__ == "__main__":
    main()
