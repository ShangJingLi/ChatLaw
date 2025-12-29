# ChatLaw 开发模块

本目录用于 ChatLaw 项目的功能开发。

## 目录结构

```
develop/
├── README.md                # 本文件
├── resources/               # 资源文件目录
│   ├── Legal Documents/     # 法律 PDF 文件存放目录
│   ├── slidding window/     # 滑动窗口提取结果输出
│   └── output/              # 批量处理输出目录
└── RAG/                     # RAG（检索增强生成）模块
    ├── requirements.txt             # Python 依赖
    ├── extract_regex_llm_repair.py  # 法律条文提取（正则+LLM+窗口修复，推荐）
    ├── extract_law_pure_regex.py    # 法律条文提取（纯正则，极速）
    ├── extract_sliding_window.py    # 法律条文提取（滑动窗口，旧版）
    ├── batch_extract_all_laws.py    # 批量提取脚本
    └── verify_extraction.py         # 提取结果验证脚本
```

---

## RAG 模块

### 功能说明

从 PDF 法律文档中提取结构化的法律条文，输出 JSON 格式。

### 提取方案对比

| 方案 | 脚本 | 速度 | 准确率 | 依赖 |
|------|------|------|--------|------|
| **正则+LLM+窗口修复（推荐）** | `extract_regex_llm_repair.py` | 中等 | 高 | GPU + 模型 |
| 纯正则 | `extract_law_pure_regex.py` | **极快** | 低 | 无 |
| 滑动窗口+LLM | `extract_sliding_window.py` | 慢 | 中 | GPU + 模型 |

### extract_regex_llm_repair.py 处理流程

```
PDF全文
   ↓
第一轮：正则切分（按"第X条"分割）
   ↓
LLM 批量结构化（提取 article_number, content, title）
   ↓
检测不完整条目（article_number 为空 或 content 过短）
   ↓
第二轮：大窗口修复（扩展 2000 字符上下文，重新提取）
   ↓
输出 JSON
```

### 快速开始

#### 方式一：正则+LLM（推荐）

1. **安装依赖**
   ```bash
   cd RAG
   pip install -r developer_requirements.txt
   ```

2. **下载模型**（约 15GB）
   ```bash
   # 从 ModelScope 下载
   modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir ../resources/Qwen2.5-7B-Instruct
   ```

3. **运行提取**
   ```bash
   python extract_regex_llm_repair.py -p "你的PDF文件路径"
   ```

4. **验证结果**（基于正则匹配验证）
   ```bash
   python verify_extraction.py -p "原PDF路径" -j "输出JSON路径" -v
   ```

### 方式二：纯正则（极速，无需模型）

```bash
python extract_law_pure_regex.py -p "D:\Legal Documents\中华人民共和国民法典_20200528.pdf"
```

### 批量处理所有法律文件

```bash
python batch_extract_all_laws.py
```

> 注意：需先修改 `batch_extract_all_laws.py` 中的路径配置

## 命令行参数

### extract_regex_llm_repair.py

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--pdf` | `-p` | 输入 PDF 文件路径 | 必填 |
| `--output` | `-o` | 输出 JSON 文件路径 | `./resources/extracted_v2/<pdf名>_struct.json` |
| `--model` | `-m` | 模型路径 | `./resources/Qwen2.5-7B-Instruct`（相对路径） |
| `--batch` | `-b` | 批处理大小 | 8 |

### verify_extraction.py

| 参数 | 简写 | 说明 |
|------|------|------|
| `--pdf` | `-p` | 原 PDF 文件路径 |
| `--json` | `-j` | 提取结果 JSON 路径 |
| `--verbose` | `-v` | 显示详细信息 |

## 输出格式示例

```json
{
  "source": "中华人民共和国民法典_20200528.pdf",
  "total": 1260,
  "articles": [
    {
      "article_number": "第一条",
      "content": "为了保护民事主体的合法权益，调整民事关系，维护社会和经济秩序，适应中国特色社会主义发展要求，弘扬社会主义核心价值观，根据宪法，制定本法。",
      "title": ""
    }
  ]
}
```

## 核心依赖

- Python 3.10+
- PyMuPDF >= 1.23.0（PDF 解析）
- transformers >= 4.37.0（模型推理）
- torch >= 2.0.0（深度学习框架）
- bitsandbytes >= 0.41.0（4bit 量化，可选）
- tqdm >= 4.66.0（进度条）

## 开发说明

1. 所有代码和文档使用中文注释
2. 下载的模型和生成的文件请放入 `resources/` 目录
3. `resources/` 目录已加入 `.gitignore`，不会推送到远程
4. 测试通过后再提交 PR
