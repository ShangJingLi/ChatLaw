"""
批量提取所有法律PDF的条文
"""
import os
import subprocess
from pathlib import Path
from tqdm import tqdm

# 配置（请根据实际情况修改）
SCRIPT_DIR = Path(__file__).parent
PDF_DIR = r".\Legal Documents"                    # PDF 文件目录
OUTPUT_DIR = str(SCRIPT_DIR / "output")           # 输出目录（相对路径）
SCRIPT_PATH = str(SCRIPT_DIR / "extract_regex_llm_repair.py")
PYTHON_PATH = "python"                            # 使用当前环境的 python

def main():
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 获取所有PDF文件
    pdf_files = list(Path(PDF_DIR).glob("*.pdf"))
    print(f"找到 {len(pdf_files)} 个PDF文件")
    
    # 记录处理结果
    success = []
    failed = []
    
    for pdf_file in tqdm(pdf_files, desc="处理进度"):
        # 构造输出文件名（保持原名）
        output_name = pdf_file.stem + ".json"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        
        # 检查是否已经处理过
        if os.path.exists(output_path):
            print(f"\n跳过（已存在）: {pdf_file.name}")
            success.append(pdf_file.name)
            continue
        
        print(f"\n\n{'='*60}")
        print(f"正在处理: {pdf_file.name}")
        print(f"{'='*60}")
        
        # 构造命令
        cmd = [
            PYTHON_PATH,
            SCRIPT_PATH,
            "-p", str(pdf_file),
            "-o", output_path
        ]
        
        try:
            # 运行提取脚本
            result = subprocess.run(
                cmd,
                cwd=Path(SCRIPT_PATH).parent,
                capture_output=False,
                text=True
            )
            
            if result.returncode == 0:
                success.append(pdf_file.name)
                print(f"✅ 成功: {pdf_file.name}")
            else:
                failed.append((pdf_file.name, f"退出码: {result.returncode}"))
                print(f"❌ 失败: {pdf_file.name}")
        except Exception as e:
            failed.append((pdf_file.name, str(e)))
            print(f"❌ 异常: {pdf_file.name} - {e}")
    
    # 打印总结
    print("\n" + "="*60)
    print("处理完成汇总")
    print("="*60)
    print(f"✅ 成功: {len(success)} 个")
    print(f"❌ 失败: {len(failed)} 个")
    
    if failed:
        print("\n失败列表:")
        for name, reason in failed:
            print(f"  - {name}: {reason}")
    
    print(f"\n输出目录: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
