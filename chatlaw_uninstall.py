"""
卸载 ChatLaw 前自动执行：删除 chatlaw/resources 目录
由 pip uninstall chatlaw 自动调用
"""

import os
import shutil

def main():
    base = os.path.dirname(os.path.abspath(__file__))  # ChatLaw/
    resources_dir = os.path.join(base, "chatlaw", "resources")

    if os.path.exists(resources_dir):
        print(f"[chatlaw uninstall] 正在删除资源目录：{resources_dir}")
        shutil.rmtree(resources_dir, ignore_errors=True)
    else:
        print("[chatlaw uninstall] 没有需要清理的资源目录。")

if __name__ == "__main__":
    main()
