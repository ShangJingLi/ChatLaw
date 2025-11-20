"""
ChatLaw 资源清理模块
用于删除 chatlaw/resources 目录下的模型、tokenizer、RAG 等缓存文件。
"""

import os
import shutil


def get_resources_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")


def main():
    resource_dir = get_resources_path()

    if not os.path.exists(resource_dir):
        print(f"[提示] 资源目录不存在：{resource_dir}")
        return

    targets = [
        "tokenizer",
        "llm",
        "video_model",
        "RAG",
        "openi_resource.version",
    ]

    # ===== 打印存在的资源项 =====
    print("将要清理以下资源：")
    existing_targets = []

    for target in targets:
        if os.path.exists(os.path.join(resource_dir, target)):
            existing_targets.append(target)

    if existing_targets:
        for t in existing_targets:
            print(f"  - {t}")
    else:
        print("  （无可清理资源）")

    print(f"\n资源根目录：{resource_dir}")

    # ===== 若无可清理项，提前退出 =====
    if not existing_targets:
        return

    confirm = input("\n确认删除这些资源吗？(y/n): ").strip().lower()
    if confirm != "y":
        print("[取消] 未执行任何删除操作。")
        return

    # ===== 执行删除 =====
    for target in existing_targets:
        path = os.path.join(resource_dir, target)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
                print(f"✅ 目录 {path} 已删除")
            else:
                os.remove(path)
                print(f"✅ 文件 {path} 已删除")
        except Exception as e:
            print(f"❌ 删除 {path} 时发生异常：{e}")

    print("清理完成！")


if __name__ == "__main__":
    main()
