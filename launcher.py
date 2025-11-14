"""
ChatLaw 项目启动脚本
命令格式：chatlaw xxx
"""

import argparse
import subprocess
import sys
import os
import shutil


# =============== 工具函数 ===============

def get_project_root():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "chatlaw")


def get_resources_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "chatlaw", "resources")


def resolve_script_path(role, mode):
    root = get_project_root()

    if role == "client":
        script = os.path.join(root, "client", f"client_{mode}.py")
    elif role == "server":
        script = os.path.join(root, "server", f"server_{mode}.py")
    else:
        raise ValueError("未知角色 client/server")

    if not os.path.exists(script):
        raise FileNotFoundError(f"脚本不存在: {script}")

    return script


def clear_resources():
    """删除 chatlaw/resources 目录，并带确认"""
    path = get_resources_path()

    if not os.path.exists(path):
        print("[提示] resources 目录不存在。")
        return

    confirm = input(f"确认删除目录 {path} ? (y/n): ").strip().lower()
    if confirm == "y":
        shutil.rmtree(path)
        print(f"[OK] 已删除：{path}")
    else:
        print("[取消] 未删除任何内容。")


def run_python_script(script_path, extra_args):
    cmd = [sys.executable, script_path] + extra_args
    print(f"[启动] {cmd}")
    subprocess.run(cmd)


# =============== 主入口 ===============

def main():
    parser = argparse.ArgumentParser(prog="chatlaw", description="ChatLaw 命令行工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    client_parser = subparsers.add_parser("client")
    client_parser.add_argument("mode", choices=["ms", "pt"])
    client_parser.add_argument("extra", nargs=argparse.REMAINDER)

    server_parser = subparsers.add_parser("server")
    server_parser.add_argument("mode", choices=["ms", "pt"])
    server_parser.add_argument("extra", nargs=argparse.REMAINDER)

    subparsers.add_parser("clear_files")

    args = parser.parse_args()

    if args.command == "client":
        run_python_script(resolve_script_path("client", args.mode), args.extra)
    elif args.command == "server":
        run_python_script(resolve_script_path("server", args.mode), args.extra)
    elif args.command == "clear_files":
        clear_resources()


if __name__ == "__main__":
    main()
