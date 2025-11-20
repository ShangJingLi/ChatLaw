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
    """
    功能：
        根据执行角色（client/server）与模式（mode）动态生成并返回对应 Python
        启动脚本的绝对路径。该函数用于启动器（launcher）中，用于精确定位客户端
        或服务器端不同模式的运行脚本位置，例如：
            - client/client_normal.py
            - server/server_local.py
        等等。

        函数会检查生成的路径是否存在，若不存在则抛出异常，确保启动器不会执行
        无效路径。

    Args:
        role (str): 执行角色，必须是 `"client"` 或 `"server"`。
        mode (str): 运行模式，用于匹配对应脚本名称，例如 "normal"、"local" 等。

    Inputs:
        - **role**: 角色类别，用于确定脚本目录 client/ 或 server/。
        - **mode**: 模式名称，用于拼接脚本文件名。
        - 函数依赖全局工具：
            - **get_project_root()**: 返回项目根目录路径。
            - **os.path**: 用于路径拼接与存在性检查。

    Outputs:
        str: 返回匹配到的脚本的绝对路径。如果路径存在，则必定是一个可执行的 .py 文件。

    Raises:
        ValueError: 若 role 不是 "client" 或 "server"。
        FileNotFoundError: 若拼接出的脚本路径不存在。
    """
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


def run_python_script(script_path, extra_args):
    cmd = [sys.executable, script_path] + extra_args
    print(f"[Execute] {cmd}")
    subprocess.run(cmd)


# =============== 主入口 ===============

def main():
    """
    功能：
        ChatLaw 命令行入口函数（CLI）。
        用于解析用户在命令行输入的指令，并根据不同子命令执行对应的功能模块。
        整体功能包括：
            - 启动 ChatLaw 客户端程序（client ms/pt）
            - 启动 ChatLaw 服务端程序（server ms/pt）
            - 清理资源目录（clear）

        该函数负责构建 argparse 命令结构、解析用户输入，并调用相应的脚本启动逻辑。

    Inputs:
        本函数无显式输入参数，但依赖直接执行命令行时传入的参数，如：
            chatlaw client ms  --SERVER_IP "127.0.0.1"
            chatlaw server pt
            chatlaw clear

        内部依赖以下外部函数/模块：
            - **resolve_script_path(role, mode)**：根据角色与模式解析脚本路径。
            - **run_python_script(path, args)**：用于启动对应的 Python 子进程。
            - **clear_resources()**：清理资源目录。
            - **argparse**：解析命令行参数。
            - **get_project_root() / config**：用于路径和环境配置。

    Outputs:
        无显式返回值。
        根据不同命令执行不同效果，例如启动子进程、清理资源等。

    Raises:
        argparse 会在命令格式错误时自动抛出 SystemExit。
        本函数本身不显式抛出异常。
    """
    parser = argparse.ArgumentParser(
        prog="chatlaw",
        description=(
            "ChatLaw 命令行工具\n"
            "\n"
            "可用指令包括：\n"
            "  chatlaw client <ms|pt> [其他参数]    启动客户端模式\n"
            "  chatlaw server <ms|pt> [其他参数]    启动服务端模式\n"
            "  chatlaw clear                 清理 resources 文件夹\n"
            "\n"
            "说明：\n"
            "  - ms  表示使用MindNLP版本\n"
            "  - pt  表示使用Transformers版本\n"
            "  - clear 会在删除前执行交互式确认\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    client_parser = subparsers.add_parser(
        "client",
        description=(
            "客户端模式：\n"
            "  用于启动 ChatLaw 客户端程序。\n"
            "  支持两种模式：ms（MindNLP版本）与 pt（Transformers版本）。\n"
        )
    )
    client_parser.add_argument(
        "mode", choices=["ms", "pt"], help="启动模式：ms=MindNLP版本, pt=Transformers版本"
    )
    client_parser.add_argument("extra", nargs=argparse.REMAINDER, help="传递给客户端脚本的额外参数")

    server_parser = subparsers.add_parser(
        "server",
        description=(
            "服务端模式：\n"
            "  用于启动 ChatLaw 服务端进程。\n"
            "  同样支持 ms（MindNLP版本）与 pt（Transformers版本）两种框架实现方式。\n"
        )
    )
    server_parser.add_argument(
        "mode", choices=["ms", "pt"], help="启动模式：ms=MindNLP版本, pt=Transformers版本"
    )
    server_parser.add_argument("extra", nargs=argparse.REMAINDER, help="传递给服务端脚本的额外参数")

    subparsers.add_parser(
        "clear_files",
        description=(
            "清理 chatlaw/resources 目录：\n"
            "  删除资源缓存文件夹，删除前会进行确认。"
        )
    )

    args = parser.parse_args()

    if args.command == "client":
        run_python_script(resolve_script_path("client", args.mode), args.extra)
    elif args.command == "server":
        run_python_script(resolve_script_path("server", args.mode), args.extra)
    elif args.command == "clear":
        run_python_script(os.path.join(get_project_root(), "clear_files.py"), [])


if __name__ == "__main__":
    main()
