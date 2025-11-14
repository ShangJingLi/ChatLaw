from setuptools import setup, find_packages
import os

def read_requirements():
    if not os.path.exists("requirements.txt"):
        return []
    with open("requirements.txt") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="chatlaw",
    version="0.1.0",

    packages=find_packages(include=["chatlaw", "chatlaw.*"]),

    package_data={
        "chatlaw": [
            "client/*.py",
            "server/*.py"
        ]
    },

    include_package_data=True,
    install_requires=read_requirements(),

    # 两个可执行命令：
    #   chatlaw         → 正常运行入口
    #   chatlaw-uninstall-hook → pip uninstall 时自动执行
    entry_points={
        "console_scripts": [
            "chatlaw = launcher:main",
            "chatlaw-uninstall-hook = chatlaw_uninstall:main",
        ],
    },

    py_modules=[
        "launcher",
        "chatlaw_uninstall"
    ],
)
