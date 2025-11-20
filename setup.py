from setuptools import setup, find_packages
import os

def read_requirements():
    if not os.path.exists("requirements.txt"):
        return []
    with open("requirements.txt", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="chatlaw",
    version="0.1.0",

    packages=find_packages(include=["chatlaw", "chatlaw.*"]),

    # 递归包含 chatlaw 目录下的所有文件
    package_data={
        "chatlaw": ["**/*"],
    },

    include_package_data=True,
    install_requires=read_requirements(),

    entry_points={
        "console_scripts": [
            "chatlaw = launcher:main",
        ],
    },

    py_modules=[
        "launcher",
    ],
)
