#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

if [ ! -d ".venv" ]; then
  echo "未找到 .venv，请先创建虚拟环境"
  exit 1
fi

source .venv/bin/activate
python launcher_gui.py