#!/usr/bin/env bash
# ChatLaw 服务端启动脚本（Linux + NVIDIA GPU，conda 环境）。
#
# 为什么需要这个包装脚本：
#   vLLM 依赖 torchcodec，torchcodec 用 ctypes 独立加载其扩展，需要在动态
#   加载器搜索路径上找到 CUDA 运行库（典型如 libnvrtc.so.13）。conda 安装的
#   cuda-toolkit 把这些库放在 $CONDA_PREFIX/lib，但默认不在 LD_LIBRARY_PATH
#   上，导致 `import vllm` 失败、auto 后端静默回退到 transformers。
#
#   这里在“启动层”把 $CONDA_PREFIX/lib 追加进 LD_LIBRARY_PATH（从当前 conda
#   环境动态派生，不含硬编码绝对路径），使部署到不同机器/环境时自动适配。
#
# 用法：
#   conda activate chatlaw
#   ./run_server.sh                # 等价于 python launcher.py server
#   ./run_server.sh --SERVER_IP 0.0.0.0   # 透传参数给 launcher

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "[run_server] 未检测到激活的 conda 环境，请先执行： conda activate chatlaw" >&2
  exit 1
fi

# torchcodec / vLLM 运行期依赖的 CUDA 库目录
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "[run_server] CONDA_PREFIX=$CONDA_PREFIX"
echo "[run_server] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

exec python launcher.py server "$@"
