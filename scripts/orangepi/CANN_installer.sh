# 本脚本必须以root用户模式运行，否则将无法正常安装CANN
# 本脚本实现以下功能：
# pip下载openi，并在openi中下载CANN

# 检查是否以 root 用户运行
if [ "$(id -u)" -ne 0 ]; then
  echo "此脚本不允许用user模式运行。"
  exit 1
fi
pip config set global.index-url https://mirrors.huaweicloud.com/repository/pypi/simple || {
  echo "此脚本不允许用sudo bash命令运行,请以sudo bash -i命令运行"
}
pip install openi
set -e  # 如果任何命令失败，则立即退出脚本
cd /usr/local/Ascend/ascend-toolkit || {
    echo "Error: Failed to change directory to /usr/local/Ascend/ascend-toolkit. No files will be deleted." >&2
    exit 1
}

# shellcheck disable=SC2035
rm -rf *  # 这将在cd成功后执行，如果cd失败则不会执行到这里,防止文件被误删
cd /home/HwHiAiUser/Downloads || {
    echo "Error: Failed to change directory to /home/HwHiAiUser/Downloads." >&2
    exit 1
}

cd /home/HwHiAiUser/Downloads || exit

openi model download enter/nodule_segmentation cann-toolkit --save_path .
openi model download enter/nodule_segmentation cann-kernels --save_path .

export TOOLKIT_NAME=$(python -c "import os;import fnmatch;prefix_toolkit='Ascend-cann-toolkit'; extension='.run';found_toolkit = any(f.startswith(prefix_toolkit) and f.endswith(extension) for f in os.listdir('.')); toolkit_path = next((f for f in os.listdir(os.getcwd()) if fnmatch.fnmatch(f, 'Ascend-cann-toolkit*')), None); print(toolkit_path)")
export KERNELS_NAME=$(python -c "import os;import fnmatch;prefix_kernels='Ascend-cann-kernels';extension='.run';found_kernels = any(f.startswith(prefix_kernels) and f.endswith(extension) for f in os.listdir('.'));kernels_path = next((f for f in os.listdir(os.getcwd()) if fnmatch.fnmatch(f, 'Ascend-cann-kernels*')), None);print( kernels_path);")

chmod +x ./${TOOLKIT_NAME}
chmod +x ./${KERNELS_NAME}
./${TOOLKIT_NAME} --install --quiet
if mkdir -p /usr/local/Ascend/ascend-toolkit/latest; then
  ./${KERNELS_NAME} --install --quiet
else
  echo "请确认cann已经安装成功"
fi