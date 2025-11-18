#!/bin/bash
# pylint检查脚本，用于进行代码风格检查
# Windows系统下无法直接运行本脚本，请输入如下命令进行代码风格检查：
# pylint --rcfile=.pylint.conf chatlaw
NUM_CORES=$(nproc)
pylint --jobs=$NUM_CORES --rcfile=.pylint.conf chatlaw