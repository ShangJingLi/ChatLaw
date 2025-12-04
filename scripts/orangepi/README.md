# 香橙派mindspore环境配置脚本

本仓库根据芯片型号不同分别放置三个shell脚本文件，用于自动配置香橙派aipro的mindspore 2.7.1 的Ascend环境

#### 使用说明
 **一.脚本功能说明** 
1. preparation.sh脚本用于实现香橙派环境配置的准备工作，包含以下步骤：
- 配置16G的swap内存，若已经配置，则本步骤跳过
- 创建名为mindspore的conda环境，并配置环境变量使shell重启时自动激活
- 配置control CPU的个数为4，AI CPU的个数为0
- 配置静态IP为192.168.137.100，子网掩码为255.255.255.0
- 上述配置完成后立即重启开发板使配置生效
2. CANN_installer.sh脚本用于下载并安装CANN 8.1RC1的toolkit和kernels包，包含以下步骤：
- 删除当前存在的toolkit包
- 在/home/HwHiAiUser/Downloads目录下载toolkit和kernels包
- 安装toolkit包，经高人指点此处无需手动同意用户协议，静止等待安装即可
- 安装kernels包，经高人指点此处无需手动同意用户协议，静止等待安装即可
3. mindspore_installer.sh脚本用于安装mindspore，配置环境变量并运行run_check()，包含以下步骤：
- 安装昇腾AI开发者工具包
- 安装mindspore必要依赖
- 安装mindspore
- 配置环境变量
- 运行run_check()

 **二.脚本运行方法说明** 

必须按照以下顺序和命令运行脚本,且均在user模式下运行，若user用户和root用户使用错误将导致安装失败


```
bash preparation.sh  # 该脚本运行完毕后会自动重启系统，不用惊慌
```

```
sudo bash -i CANN_installer.sh  # 注意该脚本必须在user用户下以sudo -i命令执行
```

```
bash mindspore_installer.sh 
```

 _三个脚本均包含需要联网执行的命令，故运行过程中需要全程保持网络畅通_ 

 **三.其他说明**

脚本中配置静态IP的步骤是用于从Windows通过以太网（网线）用ssh连接开发板，Windows端的配置以及免密码连接的设置无法

通过脚本进行，需要手动配置。配置方法可以参考我的这篇文章：

【基于官方教程的补充的香橙派mindspore环境配置教程 - CSDN App】
https://blog.csdn.net/weixin_74531285/article/details/143940560?sharetype=blog&shareId=143940560&sharerefer=APP&sharesource=weixin_74531285&sharefrom=link 

只需要关注Windows端的配置以及将Windows的公钥写入开发板的.ssh目录部分内容即可，其他在开发板端的配置脚本可以完成。
