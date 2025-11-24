#!/bin/bash

# ===========================
# 可自定义的静态 IP 配置
# ===========================
STATIC_IP="192.168.137.101"
SUBNET_MASK="24"
GATEWAY="192.168.137.1"
DNS_SERVER="8.8.8.8"

echo "🔍 正在查找以太网设备..."

# 1) 通过 nmcli 找到 TYPE=ethernet 的设备（不会误伤 Wi-Fi）
ETH_DEV=$(nmcli -t -f DEVICE,TYPE device | grep ":ethernet" | cut -d: -f1 | head -n 1)

if [ -z "$ETH_DEV" ]; then
    echo "❌ 未发现类型为 ethernet 的网卡设备"
    exit 1
fi

echo "✅ 发现以太网设备：$ETH_DEV"

echo "🔍 正在查找与该设备绑定的连接..."

# 2) 找到与该设备绑定的 connection 名称
CON_NAME=$(nmcli -t -f NAME,DEVICE connection show | grep ":$ETH_DEV" | cut -d: -f1 | head -n 1)

if [ -z "$CON_NAME" ]; then
    echo "❌ 找不到与设备 $ETH_DEV 绑定的连接"
    exit 1
fi

echo "✅ 找到以太网连接：$CON_NAME"

echo "⚙ 正在设置静态 IP..."

# 3) 设置静态 IP、掩码、网关、DNS
sudo nmcli con mod "$CON_NAME" ipv4.addresses "$STATIC_IP/$SUBNET_MASK"
sudo nmcli con mod "$CON_NAME" ipv4.gateway "$GATEWAY"
sudo nmcli con mod "$CON_NAME" ipv4.dns "$DNS_SERVER"
sudo nmcli con mod "$CON_NAME" ipv4.method manual

# 4) 重启连接
echo "🔄 正在重启网络..."
sudo nmcli con down "$CON_NAME"
sudo nmcli con up "$CON_NAME"

echo "🎉 成功！$CON_NAME 已设置静态 IP：$STATIC_IP"
