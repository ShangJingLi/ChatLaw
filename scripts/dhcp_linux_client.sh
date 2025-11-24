#!/bin/bash

echo "🔍 正在查找以太网设备..."

# 找到 ethernet 类型的设备
ETH_DEV=$(nmcli -t -f DEVICE,TYPE device | grep ":ethernet" | cut -d: -f1 | head -n 1)

if [ -z "$ETH_DEV" ]; then
    echo "❌ 未发现以太网设备 (TYPE=ethernet)"
    exit 1
fi

echo "✅ 发现以太网设备：$ETH_DEV"

echo "🔍 正在查找绑定该设备的连接..."

# 找到该设备对应的 connection 名称
CON_NAME=$(nmcli -t -f NAME,DEVICE connection show | grep ":$ETH_DEV" | cut -d: -f1 | head -n 1)

if [ -z "$CON_NAME" ]; then
    echo "❌ 找不到与设备 $ETH_DEV 绑定的连接"
    exit 1
fi

echo "✅ 找到以太网连接：$CON_NAME"

echo "♻ 正在恢复 DHCP 模式..."

# 按你的系统要求，使用 "" 清空字段（不能使用删除字段语法）
sudo nmcli con mod "$CON_NAME" ipv4.addresses ""
sudo nmcli con mod "$CON_NAME" ipv4.gateway ""
sudo nmcli con mod "$CON_NAME" ipv4.dns ""

# 设置为 DHCP
sudo nmcli con mod "$CON_NAME" ipv4.method auto

echo "🔄 重启以太网连接..."
sudo nmcli con down "$CON_NAME"
sudo nmcli con up "$CON_NAME"

echo "🎉 DHCP 恢复成功 (连接：$CON_NAME)"
