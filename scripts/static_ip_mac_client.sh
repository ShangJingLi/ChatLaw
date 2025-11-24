#!/bin/bash

# =======================
# 你要配置的静态 IP 信息
# =======================
STATIC_IP="192.168.137.101"
SUBNET_MASK="255.255.255.0"
GATEWAY="192.168.137.1"
DNS_SERVER="8.8.8.8"

echo "🔍 正在查找以太网网络服务名称..."

# 1. 找到 Ethernet / Thunderbolt Ethernet / USB Ethernet 等服务
ETH_IFACE=$(
  networksetup -listallnetworkservices 2>/dev/null \
  | grep -Ei "ethernet|thunderbolt|usb" \
  | grep -v "Wi-Fi" \
  | head -n 1
)

if [ -z "$ETH_IFACE" ]; then
    echo "❌ 未找到以太网服务（Ethernet/Thunderbolt/USB），无法继续。"
    exit 1
fi

echo "✅ 找到以太网服务：$ETH_IFACE"

echo "🔍 获取对应的底层设备名称..."
DEVICE_NAME=$(networksetup -getinfo "$ETH_IFACE" | grep "Device:" | awk '{print $2}')

if [ -z "$DEVICE_NAME" ]; then
    echo "❌ 无法获取底层设备名称，对应接口不存在?"
    exit 1
fi

echo "📌 底层设备：$DEVICE_NAME"

echo "⚙️ 正在设置静态 IP..."

sudo networksetup -setmanual "$ETH_IFACE" $STATIC_IP $SUBNET_MASK $GATEWAY
sudo networksetup -setdnsservers "$ETH_IFACE" $DNS_SERVER

echo "🔄 正在重启以太网接口..."
sudo ifconfig "$DEVICE_NAME" down
sudo ifconfig "$DEVICE_NAME" up

echo "🎉 成功！以太网 $ETH_IFACE 已设置为静态 IP：$STATIC_IP"
