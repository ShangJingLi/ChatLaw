#!/bin/bash

echo "🔍 正在查找以太网网络服务名称..."

ETH_IFACE=$(
  networksetup -listallnetworkservices 2>/dev/null \
  | grep -Ei "ethernet|thunderbolt|usb" \
  | grep -v "Wi-Fi" \
  | head -n 1
)

if [ -z "$ETH_IFACE" ]; then
    echo "❌ 未找到以太网服务（Ethernet/Thunderbolt/USB），无法恢复 DHCP。"
    exit 1
fi

echo "✅ 找到以太网服务：$ETH_IFACE"

echo "🔍 获取底层设备名称..."
DEVICE_NAME=$(networksetup -getinfo "$ETH_IFACE" | grep "Device:" | awk '{print $2}')

if [ -z "$DEVICE_NAME" ];then
    echo "❌ 无法获取底层设备名称。"
    exit 1
fi

echo "📌 底层设备：$DEVICE_NAME"
echo "♻️ 正在恢复 DHCP..."

sudo networksetup -setdhcp "$ETH_IFACE"
sudo networksetup -setdnsservers "$ETH_IFACE" "Empty"

echo "🔄 正在重启以太网接口..."
sudo ifconfig "$DEVICE_NAME" down
sudo ifconfig "$DEVICE_NAME" up

echo "🎉 已成功恢复 DHCP（接口：$ETH_IFACE）"
