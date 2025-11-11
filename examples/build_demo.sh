#!/bin/bash

# AZ现代化特性演示构建脚本

echo "正在编译AZ现代化特性演示程序..."

# 编译演示程序
azc -o modern_features_demo examples/modern_features_demo.az

if [ $? -eq 0 ]; then
    echo "编译成功!"
    echo "正在运行演示程序..."
    echo "========================"
    ./modern_features_demo
    echo "========================"
    echo "演示程序运行完成!"
else
    echo "编译失败，请检查代码!"
fi