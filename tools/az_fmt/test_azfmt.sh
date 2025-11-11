#!/bin/bash
# AZ fmt测试脚本 - Linux/macOS

echo "========================================"
echo "AZ fmt 测试"
echo "========================================"
echo

echo "[1/4] 测试基本格式化..."
python3 azfmt.py test_unformatted.az
if [ $? -eq 0 ]; then
    echo "✓ 格式化成功"
else
    echo "✗ 格式化失败"
    exit 1
fi
echo

echo "[2/4] 测试检查模式..."
python3 azfmt.py --check test_unformatted.az
if [ $? -eq 0 ]; then
    echo "✓ 格式正确"
else
    echo "✗ 需要格式化"
fi
echo

echo "[3/4] 测试自定义缩进..."
python3 azfmt.py --indent 2 test_unformatted.az
if [ $? -eq 0 ]; then
    echo "✓ 自定义缩进成功"
else
    echo "✗ 自定义缩进失败"
    exit 1
fi
echo

echo "[4/4] 测试配置文件..."
python3 azfmt.py --config azfmt.toml test_unformatted.az
if [ $? -eq 0 ]; then
    echo "✓ 使用配置文件成功"
else
    echo "✗ 使用配置文件失败"
    exit 1
fi
echo

echo "========================================"
echo "AZ fmt 所有测试通过！"
echo "========================================"
