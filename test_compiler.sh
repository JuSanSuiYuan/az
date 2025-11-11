#!/bin/bash

echo "========================================"
echo "AZ编译器测试脚本"
echo "========================================"
echo ""

echo "测试1: Hello World"
echo "----------------------------------------"
python3 bootstrap/az_compiler.py examples/hello.az
echo ""

echo "测试2: 变量和运算"
echo "----------------------------------------"
python3 bootstrap/az_compiler.py examples/variables.az
echo ""

echo "测试3: 函数"
echo "----------------------------------------"
python3 bootstrap/az_compiler.py examples/functions.az
echo ""

echo "测试4: 控制流"
echo "----------------------------------------"
python3 bootstrap/az_compiler.py examples/control_flow.az
echo ""

echo "测试5: 斐波那契数列"
echo "----------------------------------------"
python3 bootstrap/az_compiler.py examples/fibonacci.az
echo ""

echo "========================================"
echo "所有测试完成！"
echo "========================================"
