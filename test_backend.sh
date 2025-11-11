#!/bin/bash
# AZ编译器后端测试脚本 (Linux/macOS)

echo "========================================"
echo "AZ编译器后端测试"
echo "========================================"
echo

# 检查构建目录
if [ ! -d "build" ]; then
    echo "错误: build目录不存在"
    echo "请先运行: cmake -B build"
    exit 1
fi

# 进入构建目录
cd build

echo "[1/4] 运行MLIR降级测试..."
echo "----------------------------------------"
ctest -R MLIRLoweringTest --output-on-failure
if [ $? -ne 0 ]; then
    echo "MLIR降级测试失败!"
    cd ..
    exit 1
fi
echo

echo "[2/4] 运行优化器测试..."
echo "----------------------------------------"
ctest -R OptimizerTest --output-on-failure
if [ $? -ne 0 ]; then
    echo "优化器测试失败!"
    cd ..
    exit 1
fi
echo

echo "[3/4] 运行代码生成器测试..."
echo "----------------------------------------"
ctest -R CodeGeneratorTest --output-on-failure
if [ $? -ne 0 ]; then
    echo "代码生成器测试失败!"
    cd ..
    exit 1
fi
echo

echo "[4/4] 运行所有后端测试..."
echo "----------------------------------------"
ctest -R "MLIR|Optimizer|CodeGenerator" --output-on-failure
if [ $? -ne 0 ]; then
    echo "部分测试失败!"
    cd ..
    exit 1
fi

cd ..

echo
echo "========================================"
echo "所有后端测试通过! ✓"
echo "========================================"
