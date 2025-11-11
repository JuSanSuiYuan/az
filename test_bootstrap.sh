#!/bin/bash
# AZ语言自举测试脚本 (Linux/macOS)

echo "========================================"
echo "AZ语言自举测试"
echo "========================================"
echo ""

# 检查Python
echo "[1/6] 检查Python安装..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3未安装"
    echo "请安装Python 3.7+"
    exit 1
fi
echo "✅ Python已安装: $(python3 --version)"

# 检查Clang
echo "[2/6] 检查Clang安装..."
if ! command -v clang &> /dev/null; then
    echo "❌ Clang未安装，无法编译C代码"
    echo "请安装LLVM/Clang: https://releases.llvm.org/download.html"
    exit 1
else
    echo "✅ Clang已安装: $(clang --version | head -n1)"
fi

# 测试解释执行
echo "[3/6] 测试解释执行模式..."
python3 bootstrap/az_compiler.py examples/hello.az
if [ $? -ne 0 ]; then
    echo "❌ 解释执行失败"
    exit 1
fi
echo "✅ 解释执行成功"

# 测试C代码生成
echo "[4/6] 测试C代码生成..."
python3 bootstrap/az_compiler.py examples/test_codegen.az --emit-c -o test_output.c
if [ $? -ne 0 ]; then
    echo "❌ C代码生成失败"
    exit 1
fi
echo "✅ C代码生成成功"

# 使用Clang编译生成的C代码
echo "[5/6] 使用Clang编译生成的C代码..."
clang test_output.c -o test_output
if [ $? -ne 0 ]; then
    echo "❌ C代码编译失败"
    exit 1
fi
echo "✅ C代码编译成功"

# 运行生成的程序
echo "[6/6] 运行生成的程序..."
./test_output
if [ $? -ne 0 ]; then
    echo "❌ 程序运行失败"
    exit 1
fi
echo "✅ 程序运行成功"

echo ""
echo "========================================"
echo "✅ 所有测试通过！"
echo "========================================"
echo ""
echo "下一步:"
echo "1. 创建最小化编译器 (compiler/minimal/)"
echo "2. 实现第一次自举"
echo "3. 验证自举成功"
echo ""

# 清理
rm -f test_output.c test_output

exit 0
