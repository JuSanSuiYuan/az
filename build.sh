#!/bin/bash
# AZ编译器构建脚本

set -e

echo "========================================="
echo "AZ编译器构建脚本"
echo "========================================="
echo ""

# 检查LLVM
if ! command -v llvm-config &> /dev/null; then
    echo "错误: 未找到LLVM。请安装LLVM 17或更高版本。"
    echo "Ubuntu/Debian: sudo apt install llvm-17-dev"
    echo "macOS: brew install llvm@17"
    exit 1
fi

LLVM_VERSION=$(llvm-config --version | cut -d. -f1)
if [ "$LLVM_VERSION" -lt 17 ]; then
    echo "错误: LLVM版本过低。需要17或更高版本，当前版本: $LLVM_VERSION"
    exit 1
fi

echo "找到LLVM版本: $(llvm-config --version)"
echo ""

# 创建构建目录
echo "创建构建目录..."
mkdir -p build
cd build

# 配置CMake
echo "配置CMake..."
cmake .. \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_DIR=$(llvm-config --cmakedir) \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# 构建
echo ""
echo "开始构建..."
cmake --build . -j$(nproc)

# 安装（可选）
echo ""
echo "构建完成！"
echo ""
echo "可执行文件位于: build/tools/az"
echo ""
echo "要安装到系统，请运行:"
echo "  sudo cmake --install build --prefix /usr/local"
echo ""
echo "要测试编译器，请运行:"
echo "  ./build/tools/az ../examples/hello.az"
