#!/bin/bash
# AZ语言编译测试脚本 - Linux/macOS

echo "========================================"
echo "AZ Language - Build Test"
echo "========================================"
echo

echo "[1/3] Testing build tool..."
python3 tools/az_build.py examples/hello_compiled.az -o hello_test
if [ $? -ne 0 ]; then
    echo "✗ Build failed"
    exit 1
fi
echo "✓ Build successful"
echo

echo "[2/3] Running compiled program..."
./hello_test
if [ $? -ne 0 ]; then
    echo "✗ Execution failed"
    exit 1
fi
echo "✓ Execution successful"
echo

echo "[3/3] Cleaning up..."
rm hello_test
echo "✓ Cleanup done"
echo

echo "========================================"
echo "All tests passed!"
echo "========================================"
echo
echo "🎉 AZ Language can now compile to executable!"
echo
