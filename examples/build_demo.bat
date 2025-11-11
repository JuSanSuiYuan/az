@echo off

REM AZ现代化特性演示构建脚本 (Windows版本)

echo 正在编译AZ现代化特性演示程序...

REM 编译演示程序
azc -o modern_features_demo.exe examples/modern_features_demo.az

if %errorlevel% == 0 (
    echo 编译成功!
    echo 正在运行演示程序...
    echo ========================
    modern_features_demo.exe
    echo ========================
    echo 演示程序运行完成!
) else (
    echo 编译失败，请检查代码!
)