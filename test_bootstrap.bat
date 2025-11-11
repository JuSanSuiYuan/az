@echo off
REM AZ语言自举测试脚本 (Windows)

echo ========================================
echo AZ语言自举测试
echo ========================================
echo.

REM 检查Python
echo [1/6] 检查Python安装...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python未安装或未添加到PATH
    echo 请运行: winget install Python.Python.3.12
    exit /b 1
)
echo ✅ Python已安装

REM 检查Clang
echo [2/6] 检查Clang安装...
clang --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Clang未安装，无法编译C代码
    echo 请从 https://releases.llvm.org/download.html 下载安装LLVM
    exit /b 1
) else (
    echo ✅ Clang已安装
)

REM 测试解释执行
echo [3/6] 测试解释执行模式...
python bootstrap\az_compiler.py examples\hello.az
if %errorlevel% neq 0 (
    echo ❌ 解释执行失败
    exit /b 1
)
echo ✅ 解释执行成功

REM 测试C代码生成
echo [4/6] 测试C代码生成...
python bootstrap\az_compiler.py examples\test_codegen.az --emit-c -o test_output.c
if %errorlevel% neq 0 (
    echo ❌ C代码生成失败
    exit /b 1
)
echo ✅ C代码生成成功

REM 使用Clang编译生成的C代码
echo [5/6] 使用Clang编译生成的C代码...
clang test_output.c -o test_output.exe
if %errorlevel% neq 0 (
    echo ❌ C代码编译失败
    exit /b 1
)
echo ✅ C代码编译成功

REM 运行生成的程序
echo [6/6] 运行生成的程序...
test_output.exe
if %errorlevel% neq 0 (
    echo ❌ 程序运行失败
    exit /b 1
)
echo ✅ 程序运行成功

echo.
echo ========================================
echo ✅ 所有测试通过！
echo ========================================
echo.
echo 下一步:
echo 1. 创建最小化编译器 (compiler/minimal/)
echo 2. 实现第一次自举
echo 3. 验证自举成功
echo.

REM 清理
del test_output.c test_output.exe 2>nul

exit /b 0
