@echo off
REM AZ编译器构建脚本 (Windows)

echo =========================================
echo AZ编译器构建脚本
echo =========================================
echo.

REM 检查是否安装了CMake
where cmake >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo 错误: 未找到CMake。请安装CMake。
    echo 下载地址: https://cmake.org/download/
    exit /b 1
)

REM 检查是否安装了Ninja
where ninja >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo 警告: 未找到Ninja。将使用默认生成器。
    echo 建议安装Ninja以加快构建速度。
    set GENERATOR=
) else (
    set GENERATOR=-G Ninja
)

REM 创建构建目录
echo 创建构建目录...
if not exist build mkdir build
cd build

REM 配置CMake
echo 配置CMake...
cmake .. %GENERATOR% ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

if %ERRORLEVEL% NEQ 0 (
    echo 错误: CMake配置失败
    cd ..
    exit /b 1
)

REM 构建
echo.
echo 开始构建...
cmake --build . --config Release

if %ERRORLEVEL% NEQ 0 (
    echo 错误: 构建失败
    cd ..
    exit /b 1
)

REM 完成
echo.
echo 构建完成！
echo.
echo 可执行文件位于: build\tools\Release\az.exe
echo.
echo 要测试编译器，请运行:
echo   build\tools\Release\az.exe ..\examples\hello.az
echo.

cd ..
pause
