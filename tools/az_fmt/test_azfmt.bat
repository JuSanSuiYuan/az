@echo off
REM AZ fmt测试脚本 - Windows

echo ========================================
echo AZ fmt 测试
echo ========================================
echo.

echo [1/4] 测试基本格式化...
python azfmt.py test_unformatted.az
if %ERRORLEVEL% EQU 0 (
    echo ✓ 格式化成功
) else (
    echo ✗ 格式化失败
    exit /b 1
)
echo.

echo [2/4] 测试检查模式...
python azfmt.py --check test_unformatted.az
if %ERRORLEVEL% EQU 0 (
    echo ✓ 格式正确
) else (
    echo ✗ 需要格式化
)
echo.

echo [3/4] 测试自定义缩进...
python azfmt.py --indent 2 test_unformatted.az
if %ERRORLEVEL% EQU 0 (
    echo ✓ 自定义缩进成功
) else (
    echo ✗ 自定义缩进失败
    exit /b 1
)
echo.

echo [4/4] 测试配置文件...
python azfmt.py --config azfmt.toml test_unformatted.az
if %ERRORLEVEL% EQU 0 (
    echo ✓ 使用配置文件成功
) else (
    echo ✗ 使用配置文件失败
    exit /b 1
)
echo.

echo ========================================
echo AZ fmt 所有测试通过！
echo ========================================
