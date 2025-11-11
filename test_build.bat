@echo off
REM AZ语言编译测试脚本 - Windows

echo ========================================
echo AZ Language - Build Test
echo ========================================
echo.

echo [1/3] Testing build tool...
python tools/az_build.py examples/hello_compiled.az -o hello_test.exe
if %ERRORLEVEL% NEQ 0 (
    echo ✗ Build failed
    exit /b 1
)
echo ✓ Build successful
echo.

echo [2/3] Running compiled program...
hello_test.exe
if %ERRORLEVEL% NEQ 0 (
    echo ✗ Execution failed
    exit /b 1
)
echo ✓ Execution successful
echo.

echo [3/3] Cleaning up...
del hello_test.exe
echo ✓ Cleanup done
echo.

echo ========================================
echo All tests passed!
echo ========================================
echo.
echo 🎉 AZ Language can now compile to executable!
echo.
