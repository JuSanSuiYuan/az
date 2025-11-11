@echo off
REM AZ编译器后端测试脚本 (Windows)

echo ========================================
echo AZ编译器后端测试
echo ========================================
echo.

REM 检查构建目录
if not exist build (
    echo 错误: build目录不存在
    echo 请先运行: cmake -B build
    exit /b 1
)

REM 进入构建目录
cd build

echo [1/4] 运行MLIR降级测试...
echo ----------------------------------------
ctest -R MLIRLoweringTest --output-on-failure
if %ERRORLEVEL% neq 0 (
    echo MLIR降级测试失败!
    cd ..
    exit /b 1
)
echo.

echo [2/4] 运行优化器测试...
echo ----------------------------------------
ctest -R OptimizerTest --output-on-failure
if %ERRORLEVEL% neq 0 (
    echo 优化器测试失败!
    cd ..
    exit /b 1
)
echo.

echo [3/4] 运行代码生成器测试...
echo ----------------------------------------
ctest -R CodeGeneratorTest --output-on-failure
if %ERRORLEVEL% neq 0 (
    echo 代码生成器测试失败!
    cd ..
    exit /b 1
)
echo.

echo [4/4] 运行所有后端测试...
echo ----------------------------------------
ctest -R "MLIR|Optimizer|CodeGenerator" --output-on-failure
if %ERRORLEVEL% neq 0 (
    echo 部分测试失败!
    cd ..
    exit /b 1
)

cd ..

echo.
echo ========================================
echo 所有后端测试通过! ✓
echo ========================================
