// AZ编译器 - MLIR降级单元测试

#include "AZ/Backend/MLIRLowering.h"
#include "AZ/Support/Result.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#include <gtest/gtest.h>

using namespace az::backend;
using namespace az::support;

class MLIRLoweringTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 注册必要的方言
        context.getOrLoadDialect<mlir::func::FuncDialect>();
        context.getOrLoadDialect<mlir::arith::ArithDialect>();
    }

    mlir::MLIRContext context;
    llvm::LLVMContext llvmContext;
};

// 测试：简单函数的降级
TEST_F(MLIRLoweringTest, BasicFunctionLowering) {
    // 创建MLIR模块
    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto module = mlir::ModuleOp::create(loc);
    
    builder.setInsertionPointToEnd(module.getBody());
    
    // 创建一个简单的函数: fn test() -> i32 { return 42; }
    auto funcType = builder.getFunctionType({}, {builder.getI32Type()});
    auto func = builder.create<mlir::func::FuncOp>(loc, "test", funcType);
    
    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // 创建常量42
    auto constOp = builder.create<mlir::arith::ConstantIntOp>(
        loc, 42, builder.getI32Type());
    
    // 返回
    builder.create<mlir::func::ReturnOp>(loc, constOp.getResult());
    
    // 执行降级
    MLIRLowering lowering(context);
    auto result = lowering.lower(module, llvmContext);
    
    // 验证结果
    ASSERT_TRUE(result.isOk());
    ASSERT_NE(result.value(), nullptr);
    
    auto llvmModule = std::move(result.value());
    
    // 验证LLVM模块包含函数
    auto* llvmFunc = llvmModule->getFunction("test");
    ASSERT_NE(llvmFunc, nullptr);
    EXPECT_FALSE(llvmFunc->empty());
}

// 测试：算术运算的降级
TEST_F(MLIRLoweringTest, ArithmeticOperationsLowering) {
    // 创建MLIR模块
    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto module = mlir::ModuleOp::create(loc);
    
    builder.setInsertionPointToEnd(module.getBody());
    
    // 创建函数: fn add(a: i32, b: i32) -> i32 { return a + b; }
    auto i32Type = builder.getI32Type();
    auto funcType = builder.getFunctionType({i32Type, i32Type}, {i32Type});
    auto func = builder.create<mlir::func::FuncOp>(loc, "add", funcType);
    
    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // 获取参数
    auto arg0 = entryBlock->getArgument(0);
    auto arg1 = entryBlock->getArgument(1);
    
    // 创建加法
    auto addOp = builder.create<mlir::arith::AddIOp>(loc, arg0, arg1);
    
    // 返回
    builder.create<mlir::func::ReturnOp>(loc, addOp.getResult());
    
    // 执行降级
    MLIRLowering lowering(context);
    auto result = lowering.lower(module, llvmContext);
    
    // 验证结果
    ASSERT_TRUE(result.isOk());
    ASSERT_NE(result.value(), nullptr);
    
    auto llvmModule = std::move(result.value());
    
    // 验证LLVM模块包含函数
    auto* llvmFunc = llvmModule->getFunction("add");
    ASSERT_NE(llvmFunc, nullptr);
    EXPECT_FALSE(llvmFunc->empty());
    
    // 验证函数签名
    EXPECT_EQ(llvmFunc->arg_size(), 2);
}

// 测试：多个函数的降级
TEST_F(MLIRLoweringTest, MultipleFunctionsLowering) {
    // 创建MLIR模块
    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto module = mlir::ModuleOp::create(loc);
    
    builder.setInsertionPointToEnd(module.getBody());
    
    // 创建第一个函数
    auto funcType1 = builder.getFunctionType({}, {builder.getI32Type()});
    auto func1 = builder.create<mlir::func::FuncOp>(loc, "func1", funcType1);
    auto* block1 = func1.addEntryBlock();
    builder.setInsertionPointToStart(block1);
    auto const1 = builder.create<mlir::arith::ConstantIntOp>(
        loc, 10, builder.getI32Type());
    builder.create<mlir::func::ReturnOp>(loc, const1.getResult());
    
    // 创建第二个函数
    builder.setInsertionPointToEnd(module.getBody());
    auto funcType2 = builder.getFunctionType({}, {builder.getI32Type()});
    auto func2 = builder.create<mlir::func::FuncOp>(loc, "func2", funcType2);
    auto* block2 = func2.addEntryBlock();
    builder.setInsertionPointToStart(block2);
    auto const2 = builder.create<mlir::arith::ConstantIntOp>(
        loc, 20, builder.getI32Type());
    builder.create<mlir::func::ReturnOp>(loc, const2.getResult());
    
    // 执行降级
    MLIRLowering lowering(context);
    auto result = lowering.lower(module, llvmContext);
    
    // 验证结果
    ASSERT_TRUE(result.isOk());
    ASSERT_NE(result.value(), nullptr);
    
    auto llvmModule = std::move(result.value());
    
    // 验证两个函数都存在
    ASSERT_NE(llvmModule->getFunction("func1"), nullptr);
    ASSERT_NE(llvmModule->getFunction("func2"), nullptr);
}

// 测试：错误处理 - 空模块
TEST_F(MLIRLoweringTest, EmptyModuleLowering) {
    // 创建空的MLIR模块
    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto module = mlir::ModuleOp::create(loc);
    
    // 执行降级
    MLIRLowering lowering(context);
    auto result = lowering.lower(module, llvmContext);
    
    // 空模块也应该成功降级
    ASSERT_TRUE(result.isOk());
    ASSERT_NE(result.value(), nullptr);
}

// 主函数
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
