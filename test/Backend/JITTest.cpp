// AZ编译器 - JIT编译器单元测试

#include "AZ/Backend/JIT.h"
#include "AZ/Support/Result.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <gtest/gtest.h>

using namespace az::backend;
using namespace az::support;

class JITTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 注册必要的方言
        context.getOrLoadDialect<mlir::func::FuncDialect>();
        context.getOrLoadDialect<mlir::arith::ArithDialect>();
    }

    // 创建一个简单的MLIR模块: fn main() -> i32 { return 42; }
    mlir::ModuleOp createSimpleModule() {
        mlir::OpBuilder builder(&context);
        auto loc = builder.getUnknownLoc();
        auto module = mlir::ModuleOp::create(loc);
        
        builder.setInsertionPointToEnd(module.getBody());
        
        // 创建main函数
        auto funcType = builder.getFunctionType({}, {builder.getI32Type()});
        auto func = builder.create<mlir::func::FuncOp>(loc, "main", funcType);
        
        auto* entryBlock = func.addEntryBlock();
        builder.setInsertionPointToStart(entryBlock);
        
        auto constOp = builder.create<mlir::arith::ConstantIntOp>(
            loc, 42, builder.getI32Type());
        
        builder.create<mlir::func::ReturnOp>(loc, constOp.getResult());
        
        return module;
    }

    // 创建一个带参数的函数: fn add(a: i32, b: i32) -> i32 { return a + b; }
    mlir::ModuleOp createAddModule() {
        mlir::OpBuilder builder(&context);
        auto loc = builder.getUnknownLoc();
        auto module = mlir::ModuleOp::create(loc);
        
        builder.setInsertionPointToEnd(module.getBody());
        
        auto i32Type = builder.getI32Type();
        auto funcType = builder.getFunctionType({i32Type, i32Type}, {i32Type});
        auto func = builder.create<mlir::func::FuncOp>(loc, "add", funcType);
        
        auto* entryBlock = func.addEntryBlock();
        builder.setInsertionPointToStart(entryBlock);
        
        auto arg0 = entryBlock->getArgument(0);
        auto arg1 = entryBlock->getArgument(1);
        
        auto addOp = builder.create<mlir::arith::AddIOp>(loc, arg0, arg1);
        
        builder.create<mlir::func::ReturnOp>(loc, addOp.getResult());
        
        return module;
    }

    mlir::MLIRContext context;
};

// 测试：JIT编译并运行简单函数
TEST_F(JITTest, CompileAndRunSimpleFunction) {
    auto module = createSimpleModule();
    
    JITCompiler jit;
    auto result = jit.compileAndRun(module, {});
    
    ASSERT_TRUE(result.isOk());
    EXPECT_EQ(result.value(), 42);
}

// 测试：编译单个函数
TEST_F(JITTest, CompileFunction) {
    auto module = createAddModule();
    
    JITCompiler jit;
    auto result = jit.compileFunction(module, "add");
    
    ASSERT_TRUE(result.isOk());
    ASSERT_NE(result.value(), nullptr);
    
    // 调用函数
    auto* addFunc = reinterpret_cast<int(*)(int, int)>(result.value());
    int sum = addFunc(10, 20);
    
    EXPECT_EQ(sum, 30);
}

// 测试：多次JIT编译
TEST_F(JITTest, MultipleCompilations) {
    JITCompiler jit;
    
    // 第一次编译
    auto module1 = createSimpleModule();
    auto result1 = jit.compileAndRun(module1, {});
    ASSERT_TRUE(result1.isOk());
    EXPECT_EQ(result1.value(), 42);
    
    // 第二次编译（新的JIT实例）
    JITCompiler jit2;
    auto module2 = createSimpleModule();
    auto result2 = jit2.compileAndRun(module2, {});
    ASSERT_TRUE(result2.isOk());
    EXPECT_EQ(result2.value(), 42);
}

// 测试：错误处理 - 找不到main函数
TEST_F(JITTest, NoMainFunction) {
    // 创建一个没有main函数的模块
    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto module = mlir::ModuleOp::create(loc);
    
    builder.setInsertionPointToEnd(module.getBody());
    
    auto funcType = builder.getFunctionType({}, {builder.getI32Type()});
    auto func = builder.create<mlir::func::FuncOp>(loc, "not_main", funcType);
    
    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    auto constOp = builder.create<mlir::arith::ConstantIntOp>(
        loc, 42, builder.getI32Type());
    
    builder.create<mlir::func::ReturnOp>(loc, constOp.getResult());
    
    JITCompiler jit;
    auto result = jit.compileAndRun(module, {});
    
    // 应该返回错误
    EXPECT_TRUE(result.isErr());
}

// 测试：错误处理 - 找不到指定函数
TEST_F(JITTest, FunctionNotFound) {
    auto module = createSimpleModule();
    
    JITCompiler jit;
    auto result = jit.compileFunction(module, "nonexistent");
    
    // 应该返回错误
    EXPECT_TRUE(result.isErr());
}

// 测试：空模块
TEST_F(JITTest, EmptyModule) {
    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto module = mlir::ModuleOp::create(loc);
    
    JITCompiler jit;
    auto result = jit.compileAndRun(module, {});
    
    // 应该返回错误（没有main函数）
    EXPECT_TRUE(result.isErr());
}

// 主函数
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
