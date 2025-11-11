// AZ编译器 - LLVM优化器单元测试

#include "AZ/Backend/Optimizer.h"
#include "AZ/Support/Result.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"

#include <gtest/gtest.h>

using namespace az::backend;
using namespace az::support;

class OptimizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        module = std::make_unique<llvm::Module>("test", context);
    }

    // 创建一个简单的测试函数
    llvm::Function* createSimpleFunction() {
        auto* funcType = llvm::FunctionType::get(
            llvm::Type::getInt32Ty(context),
            {llvm::Type::getInt32Ty(context)},
            false
        );
        
        auto* func = llvm::Function::Create(
            funcType,
            llvm::Function::ExternalLinkage,
            "test_func",
            module.get()
        );
        
        auto* entry = llvm::BasicBlock::Create(context, "entry", func);
        llvm::IRBuilder<> builder(entry);
        
        // 创建一些可以优化的代码
        auto* arg = func->arg_begin();
        auto* const1 = builder.getInt32(1);
        auto* add1 = builder.CreateAdd(arg, const1);
        auto* const2 = builder.getInt32(2);
        auto* add2 = builder.CreateAdd(add1, const2);
        
        builder.CreateRet(add2);
        
        return func;
    }

    // 创建一个包含死代码的函数
    llvm::Function* createFunctionWithDeadCode() {
        auto* funcType = llvm::FunctionType::get(
            llvm::Type::getInt32Ty(context),
            {},
            false
        );
        
        auto* func = llvm::Function::Create(
            funcType,
            llvm::Function::ExternalLinkage,
            "dead_code_func",
            module.get()
        );
        
        auto* entry = llvm::BasicBlock::Create(context, "entry", func);
        llvm::IRBuilder<> builder(entry);
        
        // 创建一些死代码
        auto* const1 = builder.getInt32(10);
        auto* const2 = builder.getInt32(20);
        auto* deadAdd = builder.CreateAdd(const1, const2);  // 死代码
        
        // 实际返回值
        auto* result = builder.getInt32(42);
        builder.CreateRet(result);
        
        return func;
    }

    llvm::LLVMContext context;
    std::unique_ptr<llvm::Module> module;
};

// 测试：O0优化级别（无优化）
TEST_F(OptimizerTest, O0OptimizationLevel) {
    createSimpleFunction();
    
    // 记录优化前的指令数
    size_t instrCountBefore = 0;
    for (auto& func : *module) {
        for (auto& bb : func) {
            instrCountBefore += bb.size();
        }
    }
    
    // 执行O0优化（应该不做任何优化）
    Optimizer optimizer(OptLevel::O0);
    auto result = optimizer.optimize(*module);
    
    ASSERT_TRUE(result.isOk());
    
    // 验证模块仍然有效
    EXPECT_FALSE(llvm::verifyModule(*module, &llvm::errs()));
    
    // O0不应该改变指令数
    size_t instrCountAfter = 0;
    for (auto& func : *module) {
        for (auto& bb : func) {
            instrCountAfter += bb.size();
        }
    }
    
    EXPECT_EQ(instrCountBefore, instrCountAfter);
}

// 测试：O1优化级别
TEST_F(OptimizerTest, O1OptimizationLevel) {
    createSimpleFunction();
    
    // 执行O1优化
    Optimizer optimizer(OptLevel::O1);
    auto result = optimizer.optimize(*module);
    
    ASSERT_TRUE(result.isOk());
    
    // 验证模块仍然有效
    EXPECT_FALSE(llvm::verifyModule(*module, &llvm::errs()));
}

// 测试：O2优化级别
TEST_F(OptimizerTest, O2OptimizationLevel) {
    createSimpleFunction();
    
    // 执行O2优化
    Optimizer optimizer(OptLevel::O2);
    auto result = optimizer.optimize(*module);
    
    ASSERT_TRUE(result.isOk());
    
    // 验证模块仍然有效
    EXPECT_FALSE(llvm::verifyModule(*module, &llvm::errs()));
}

// 测试：O3优化级别
TEST_F(OptimizerTest, O3OptimizationLevel) {
    createSimpleFunction();
    
    // 执行O3优化
    Optimizer optimizer(OptLevel::O3);
    auto result = optimizer.optimize(*module);
    
    ASSERT_TRUE(result.isOk());
    
    // 验证模块仍然有效
    EXPECT_FALSE(llvm::verifyModule(*module, &llvm::errs()));
}

// 测试：Os优化级别（大小优化）
TEST_F(OptimizerTest, OsOptimizationLevel) {
    createSimpleFunction();
    
    // 执行Os优化
    Optimizer optimizer(OptLevel::Os);
    auto result = optimizer.optimize(*module);
    
    ASSERT_TRUE(result.isOk());
    
    // 验证模块仍然有效
    EXPECT_FALSE(llvm::verifyModule(*module, &llvm::errs()));
}

// 测试：Oz优化级别（极致大小优化）
TEST_F(OptimizerTest, OzOptimizationLevel) {
    createSimpleFunction();
    
    // 执行Oz优化
    Optimizer optimizer(OptLevel::Oz);
    auto result = optimizer.optimize(*module);
    
    ASSERT_TRUE(result.isOk());
    
    // 验证模块仍然有效
    EXPECT_FALSE(llvm::verifyModule(*module, &llvm::errs()));
}

// 测试：死代码消除
TEST_F(OptimizerTest, DeadCodeElimination) {
    createFunctionWithDeadCode();
    
    // 记录优化前的指令数
    size_t instrCountBefore = 0;
    for (auto& func : *module) {
        for (auto& bb : func) {
            instrCountBefore += bb.size();
        }
    }
    
    // 执行O2优化（应该消除死代码）
    Optimizer optimizer(OptLevel::O2);
    auto result = optimizer.optimize(*module);
    
    ASSERT_TRUE(result.isOk());
    
    // 验证模块仍然有效
    EXPECT_FALSE(llvm::verifyModule(*module, &llvm::errs()));
    
    // 优化后指令数应该减少（死代码被消除）
    size_t instrCountAfter = 0;
    for (auto& func : *module) {
        for (auto& bb : func) {
            instrCountAfter += bb.size();
        }
    }
    
    EXPECT_LT(instrCountAfter, instrCountBefore);
}

// 测试：设置优化级别
TEST_F(OptimizerTest, SetOptLevel) {
    createSimpleFunction();
    
    Optimizer optimizer(OptLevel::O0);
    
    // 改变优化级别
    optimizer.setOptLevel(OptLevel::O2);
    
    auto result = optimizer.optimize(*module);
    
    ASSERT_TRUE(result.isOk());
    EXPECT_FALSE(llvm::verifyModule(*module, &llvm::errs()));
}

// 测试：空模块优化
TEST_F(OptimizerTest, EmptyModuleOptimization) {
    // 空模块
    Optimizer optimizer(OptLevel::O2);
    auto result = optimizer.optimize(*module);
    
    ASSERT_TRUE(result.isOk());
    EXPECT_FALSE(llvm::verifyModule(*module, &llvm::errs()));
}

// 主函数
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
