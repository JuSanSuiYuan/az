// AZ编译器 - 调试信息生成器单元测试

#include "AZ/Backend/DebugInfo.h"

#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"

#include <gtest/gtest.h>

using namespace az::backend;

class DebugInfoTest : public ::testing::Test {
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
        
        auto* arg = func->arg_begin();
        auto* result = builder.CreateAdd(arg, builder.getInt32(1));
        builder.CreateRet(result);
        
        return func;
    }

    llvm::LLVMContext context;
    std::unique_ptr<llvm::Module> module;
};

// 测试：创建编译单元
TEST_F(DebugInfoTest, CreateCompileUnit) {
    DebugInfoGenerator debugInfo(*module);
    
    debugInfo.createCompileUnit(
        "test.az",
        "/path/to/project",
        "AZ Compiler v0.4.0"
    );
    
    debugInfo.finalize();
    
    // 验证模块有效
    EXPECT_FALSE(llvm::verifyModule(*module, &llvm::errs()));
    
    // 验证编译单元存在
    auto* cu = module->getNamedMetadata("llvm.dbg.cu");
    ASSERT_NE(cu, nullptr);
    EXPECT_GT(cu->getNumOperands(), 0);
}

// 测试：创建函数调试信息
TEST_F(DebugInfoTest, CreateFunctionDebugInfo) {
    auto* func = createSimpleFunction();
    
    DebugInfoGenerator debugInfo(*module);
    
    debugInfo.createCompileUnit(
        "test.az",
        "/path/to/project",
        "AZ Compiler"
    );
    
    debugInfo.createFunctionDebugInfo(
        func,
        "test_func",
        10,  // 行号
        5    // 列号
    );
    
    debugInfo.finalize();
    
    // 验证模块有效
    EXPECT_FALSE(llvm::verifyModule(*module, &llvm::errs()));
    
    // 验证函数有子程序
    auto* sp = func->getSubprogram();
    ASSERT_NE(sp, nullptr);
    EXPECT_EQ(sp->getName(), "test_func");
    EXPECT_EQ(sp->getLine(), 10);
}

// 测试：设置位置信息
TEST_F(DebugInfoTest, SetLocation) {
    auto* func = createSimpleFunction();
    
    DebugInfoGenerator debugInfo(*module);
    
    debugInfo.createCompileUnit(
        "test.az",
        "/path/to/project",
        "AZ Compiler"
    );
    
    debugInfo.createFunctionDebugInfo(
        func,
        "test_func",
        10,
        5
    );
    
    // 为第一条指令设置位置
    auto& entry = func->getEntryBlock();
    auto* firstInst = &entry.front();
    
    debugInfo.setLocation(firstInst, 11, 10);
    
    debugInfo.finalize();
    
    // 验证模块有效
    EXPECT_FALSE(llvm::verifyModule(*module, &llvm::errs()));
    
    // 验证指令有调试位置
    auto loc = firstInst->getDebugLoc();
    ASSERT_TRUE(loc);
    EXPECT_EQ(loc.getLine(), 11);
    EXPECT_EQ(loc.getCol(), 10);
}

// 测试：创建变量调试信息
TEST_F(DebugInfoTest, CreateVariableDebugInfo) {
    auto* funcType = llvm::FunctionType::get(
        llvm::Type::getInt32Ty(context),
        {},
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
    
    // 创建一个局部变量
    auto* alloca = builder.CreateAlloca(builder.getInt32Ty(), nullptr, "x");
    builder.CreateStore(builder.getInt32(42), alloca);
    auto* load = builder.CreateLoad(builder.getInt32Ty(), alloca);
    builder.CreateRet(load);
    
    DebugInfoGenerator debugInfo(*module);
    
    debugInfo.createCompileUnit(
        "test.az",
        "/path/to/project",
        "AZ Compiler"
    );
    
    debugInfo.createFunctionDebugInfo(
        func,
        "test_func",
        10,
        5
    );
    
    // 为变量创建调试信息
    debugInfo.createVariableDebugInfo(
        alloca,
        "x",
        builder.getInt32Ty(),
        11,
        10
    );
    
    debugInfo.finalize();
    
    // 验证模块有效
    EXPECT_FALSE(llvm::verifyModule(*module, &llvm::errs()));
}

// 测试：多个函数的调试信息
TEST_F(DebugInfoTest, MultipleFunctionsDebugInfo) {
    DebugInfoGenerator debugInfo(*module);
    
    debugInfo.createCompileUnit(
        "test.az",
        "/path/to/project",
        "AZ Compiler"
    );
    
    // 创建多个函数
    for (int i = 0; i < 3; ++i) {
        auto* funcType = llvm::FunctionType::get(
            llvm::Type::getInt32Ty(context),
            {},
            false
        );
        
        std::string funcName = "func" + std::to_string(i);
        auto* func = llvm::Function::Create(
            funcType,
            llvm::Function::ExternalLinkage,
            funcName,
            module.get()
        );
        
        auto* entry = llvm::BasicBlock::Create(context, "entry", func);
        llvm::IRBuilder<> builder(entry);
        builder.CreateRet(builder.getInt32(i * 10));
        
        debugInfo.createFunctionDebugInfo(
            func,
            funcName,
            10 + i,
            5
        );
    }
    
    debugInfo.finalize();
    
    // 验证模块有效
    EXPECT_FALSE(llvm::verifyModule(*module, &llvm::errs()));
    
    // 验证所有函数都有子程序
    for (auto& func : *module) {
        if (!func.isDeclaration()) {
            EXPECT_NE(func.getSubprogram(), nullptr);
        }
    }
}

// 测试：没有编译单元的情况
TEST_F(DebugInfoTest, NoCompileUnit) {
    auto* func = createSimpleFunction();
    
    DebugInfoGenerator debugInfo(*module);
    
    // 不创建编译单元，直接创建函数调试信息
    // 应该不会崩溃，但也不会生成调试信息
    debugInfo.createFunctionDebugInfo(
        func,
        "test_func",
        10,
        5
    );
    
    debugInfo.finalize();
    
    // 验证模块有效
    EXPECT_FALSE(llvm::verifyModule(*module, &llvm::errs()));
    
    // 函数不应该有子程序
    EXPECT_EQ(func->getSubprogram(), nullptr);
}

// 主函数
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
