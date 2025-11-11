// AZ编译器 - 代码生成器单元测试

#include "AZ/Backend/CodeGenerator.h"
#include "AZ/Support/Result.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>

using namespace az::backend;
using namespace az::support;

class CodeGeneratorTest : public ::testing::Test {
protected:
    void SetUp() override {
        module = std::make_unique<llvm::Module>("test", context);
        codegen = std::make_unique<CodeGenerator>();
        
        // 创建测试输出目录
        std::filesystem::create_directories("test_output");
    }

    void TearDown() override {
        // 清理测试文件
        std::filesystem::remove_all("test_output");
    }

    // 创建一个简单的测试函数
    void createSimpleFunction() {
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
        
        auto* result = builder.getInt32(42);
        builder.CreateRet(result);
    }

    llvm::LLVMContext context;
    std::unique_ptr<llvm::Module> module;
    std::unique_ptr<CodeGenerator> codegen;
};

// 测试：x86_64目标文件生成
TEST_F(CodeGeneratorTest, X86_64ObjectFileGeneration) {
    createSimpleFunction();
    
    std::string outputPath = "test_output/test_x86_64.o";
    std::string targetTriple = "x86_64-unknown-linux-gnu";
    
    auto result = codegen->generateObjectFile(*module, outputPath, targetTriple);
    
    ASSERT_TRUE(result.isOk());
    
    // 验证文件已创建
    EXPECT_TRUE(std::filesystem::exists(outputPath));
    
    // 验证文件不为空
    auto fileSize = std::filesystem::file_size(outputPath);
    EXPECT_GT(fileSize, 0);
}

// 测试：本机目标文件生成（空targetTriple）
TEST_F(CodeGeneratorTest, NativeObjectFileGeneration) {
    createSimpleFunction();
    
    std::string outputPath = "test_output/test_native.o";
    std::string targetTriple = "";  // 空表示本机
    
    auto result = codegen->generateObjectFile(*module, outputPath, targetTriple);
    
    ASSERT_TRUE(result.isOk());
    EXPECT_TRUE(std::filesystem::exists(outputPath));
}

// 测试：汇编代码生成
TEST_F(CodeGeneratorTest, AssemblyGeneration) {
    createSimpleFunction();
    
    std::string targetTriple = "x86_64-unknown-linux-gnu";
    
    auto result = codegen->generateAssembly(*module, targetTriple);
    
    ASSERT_TRUE(result.isOk());
    
    std::string assembly = result.value();
    
    // 验证汇编代码不为空
    EXPECT_FALSE(assembly.empty());
    
    // 验证包含函数名
    EXPECT_NE(assembly.find("test_func"), std::string::npos);
}

// 测试：Bitcode生成
TEST_F(CodeGeneratorTest, BitcodeGeneration) {
    createSimpleFunction();
    
    std::string outputPath = "test_output/test.bc";
    
    auto result = codegen->generateBitcode(*module, outputPath);
    
    ASSERT_TRUE(result.isOk());
    
    // 验证文件已创建
    EXPECT_TRUE(std::filesystem::exists(outputPath));
    
    // 验证文件不为空
    auto fileSize = std::filesystem::file_size(outputPath);
    EXPECT_GT(fileSize, 0);
}

// 测试：ARM64目标文件生成
TEST_F(CodeGeneratorTest, ARM64ObjectFileGeneration) {
    createSimpleFunction();
    
    std::string outputPath = "test_output/test_arm64.o";
    std::string targetTriple = "aarch64-unknown-linux-gnu";
    
    auto result = codegen->generateObjectFile(*module, outputPath, targetTriple);
    
    ASSERT_TRUE(result.isOk());
    EXPECT_TRUE(std::filesystem::exists(outputPath));
}

// 测试：不支持的目标平台
TEST_F(CodeGeneratorTest, UnsupportedTargetTriple) {
    createSimpleFunction();
    
    std::string outputPath = "test_output/test_invalid.o";
    std::string targetTriple = "invalid-unknown-unknown";
    
    auto result = codegen->generateObjectFile(*module, outputPath, targetTriple);
    
    // 应该返回错误
    EXPECT_TRUE(result.isErr());
}

// 测试：无效的输出路径
TEST_F(CodeGeneratorTest, InvalidOutputPath) {
    createSimpleFunction();
    
    // 使用无效的路径（包含不存在的目录）
    std::string outputPath = "/nonexistent/directory/test.o";
    std::string targetTriple = "x86_64-unknown-linux-gnu";
    
    auto result = codegen->generateObjectFile(*module, outputPath, targetTriple);
    
    // 应该返回错误
    EXPECT_TRUE(result.isErr());
}

// 测试：空模块代码生成
TEST_F(CodeGeneratorTest, EmptyModuleCodeGeneration) {
    // 空模块（没有函数）
    std::string outputPath = "test_output/test_empty.o";
    std::string targetTriple = "x86_64-unknown-linux-gnu";
    
    auto result = codegen->generateObjectFile(*module, outputPath, targetTriple);
    
    // 空模块也应该能生成目标文件
    ASSERT_TRUE(result.isOk());
    EXPECT_TRUE(std::filesystem::exists(outputPath));
}

// 测试：多个函数的代码生成
TEST_F(CodeGeneratorTest, MultipleFunctionsCodeGeneration) {
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
        
        auto* result = builder.getInt32(i * 10);
        builder.CreateRet(result);
    }
    
    std::string outputPath = "test_output/test_multiple.o";
    std::string targetTriple = "x86_64-unknown-linux-gnu";
    
    auto result = codegen->generateObjectFile(*module, outputPath, targetTriple);
    
    ASSERT_TRUE(result.isOk());
    EXPECT_TRUE(std::filesystem::exists(outputPath));
}

// 主函数
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
