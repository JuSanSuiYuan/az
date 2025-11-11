// AZ编译器 - 后端集成测试

#include "AZ/Backend/LLVMBackend.h"
#include "AZ/Support/Result.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <gtest/gtest.h>
#include <filesystem>

using namespace az::backend;
using namespace az::support;

class IntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 注册必要的方言
        context.getOrLoadDialect<mlir::func::FuncDialect>();
        context.getOrLoadDialect<mlir::arith::ArithDialect>();
        
        // 创建测试输出目录
        std::filesystem::create_directories("test_integration");
    }

    void TearDown() override {
        // 清理测试文件
        std::filesystem::remove_all("test_integration");
    }

    // 创建一个简单的MLIR模块
    mlir::ModuleOp createSimpleModule() {
        mlir::OpBuilder builder(&context);
        auto loc = builder.getUnknownLoc();
        auto module = mlir::ModuleOp::create(loc);
        
        builder.setInsertionPointToEnd(module.getBody());
        
        // 创建main函数: fn main() -> i32 { return 42; }
        auto funcType = builder.getFunctionType({}, {builder.getI32Type()});
        auto func = builder.create<mlir::func::FuncOp>(loc, "main", funcType);
        
        auto* entryBlock = func.addEntryBlock();
        builder.setInsertionPointToStart(entryBlock);
        
        auto constOp = builder.create<mlir::arith::ConstantIntOp>(
            loc, 42, builder.getI32Type());
        
        builder.create<mlir::func::ReturnOp>(loc, constOp.getResult());
        
        return module;
    }

    mlir::MLIRContext context;
};

// 测试：完整的编译流程 - LLVM IR输出
TEST_F(IntegrationTest, CompileToLLVMIR) {
    auto module = createSimpleModule();
    
    LLVMBackend backend(context);
    LLVMBackend::Options options;
    options.outputType = LLVMBackend::OutputType::LLVMIR;
    options.optLevel = OptLevel::O0;
    backend.setOptions(options);
    
    std::string outputPath = "test_integration/output.ll";
    auto result = backend.compile(module, outputPath);
    
    ASSERT_TRUE(result.isOk());
    EXPECT_TRUE(std::filesystem::exists(outputPath));
    
    // 验证文件内容
    std::ifstream file(outputPath);
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    
    // 应该包含main函数
    EXPECT_NE(content.find("main"), std::string::npos);
}

// 测试：完整的编译流程 - 汇编输出
TEST_F(IntegrationTest, CompileToAssembly) {
    auto module = createSimpleModule();
    
    LLVMBackend backend(context);
    LLVMBackend::Options options;
    options.outputType = LLVMBackend::OutputType::Assembly;
    options.optLevel = OptLevel::O2;
    backend.setOptions(options);
    
    std::string outputPath = "test_integration/output.s";
    auto result = backend.compile(module, outputPath);
    
    ASSERT_TRUE(result.isOk());
    EXPECT_TRUE(std::filesystem::exists(outputPath));
}

// 测试：完整的编译流程 - Bitcode输出
TEST_F(IntegrationTest, CompileToBitcode) {
    auto module = createSimpleModule();
    
    LLVMBackend backend(context);
    LLVMBackend::Options options;
    options.outputType = LLVMBackend::OutputType::Bitcode;
    options.optLevel = OptLevel::O2;
    backend.setOptions(options);
    
    std::string outputPath = "test_integration/output.bc";
    auto result = backend.compile(module, outputPath);
    
    ASSERT_TRUE(result.isOk());
    EXPECT_TRUE(std::filesystem::exists(outputPath));
}

// 测试：完整的编译流程 - 目标文件输出
TEST_F(IntegrationTest, CompileToObjectFile) {
    auto module = createSimpleModule();
    
    LLVMBackend backend(context);
    LLVMBackend::Options options;
    options.outputType = LLVMBackend::OutputType::Object;
    options.optLevel = OptLevel::O2;
    backend.setOptions(options);
    
    std::string outputPath = "test_integration/output.o";
    auto result = backend.compile(module, outputPath);
    
    ASSERT_TRUE(result.isOk());
    EXPECT_TRUE(std::filesystem::exists(outputPath));
}

// 测试：不同优化级别
TEST_F(IntegrationTest, DifferentOptimizationLevels) {
    auto module = createSimpleModule();
    
    std::vector<OptLevel> levels = {
        OptLevel::O0,
        OptLevel::O1,
        OptLevel::O2,
        OptLevel::O3,
        OptLevel::Os,
        OptLevel::Oz
    };
    
    for (auto level : levels) {
        LLVMBackend backend(context);
        LLVMBackend::Options options;
        options.outputType = LLVMBackend::OutputType::Object;
        options.optLevel = level;
        backend.setOptions(options);
        
        std::string outputPath = "test_integration/output_opt.o";
        auto result = backend.compile(module, outputPath);
        
        ASSERT_TRUE(result.isOk());
        EXPECT_TRUE(std::filesystem::exists(outputPath));
        
        // 清理
        std::filesystem::remove(outputPath);
    }
}

// 测试：emitLLVMIR方法
TEST_F(IntegrationTest, EmitLLVMIR) {
    auto module = createSimpleModule();
    
    LLVMBackend backend(context);
    auto result = backend.emitLLVMIR(module);
    
    ASSERT_TRUE(result.isOk());
    
    std::string ir = result.value();
    EXPECT_FALSE(ir.empty());
    EXPECT_NE(ir.find("main"), std::string::npos);
}

// 测试：emitAssembly方法
TEST_F(IntegrationTest, EmitAssembly) {
    auto module = createSimpleModule();
    
    LLVMBackend backend(context);
    LLVMBackend::Options options;
    options.optLevel = OptLevel::O2;
    backend.setOptions(options);
    
    auto result = backend.emitAssembly(module);
    
    ASSERT_TRUE(result.isOk());
    
    std::string asm_code = result.value();
    EXPECT_FALSE(asm_code.empty());
    EXPECT_NE(asm_code.find("main"), std::string::npos);
}

// 测试：错误处理 - 无效的输出路径
TEST_F(IntegrationTest, InvalidOutputPath) {
    auto module = createSimpleModule();
    
    LLVMBackend backend(context);
    LLVMBackend::Options options;
    options.outputType = LLVMBackend::OutputType::Object;
    backend.setOptions(options);
    
    // 使用无效的路径
    std::string outputPath = "/nonexistent/directory/output.o";
    auto result = backend.compile(module, outputPath);
    
    // 应该返回错误
    EXPECT_TRUE(result.isErr());
}

// 主函数
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
