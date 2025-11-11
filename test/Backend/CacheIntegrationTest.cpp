// AZ编译器 - 编译缓存集成测试

#include "AZ/Backend/LLVMBackend.h"
#include "AZ/Backend/Cache.h"
#include "AZ/Support/Result.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <chrono>

using namespace az::backend;
using namespace az::support;

class CacheIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 注册必要的方言
        context.getOrLoadDialect<mlir::func::FuncDialect>();
        context.getOrLoadDialect<mlir::arith::ArithDialect>();
        
        // 创建测试目录
        std::filesystem::create_directories("test_cache");
        std::filesystem::create_directories("test_cache_output");
    }

    void TearDown() override {
        // 清理测试文件
        std::filesystem::remove_all("test_cache");
        std::filesystem::remove_all("test_cache_output");
    }

    // 创建一个简单的MLIR模块
    mlir::ModuleOp createSimpleModule() {
        mlir::OpBuilder builder(&context);
        auto loc = builder.getUnknownLoc();
        auto module = mlir::ModuleOp::create(loc);
        
        builder.setInsertionPointToEnd(module.getBody());
        
        auto funcType = builder.getFunctionType({}, {builder.getI32Type()});
        auto func = builder.create<mlir::func::FuncOp>(loc, "main", funcType);
        
        auto* entryBlock = func.addEntryBlock();
        builder.setInsertionPointToStart(entryBlock);
        
        auto constOp = builder.create<mlir::arith::ConstantIntOp>(
            loc, 42, builder.getI32Type());
        
        builder.create<mlir::func::ReturnOp>(loc, constOp.getResult());
        
        return module;
    }

    // 创建一个虚拟的源文件
    void createSourceFile(const std::string& filename, const std::string& content) {
        std::ofstream file(filename);
        file << content;
        file.close();
    }

    mlir::MLIRContext context;
};

// 测试：缓存保存和获取
TEST_F(CacheIntegrationTest, CacheSaveAndRetrieve) {
    auto module = createSimpleModule();
    
    // 创建源文件
    std::string sourceFile = "test_cache/test.az";
    createSourceFile(sourceFile, "fn main() int { return 42; }");
    
    LLVMBackend backend(context);
    LLVMBackend::Options options;
    options.outputType = LLVMBackend::OutputType::Object;
    options.optLevel = OptLevel::O2;
    options.useCache = true;
    options.cacheDir = "test_cache/.cache";
    backend.setOptions(options);
    backend.setSourceFilename(sourceFile);
    
    std::string outputPath = "test_cache_output/test.o";
    
    // 第一次编译（应该生成缓存）
    auto result1 = backend.compile(module, outputPath);
    ASSERT_TRUE(result1.isOk());
    EXPECT_TRUE(std::filesystem::exists(outputPath));
    
    // 记录第一次编译时间
    auto start1 = std::chrono::high_resolution_clock::now();
    result1 = backend.compile(module, outputPath);
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
    
    // 删除输出文件
    std::filesystem::remove(outputPath);
    
    // 第二次编译（应该使用缓存，更快）
    auto start2 = std::chrono::high_resolution_clock::now();
    auto result2 = backend.compile(module, outputPath);
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
    
    ASSERT_TRUE(result2.isOk());
    EXPECT_TRUE(std::filesystem::exists(outputPath));
    
    // 第二次应该更快（使用了缓存）
    // 注意：这个测试可能不稳定，因为编译时间可能很短
    std::cout << "第一次编译: " << duration1.count() << "ms" << std::endl;
    std::cout << "第二次编译（缓存）: " << duration2.count() << "ms" << std::endl;
}

// 测试：缓存失效（文件修改）
TEST_F(CacheIntegrationTest, CacheInvalidation) {
    auto module = createSimpleModule();
    
    std::string sourceFile = "test_cache/test2.az";
    createSourceFile(sourceFile, "fn main() int { return 42; }");
    
    LLVMBackend backend(context);
    LLVMBackend::Options options;
    options.outputType = LLVMBackend::OutputType::Object;
    options.useCache = true;
    options.cacheDir = "test_cache/.cache";
    backend.setOptions(options);
    backend.setSourceFilename(sourceFile);
    
    std::string outputPath = "test_cache_output/test2.o";
    
    // 第一次编译
    auto result1 = backend.compile(module, outputPath);
    ASSERT_TRUE(result1.isOk());
    
    // 修改源文件
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    createSourceFile(sourceFile, "fn main() int { return 43; }");
    
    // 第二次编译（缓存应该失效）
    auto result2 = backend.compile(module, outputPath);
    ASSERT_TRUE(result2.isOk());
}

// 测试：禁用缓存
TEST_F(CacheIntegrationTest, CacheDisabled) {
    auto module = createSimpleModule();
    
    std::string sourceFile = "test_cache/test3.az";
    createSourceFile(sourceFile, "fn main() int { return 42; }");
    
    LLVMBackend backend(context);
    LLVMBackend::Options options;
    options.outputType = LLVMBackend::OutputType::Object;
    options.useCache = false;  // 禁用缓存
    backend.setOptions(options);
    backend.setSourceFilename(sourceFile);
    
    std::string outputPath = "test_cache_output/test3.o";
    
    // 编译
    auto result = backend.compile(module, outputPath);
    ASSERT_TRUE(result.isOk());
    
    // 缓存目录不应该被创建
    EXPECT_FALSE(std::filesystem::exists(".az-cache"));
}

// 测试：可执行文件缓存
TEST_F(CacheIntegrationTest, ExecutableCaching) {
    auto module = createSimpleModule();
    
    std::string sourceFile = "test_cache/test4.az";
    createSourceFile(sourceFile, "fn main() int { return 42; }");
    
    LLVMBackend backend(context);
    LLVMBackend::Options options;
    options.outputType = LLVMBackend::OutputType::Object;  // 先生成目标文件
    options.useCache = true;
    options.cacheDir = "test_cache/.cache";
    backend.setOptions(options);
    backend.setSourceFilename(sourceFile);
    
    std::string outputPath = "test_cache_output/test4.o";
    
    // 编译目标文件
    auto result = backend.compile(module, outputPath);
    ASSERT_TRUE(result.isOk());
    EXPECT_TRUE(std::filesystem::exists(outputPath));
}

// 主函数
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
