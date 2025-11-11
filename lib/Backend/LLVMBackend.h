// AZ编译器 - LLVM 后端头文件

#ifndef AZ_BACKEND_LLVMBACKEND_H
#define AZ_BACKEND_LLVMBACKEND_H

#include "AZ/Support/Result.h"
#include "AZ/Frontend/Error.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetMachine.h"

#include <memory>
#include <string>
#include <vector>

namespace llvm {
    class LLVMContext;
    class Module;
}

namespace az {
namespace backend {

class MLIRLowering;
class Optimizer;
class CodeGenerator;
class Linker;
class DebugInfoGenerator;
class JITCompiler;
class CompilationCache;
class TargetManager;

class LLVMBackend {
public:
    enum class OptLevel {
        O0,  // 无优化
        O1,  // 基本优化
        O2,  // 标准优化
        O3,  // 激进优化
        Os,  // 大小优化
        Oz   // 极致大小优化
    };
    
    enum class OutputType {
        LLVMIR,        // LLVM IR
        Assembly,      // 汇编代码
        Bitcode,       // LLVM Bitcode
        Object,        // 目标文件
        Executable,    // 可执行文件
        StaticLibrary, // 静态库
        SharedLibrary, // 共享库
        AOTExecutable  // AOT可执行文件
    };
    
    enum class LinkerType {
        LLD,   // LLVM linker
        MOLD,  // mold linker
        GCC,   // GNU linker
        CLANG, // Clang linker
        MSVC   // MSVC linker
    };
    
    struct Options {
        OptLevel optLevel;
        OutputType outputType = OutputType::Executable;
        LinkerType linkerType = LinkerType::LLD; // 链接器类型
        bool debugInfo = false;
        bool useCache = false;
        std::string cacheDir = ".az_cache";
        std::vector<std::string> libraryPaths;
        std::vector<std::string> libraries;
        bool staticLink = false;
        bool lto = false;
        bool pie = false;
        bool strip = false;
        std::string targetTriple;
        std::string cpu;
        std::string features;
        llvm::Reloc::Model relocModel;
        llvm::CodeModel::Model codeModel;
        bool enableAOT = false; // 启用AOT编译
        std::string aotBackend = "llvm"; // AOT后端类型
        std::vector<std::string> aotOptions; // AOT选项
        
        Options();
    };

    LLVMBackend(mlir::MLIRContext& context);
    ~LLVMBackend();
    
    /// 设置选项
    void setOptions(const Options& options);
    
    /// 设置源文件名（用于缓存和调试信息）
    void setSourceFilename(const std::string& filename);
    
    /// 编译MLIR模块
    Result<std::string> compile(
        mlir::ModuleOp module,
        const std::string& outputPath);
    
    /// 降级MLIR到LLVM IR
    Result<std::unique_ptr<llvm::Module>> lowerToLLVM(
        mlir::ModuleOp module,
        llvm::LLVMContext& llvmContext);
    
    /// 生成LLVM IR字符串
    Result<std::string> emitLLVMIR(mlir::ModuleOp module);
    
    /// 生成汇编代码
    Result<std::string> emitAssembly(mlir::ModuleOp module);
    
    /// JIT编译并运行
    Result<int> jitCompileAndRun(
        mlir::ModuleOp module,
        const std::vector<std::string>& args = {});
    
    /// AOT编译
    Result<std::string> aotCompile(
        mlir::ModuleOp module,
        const std::string& outputPath);
    
    /// 获取可用目标列表
    Result<std::vector<std::string>> getAvailableTargets();
    
    /// 获取本机目标三元组
    Result<std::string> getNativeTargetTriple();

private:
    CompileError makeError(const std::string& message);
    
    mlir::MLIRContext& context_;
    std::string filename_;
    Options options_;
    
    std::unique_ptr<MLIRLowering> lowering_;
    std::unique_ptr<Optimizer> optimizer_;
    std::unique_ptr<CodeGenerator> codegen_;
    std::unique_ptr<Linker> linker_;
    std::unique_ptr<DebugInfoGenerator> debugInfo_;
    std::unique_ptr<JITCompiler> jit_;
    std::unique_ptr<CompilationCache> cache_;
    std::unique_ptr<TargetManager> targetManager_;
};

} // namespace backend
} // namespace az

#endif // AZ_BACKEND_LLVMBACKEND_H