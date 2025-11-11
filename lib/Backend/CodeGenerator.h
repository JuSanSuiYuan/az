// AZ编译器 - 代码生成器头文件

#ifndef AZ_BACKEND_CODEGENERATOR_H
#define AZ_BACKEND_CODEGENERATOR_H

#include "AZ/Support/Result.h"
#include "AZ/Frontend/Error.h"

#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetMachine.h"

#include <memory>
#include <string>
#include <vector>

namespace llvm {
    class Module;
    class raw_pwrite_stream;
    class TargetMachine;
}

namespace az {
namespace backend {

class CodeGenerator {
public:
    CodeGenerator();
    
    /// 生成目标文件
    Result<void> generateObjectFile(
        llvm::Module& module,
        const std::string& outputPath,
        const std::string& targetTriple = "");
    
    /// 生成汇编代码
    Result<std::string> generateAssembly(
        llvm::Module& module,
        const std::string& targetTriple = "");
    
    /// 生成Bitcode
    Result<void> generateBitcode(
        llvm::Module& module,
        const std::string& outputPath);
    
    /// 获取目标机器
    Result<std::unique_ptr<llvm::TargetMachine>> getTargetMachine(
        const std::string& targetTriple = "");
    
    /// 获取目标机器（带详细选项）
    Result<std::unique_ptr<llvm::TargetMachine>> getTargetMachine(
        const std::string& targetTriple,
        const std::string& cpu,
        const std::string& features,
        llvm::Reloc::Model relocModel = llvm::Reloc::PIC_,
        llvm::CodeModel::Model codeModel = llvm::CodeModel::Small,
        llvm::CodeGenOpt::Level optLevel = llvm::CodeGenOpt::Default);
    
    /// 获取可用目标列表
    Result<std::vector<std::string>> getAvailableTargets();
    
    /// 获取本机目标三元组
    Result<std::string> getNativeTargetTriple();

private:
    Result<void> emitCode(
        llvm::Module& module,
        llvm::TargetMachine& targetMachine,
        llvm::raw_pwrite_stream& os,
        llvm::CodeGenFileType fileType);
    
    CompileError makeError(const std::string& message);
};

} // namespace backend
} // namespace az

#endif // AZ_BACKEND_CODEGENERATOR_H