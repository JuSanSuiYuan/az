// AZ编译器 - 目标管理器头文件

#ifndef AZ_BACKEND_TARGETMANAGER_H
#define AZ_BACKEND_TARGETMANAGER_H

#include "AZ/Support/Result.h"
#include "AZ/Frontend/Error.h"

#include "llvm/ADT/Triple.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetMachine.h"

#include <memory>
#include <string>
#include <vector>

namespace az {
namespace backend {

class TargetManager {
public:
    TargetManager();
    
    /// 创建目标机器
    Result<std::unique_ptr<llvm::TargetMachine>> createTargetMachine(
        const std::string& targetTriple = "",
        const std::string& cpu = "",
        const std::string& features = "",
        llvm::CodeGenOpt::Level optLevel = llvm::CodeGenOpt::Default,
        llvm::Reloc::Model relocModel = llvm::Reloc::PIC_,
        llvm::CodeModel::Model codeModel = llvm::CodeModel::Small);
    
    /// 获取可用目标列表
    Result<std::vector<std::string>> getAvailableTargets();
    
    /// 获取本机目标三元组
    Result<std::string> getNativeTargetTriple();
    
    /// 获取本机CPU
    Result<std::string> getNativeCPU();
    
    /// 获取本机特性
    Result<std::string> getNativeFeatures();
    
    /// 检查目标是否受支持
    bool isTargetSupported(const std::string& targetName);
    
    /// 解析目标三元组
    Result<llvm::Triple> parseTriple(const std::string& tripleStr);
    
    /// 获取架构名称
    std::string getTargetArchName(llvm::Triple::ArchType arch);
    
    /// 获取操作系统名称
    std::string getTargetOSName(llvm::Triple::OSType os);

private:
    CompileError makeError(const std::string& message);
};

} // namespace backend
} // namespace az

#endif // AZ_BACKEND_TARGETMANAGER_H