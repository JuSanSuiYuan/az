// AZ编译器 - 代码生成器实现

#include "AZ/Backend/CodeGenerator.h"
#include "AZ/Backend/TargetManager.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/CommandFlags.h"

namespace az {
namespace backend {

CodeGenerator::CodeGenerator() {
    // 初始化所有目标
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();
}

Result<void> CodeGenerator::generateObjectFile(
    llvm::Module& module,
    const std::string& outputPath,
    const std::string& targetTriple) {
    
    // 获取目标机器
    auto tmResult = getTargetMachine(targetTriple);
    if (tmResult.isErr()) {
        return Result<void>::Err(tmResult.error());
    }
    auto targetMachine = std::move(tmResult.value());

    // 打开输出文件
    std::error_code EC;
    llvm::raw_fd_ostream dest(outputPath, EC, llvm::sys::fs::OF_None);
    if (EC) {
        return Result<void>::Err(
            makeError("无法打开输出文件: " + EC.message()));
    }

    // 生成目标文件
    return emitCode(module, *targetMachine, dest, 
                    llvm::CodeGenFileType::ObjectFile);
}

Result<std::string> CodeGenerator::generateAssembly(
    llvm::Module& module,
    const std::string& targetTriple) {
    
    // 获取目标机器
    auto tmResult = getTargetMachine(targetTriple);
    if (tmResult.isErr()) {
        return Result<std::string>::Err(tmResult.error());
    }
    auto targetMachine = std::move(tmResult.value());

    // 创建字符串输出流
    std::string asmStr;
    llvm::raw_string_ostream dest(asmStr);

    // 生成汇编代码
    auto result = emitCode(module, *targetMachine, dest,
                          llvm::CodeGenFileType::AssemblyFile);
    if (result.isErr()) {
        return Result<std::string>::Err(result.error());
    }

    dest.flush();
    return Result<std::string>::Ok(std::move(asmStr));
}

Result<void> CodeGenerator::generateBitcode(
    llvm::Module& module,
    const std::string& outputPath) {
    
    // 打开输出文件
    std::error_code EC;
    llvm::raw_fd_ostream dest(outputPath, EC, llvm::sys::fs::OF_None);
    if (EC) {
        return Result<void>::Err(
            makeError("无法打开输出文件: " + EC.message()));
    }

    // 写入Bitcode
    llvm::WriteBitcodeToFile(module, dest);
    dest.flush();

    return Result<void>::Ok();
}

Result<std::unique_ptr<llvm::TargetMachine>> CodeGenerator::getTargetMachine(
    const std::string& targetTriple) {
    
    // 使用指定的目标或本机目标
    std::string triple = targetTriple.empty() 
        ? llvm::sys::getDefaultTargetTriple() 
        : targetTriple;

    // 查找目标
    std::string error;
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget(triple, error);
    if (!target) {
        return Result<std::unique_ptr<llvm::TargetMachine>>::Err(
            makeError("无法找到目标: " + error));
    }

    // 创建目标机器
    llvm::TargetOptions opt;
    
    // 设置重定位模型
    auto RM = llvm::Reloc::Model::PIC_;
    
    // 设置代码模型
    auto CM = llvm::CodeModel::Small;
    
    // 设置CPU和特性
    std::string cpu = "generic";
    std::string features = "";
    
    auto targetMachine = std::unique_ptr<llvm::TargetMachine>(
        target->createTargetMachine(
            triple, 
            cpu, 
            features, 
            opt, 
            RM,
            CM,
            llvm::CodeGenOpt::Default
        )
    );

    if (!targetMachine) {
        return Result<std::unique_ptr<llvm::TargetMachine>>::Err(
            makeError("无法创建目标机器"));
    }

    return Result<std::unique_ptr<llvm::TargetMachine>>::Ok(
        std::move(targetMachine));
}

Result<std::unique_ptr<llvm::TargetMachine>> CodeGenerator::getTargetMachine(
    const std::string& targetTriple,
    const std::string& cpu,
    const std::string& features,
    llvm::Reloc::Model relocModel,
    llvm::CodeModel::Model codeModel,
    llvm::CodeGenOpt::Level optLevel) {
    
    // 使用指定的目标或本机目标
    std::string triple = targetTriple.empty() 
        ? llvm::sys::getDefaultTargetTriple() 
        : targetTriple;

    // 查找目标
    std::string error;
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget(triple, error);
    if (!target) {
        return Result<std::unique_ptr<llvm::TargetMachine>>::Err(
            makeError("无法找到目标: " + error));
    }

    // 创建目标机器
    llvm::TargetOptions opt;
    
    auto targetMachine = std::unique_ptr<llvm::TargetMachine>(
        target->createTargetMachine(
            triple, 
            cpu.empty() ? "generic" : cpu,
            features, 
            opt, 
            relocModel,
            codeModel,
            optLevel
        )
    );

    if (!targetMachine) {
        return Result<std::unique_ptr<llvm::TargetMachine>>::Err(
            makeError("无法创建目标机器"));
    }

    return Result<std::unique_ptr<llvm::TargetMachine>>::Ok(
        std::move(targetMachine));
}

Result<void> CodeGenerator::emitCode(
    llvm::Module& module,
    llvm::TargetMachine& targetMachine,
    llvm::raw_pwrite_stream& os,
    llvm::CodeGenFileType fileType) {
    
    // 设置数据布局
    module.setDataLayout(targetMachine.createDataLayout());
    module.setTargetTriple(targetMachine.getTargetTriple().str());

    // 创建Pass管理器
    llvm::legacy::PassManager pass;

    // 添加代码生成Pass
    if (targetMachine.addPassesToEmitFile(pass, os, nullptr, fileType)) {
        return Result<void>::Err(
            makeError("目标机器不支持该文件类型"));
    }

    // 运行Pass
    pass.run(module);
    os.flush();

    return Result<void>::Ok();
}

Result<std::vector<std::string>> CodeGenerator::getAvailableTargets() {
    std::vector<std::string> targets;
    
    // 获取所有可用目标
    for (auto it = llvm::TargetRegistry::targets().begin();
         it != llvm::TargetRegistry::targets().end();
         ++it) {
        targets.push_back(it->getName());
    }
    
    return Result<std::vector<std::string>>::Ok(std::move(targets));
}

Result<std::string> CodeGenerator::getNativeTargetTriple() {
    return Result<std::string>::Ok(llvm::sys::getDefaultTargetTriple());
}

CompileError CodeGenerator::makeError(const std::string& message) {
    return CompileError{
        ErrorKind::UnknownError,
        message,
        0, 0, ""
    };
}

} // namespace backend
} // namespace az