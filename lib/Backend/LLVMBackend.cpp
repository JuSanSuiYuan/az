// AZ编译器 - LLVM 后端实现

#include "AZ/Backend/LLVMBackend.h"
#include "AZ/Backend/MLIRLowering.h"
#include "AZ/Backend/Optimizer.h"
#include "AZ/Backend/CodeGenerator.h"
#include "AZ/Backend/Linker.h"
#include "AZ/Backend/DebugInfo.h"
#include "AZ/Backend/JIT.h"
#include "AZ/Backend/Cache.h"
#include "AZ/Backend/TargetManager.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"

#include <filesystem>
#include <fstream>

namespace az {
namespace backend {

// Options构造函数
LLVMBackend::Options::Options()
    : optLevel(OptLevel::O2),
      linkerType(LinkerType::LLD),
      targetTriple(""),
      cpu("generic"),
      features(""),
      relocModel(llvm::Reloc::PIC_),
      codeModel(llvm::CodeModel::Small),
      enableAOT(false),
      aotBackend("llvm") {
}

LLVMBackend::LLVMBackend(mlir::MLIRContext& context)
    : context_(context),
      filename_("") {
    
    // 初始化组件
    lowering_ = std::make_unique<MLIRLowering>(context_);
    optimizer_ = std::make_unique<Optimizer>(OptLevel::O2);
    codegen_ = std::make_unique<CodeGenerator>();
    linker_ = std::make_unique<Linker>();
    jit_ = std::make_unique<JITCompiler>();
    targetManager_ = std::make_unique<TargetManager>();
}

LLVMBackend::~LLVMBackend() {
}

void LLVMBackend::setOptions(const Options& options) {
    options_ = options;
    
    // 更新优化器级别
    optimizer_->setOptLevel(options_.optLevel);
    
    // 创建缓存（如果启用）
    if (options_.useCache) {
        cache_ = std::make_unique<CompilationCache>(options_.cacheDir);
    }
}

void LLVMBackend::setSourceFilename(const std::string& filename) {
    filename_ = filename;
}

Result<std::string> LLVMBackend::compile(
    mlir::ModuleOp module,
    const std::string& outputPath) {
    
    // 如果启用AOT编译且输出类型是可执行文件，则使用AOT编译
    if (options_.enableAOT && options_.outputType == OutputType::Executable) {
        return aotCompile(module, outputPath);
    }
    
    // 0. 检查缓存（如果启用且输出类型是目标文件或可执行文件）
    if (cache_ && !filename_.empty() && 
        (options_.outputType == OutputType::Object || 
         options_.outputType == OutputType::Executable ||
         options_.outputType == OutputType::AOTExecutable)) {
        
        auto hasCacheResult = cache_->hasCache(filename_);
        if (hasCacheResult.isOk() && hasCacheResult.value()) {
            // 缓存存在，尝试获取
            auto cachedResult = cache_->getCachedObjectFile(filename_);
            if (cachedResult.isOk()) {
                std::string cachedFile = cachedResult.value();
                
                // 如果是目标文件，直接返回缓存
                if (options_.outputType == OutputType::Object) {
                    // 复制缓存文件到输出路径
                    std::filesystem::copy_file(
                        cachedFile,
                        outputPath,
                        std::filesystem::copy_options::overwrite_existing
                    );
                    return Result<std::string>::Ok(outputPath);
                }
                
                // 如果是可执行文件，使用缓存的目标文件进行链接
                if (options_.outputType == OutputType::Executable || 
                    options_.outputType == OutputType::AOTExecutable) {
                    Linker::LinkOptions linkOpts;
                    linkOpts.objectFiles.push_back(cachedFile);
                    linkOpts.libraryPaths = options_.libraryPaths;
                    linkOpts.libraries = options_.libraries;
                    linkOpts.outputPath = outputPath;
                    linkOpts.staticLink = options_.staticLink;
                    linkOpts.lto = options_.lto;
                    linkOpts.pie = options_.pie;
                    linkOpts.strip = options_.strip;
                    linkOpts.targetTriple = options_.targetTriple;
                    linkOpts.linkerType = options_.linkerType;
                    
                    auto linkResult = linker_->link(linkOpts);
                    if (linkResult.isErr()) {
                        // 链接失败，继续正常编译
                        // 记录警告信息
                        llvm::errs() << "警告: 使用缓存链接失败，将重新编译: " 
                                     << linkResult.error().message << "\n";
                    } else {
                        return Result<std::string>::Ok(outputPath);
                    }
                }
            }
        }
    }
    
    llvm::LLVMContext llvmContext;
    
    // 1. 降级MLIR到LLVM IR
    auto llvmModuleResult = lowering_->lower(module, llvmContext);
    if (llvmModuleResult.isErr()) {
        return Result<std::string>::Err(llvmModuleResult.error());
    }
    auto llvmModule = std::move(llvmModuleResult.value());
    
    // 2. 设置目标信息
    if (!options_.targetTriple.empty()) {
        llvmModule->setTargetTriple(options_.targetTriple);
    } else {
        // 设置本机目标三元组
        auto tripleResult = targetManager_->getNativeTargetTriple();
        if (tripleResult.isOk()) {
            llvmModule->setTargetTriple(tripleResult.value());
        }
    }
    
    // 3. 生成调试信息（如果需要）
    if (options_.debugInfo) {
        debugInfo_ = std::make_unique<DebugInfoGenerator>(*llvmModule);
        
        // 创建编译单元
        debugInfo_->createCompileUnit(
            filename_.empty() ? "module.az" : filename_,
            ".",
            "AZ Compiler v0.4.0"
        );
        
        // 为所有函数添加调试信息
        for (auto& func : *llvmModule) {
            if (!func.isDeclaration()) {
                debugInfo_->createFunctionDebugInfo(
                    &func,
                    func.getName().str(),
                    1,  // 行号（实际应该从AST获取）
                    0   // 列号
                );
            }
        }
        
        // 完成调试信息生成
        debugInfo_->finalize();
    }
    
    // 4. 优化（如果需要）
    if (options_.optLevel != OptLevel::O0) {
        auto optResult = optimizer_->optimize(*llvmModule);
        if (optResult.isErr()) {
            return Result<std::string>::Err(optResult.error());
        }
    }
    
    // 5. 根据输出类型生成代码
    switch (options_.outputType) {
    case OutputType::LLVMIR: {
        // 输出LLVM IR
        std::string buffer;
        llvm::raw_string_ostream os(buffer);
        llvmModule->print(os, nullptr);
        os.flush();
        
        // 写入文件
        std::ofstream outFile(outputPath);
        if (!outFile) {
            return Result<std::string>::Err(
                makeError("无法创建输出文件: " + outputPath));
        }
        outFile << buffer;
        outFile.close();
        
        return Result<std::string>::Ok(outputPath);
    }
    
    case OutputType::Assembly: {
        // 生成汇编代码
        auto asmResult = codegen_->generateAssembly(*llvmModule, options_.targetTriple);
        if (asmResult.isErr()) {
            return Result<std::string>::Err(asmResult.error());
        }
        
        // 写入文件
        std::ofstream outFile(outputPath);
        if (!outFile) {
            return Result<std::string>::Err(
                makeError("无法创建输出文件: " + outputPath));
        }
        outFile << asmResult.value();
        outFile.close();
        
        return Result<std::string>::Ok(outputPath);
    }
    
    case OutputType::Bitcode: {
        // 生成Bitcode
        auto bcResult = codegen_->generateBitcode(*llvmModule, outputPath);
        if (bcResult.isErr()) {
            return Result<std::string>::Err(bcResult.error());
        }
        
        return Result<std::string>::Ok(outputPath);
    }
    
    case OutputType::Object: {
        // 生成目标文件
        auto objResult = codegen_->generateObjectFile(
            *llvmModule, outputPath, options_.targetTriple);
        if (objResult.isErr()) {
            return Result<std::string>::Err(objResult.error());
        }
        
        // 保存到缓存（如果启用）
        if (cache_ && !filename_.empty()) {
            cache_->saveToCache(filename_, outputPath);
        }
        
        return Result<std::string>::Ok(outputPath);
    }
    
    case OutputType::Executable:
    case OutputType::AOTExecutable: {
        // 生成可执行文件
        // 首先生成目标文件
        std::string objPath = outputPath + ".o";
        auto objResult = codegen_->generateObjectFile(
            *llvmModule, objPath, options_.targetTriple);
        if (objResult.isErr()) {
            return Result<std::string>::Err(objResult.error());
        }
        
        // 保存目标文件到缓存（如果启用）
        if (cache_ && !filename_.empty()) {
            cache_->saveToCache(filename_, objPath);
        }
        
        // 然后链接
        Linker::LinkOptions linkOpts;
        linkOpts.objectFiles.push_back(objPath);
        linkOpts.libraryPaths = options_.libraryPaths;
        linkOpts.libraries = options_.libraries;
        linkOpts.outputPath = outputPath;
        linkOpts.staticLink = options_.staticLink;
        linkOpts.lto = options_.lto;
        linkOpts.pie = options_.pie;
        linkOpts.strip = options_.strip;
        linkOpts.targetTriple = options_.targetTriple;
        linkOpts.linkerType = options_.linkerType;
        
        auto linkResult = linker_->link(linkOpts);
        if (linkResult.isErr()) {
            return Result<std::string>::Err(linkResult.error());
        }
        
        // 删除临时目标文件
        std::filesystem::remove(objPath);
        
        return Result<std::string>::Ok(outputPath);
    }
    
    case OutputType::StaticLibrary: {
        // 生成静态库
        // 首先生成目标文件
        std::string objPath = outputPath + ".o";
        auto objResult = codegen_->generateObjectFile(
            *llvmModule, objPath, options_.targetTriple);
        if (objResult.isErr()) {
            return Result<std::string>::Err(objResult.error());
        }
        
        // 创建静态库
        std::vector<std::string> objFiles = {objPath};
        auto libResult = linker_->createStaticLibrary(objFiles, outputPath);
        if (libResult.isErr()) {
            return Result<std::string>::Err(libResult.error());
        }
        
        // 删除临时目标文件
        std::filesystem::remove(objPath);
        
        return Result<std::string>::Ok(outputPath);
    }
    
    case OutputType::SharedLibrary: {
        // 生成共享库
        // 首先生成目标文件
        std::string objPath = outputPath + ".o";
        auto objResult = codegen_->generateObjectFile(
            *llvmModule, objPath, options_.targetTriple);
        if (objResult.isErr()) {
            return Result<std::string>::Err(objResult.error());
        }
        
        // 创建共享库
        std::vector<std::string> objFiles = {objPath};
        auto libResult = linker_->createSharedLibrary(
            objFiles, outputPath, options_.libraryPaths, options_.libraries);
        if (libResult.isErr()) {
            return Result<std::string>::Err(libResult.error());
        }
        
        // 删除临时目标文件
        std::filesystem::remove(objPath);
        
        return Result<std::string>::Ok(outputPath);
    }
    }
    
    return Result<std::string>::Err(
        makeError("未知的输出类型"));
}

Result<std::unique_ptr<llvm::Module>> LLVMBackend::lowerToLLVM(
    mlir::ModuleOp module,
    llvm::LLVMContext& llvmContext) {
    
    return lowering_->lower(module, llvmContext);
}

Result<std::string> LLVMBackend::emitLLVMIR(mlir::ModuleOp module) {
    llvm::LLVMContext llvmContext;
    auto llvmModuleResult = lowerToLLVM(module, llvmContext);
    if (llvmModuleResult.isErr()) {
        return Result<std::string>::Err(llvmModuleResult.error());
    }

    auto llvmModule = std::move(llvmModuleResult.value());
    std::string buffer;
    llvm::raw_string_ostream os(buffer);
    llvmModule->print(os, nullptr);
    os.flush();

    return Result<std::string>::Ok(std::move(buffer));
}

Result<std::string> LLVMBackend::emitAssembly(mlir::ModuleOp module) {
    // 降级到LLVM IR
    llvm::LLVMContext llvmContext;
    auto llvmModuleResult = lowerToLLVM(module, llvmContext);
    if (llvmModuleResult.isErr()) {
        return Result<std::string>::Err(llvmModuleResult.error());
    }

    auto llvmModule = std::move(llvmModuleResult.value());
    
    // 优化（如果需要）
    if (options_.optLevel != OptLevel::O0) {
        auto optResult = optimizer_->optimize(*llvmModule);
        if (optResult.isErr()) {
            return Result<std::string>::Err(optResult.error());
        }
    }
    
    // 生成汇编
    return codegen_->generateAssembly(*llvmModule, options_.targetTriple);
}

Result<int> LLVMBackend::jitCompileAndRun(
    mlir::ModuleOp module,
    const std::vector<std::string>& args) {
    
    if (!jit_) {
        jit_ = std::make_unique<JITCompiler>();
    }
    
    return jit_->compileAndRun(module, args);
}

Result<std::string> LLVMBackend::aotCompile(
    mlir::ModuleOp module,
    const std::string& outputPath) {
    
    // AOT编译实现
    llvm::LLVMContext llvmContext;
    
    // 1. 降级MLIR到LLVM IR
    auto llvmModuleResult = lowering_->lower(module, llvmContext);
    if (llvmModuleResult.isErr()) {
        return Result<std::string>::Err(llvmModuleResult.error());
    }
    auto llvmModule = std::move(llvmModuleResult.value());
    
    // 2. 设置目标信息
    if (!options_.targetTriple.empty()) {
        llvmModule->setTargetTriple(options_.targetTriple);
    } else {
        // 设置本机目标三元组
        auto tripleResult = targetManager_->getNativeTargetTriple();
        if (tripleResult.isOk()) {
            llvmModule->setTargetTriple(tripleResult.value());
        }
    }
    
    // 3. 生成调试信息（如果需要）
    if (options_.debugInfo) {
        debugInfo_ = std::make_unique<DebugInfoGenerator>(*llvmModule);
        
        // 创建编译单元
        debugInfo_->createCompileUnit(
            filename_.empty() ? "module.az" : filename_,
            ".",
            "AZ Compiler v0.4.0"
        );
        
        // 为所有函数添加调试信息
        for (auto& func : *llvmModule) {
            if (!func.isDeclaration()) {
                debugInfo_->createFunctionDebugInfo(
                    &func,
                    func.getName().str(),
                    1,  // 行号（实际应该从AST获取）
                    0   // 列号
                );
            }
        }
        
        // 完成调试信息生成
        debugInfo_->finalize();
    }
    
    // 4. 优化（AOT优化）
    if (options_.optLevel != OptLevel::O0) {
        auto optResult = optimizer_->optimize(*llvmModule);
        if (optResult.isErr()) {
            return Result<std::string>::Err(optResult.error());
        }
    }
    
    // 5. 根据AOT后端生成可执行文件
    if (options_.aotBackend == "llvm") {
        // 使用LLVM生成可执行文件
        // 首先生成目标文件
        std::string objPath = outputPath + ".o";
        auto objResult = codegen_->generateObjectFile(
            *llvmModule, objPath, options_.targetTriple);
        if (objResult.isErr()) {
            return Result<std::string>::Err(objResult.error());
        }
        
        // 然后链接
        Linker::LinkOptions linkOpts;
        linkOpts.objectFiles.push_back(objPath);
        linkOpts.libraryPaths = options_.libraryPaths;
        linkOpts.libraries = options_.libraries;
        linkOpts.outputPath = outputPath;
        linkOpts.staticLink = options_.staticLink;
        linkOpts.lto = options_.lto;
        linkOpts.pie = options_.pie;
        linkOpts.strip = options_.strip;
        linkOpts.targetTriple = options_.targetTriple;
        linkOpts.linkerType = options_.linkerType;
        
        auto linkResult = linker_->link(linkOpts);
        if (linkResult.isErr()) {
            return Result<std::string>::Err(linkResult.error());
        }
        
        // 删除临时目标文件
        std::filesystem::remove(objPath);
        
        return Result<std::string>::Ok(outputPath);
    } else {
        // 其他AOT后端实现
        return Result<std::string>::Err(
            makeError("不支持的AOT后端: " + options_.aotBackend));
    }
}

Result<std::vector<std::string>> LLVMBackend::getAvailableTargets() {
    if (!targetManager_) {
        targetManager_ = std::make_unique<TargetManager>();
    }
    
    return targetManager_->getAvailableTargets();
}

Result<std::string> LLVMBackend::getNativeTargetTriple() {
    if (!targetManager_) {
        targetManager_ = std::make_unique<TargetManager>();
    }
    
    return targetManager_->getNativeTargetTriple();
}

CompileError LLVMBackend::makeError(const std::string& message) {
    return CompileError{
        ErrorKind::UnknownError,
        message,
        0, 0,
        filename_
    };
}

} // namespace backend
} // namespace az