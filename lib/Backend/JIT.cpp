// AZ编译器 - JIT编译器实现

#include "AZ/Backend/JIT.h"
#include "AZ/Backend/MLIRLowering.h"

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Error.h"

// 添加全局变量支持
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Type.h"

namespace az {
namespace backend {

JITCompiler::JITCompiler() 
    : context_(std::make_unique<llvm::LLVMContext>()) {
    
    // 初始化JIT所需的目标
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();

    // 创建LLJIT实例
    auto jitOrErr = llvm::orc::LLJITBuilder().create();
    if (!jitOrErr) {
        // 处理错误
        llvm::consumeError(jitOrErr.takeError());
        jit_ = nullptr;
    } else {
        jit_ = std::move(*jitOrErr);
    }
}

JITCompiler::~JITCompiler() {
}

Result<int> JITCompiler::compileAndRun(
    mlir::ModuleOp module,
    const std::vector<std::string>& args) {
    
    if (!jit_) {
        return Result<int>::Err(
            makeError("JIT编译器初始化失败"));
    }

    // 1. 降级MLIR到LLVM IR
    mlir::MLIRContext* mlirContext = module.getContext();
    MLIRLowering lowering(*mlirContext);
    
    auto llvmModuleResult = lowering.lower(module, *context_);
    if (llvmModuleResult.isErr()) {
        return Result<int>::Err(llvmModuleResult.error());
    }
    
    auto llvmModule = std::move(llvmModuleResult.value());
    
    // 2. 设置命令行参数
    auto setupResult = setupCommandLineArgs(*llvmModule, args);
    if (setupResult.isErr()) {
        return Result<int>::Err(setupResult.error());
    }
    
    // 3. 创建ThreadSafeModule
    auto tsm = llvm::orc::ThreadSafeModule(
        std::move(llvmModule),
        std::make_unique<llvm::LLVMContext>()
    );
    
    // 4. 添加模块到JIT
    auto err = jit_->addIRModule(std::move(tsm));
    if (err) {
        std::string errMsg;
        llvm::raw_string_ostream os(errMsg);
        os << err;
        llvm::consumeError(std::move(err));
        return Result<int>::Err(
            makeError("添加模块到JIT失败: " + errMsg));
    }
    
    // 5. 查找main函数
    auto mainSymbol = jit_->lookup("main");
    if (!mainSymbol) {
        llvm::consumeError(mainSymbol.takeError());
        return Result<int>::Err(
            makeError("找不到main函数"));
    }
    
    // 6. 获取函数指针
    auto mainAddr = mainSymbol->getAddress();
    // 支持带参数的main函数: int main(int argc, char* argv[])
    auto* mainFunc = reinterpret_cast<int(*)(int, char**)>(mainAddr);
    
    // 7. 执行main函数
    int exitCode;
    if (!args.empty()) {
        // 创建argv数组
        std::vector<char*> argv(args.size() + 1);
        for (size_t i = 0; i < args.size(); ++i) {
            argv[i] = const_cast<char*>(args[i].c_str());
        }
        argv[args.size()] = nullptr;
        
        exitCode = mainFunc(static_cast<int>(args.size()), argv.data());
    } else {
        exitCode = mainFunc(0, nullptr);
    }
    
    return Result<int>::Ok(exitCode);
}

Result<void*> JITCompiler::compileFunction(
    mlir::ModuleOp module,
    const std::string& functionName) {
    
    if (!jit_) {
        return Result<void*>::Err(
            makeError("JIT编译器初始化失败"));
    }

    // 1. 降级MLIR到LLVM IR
    mlir::MLIRContext* mlirContext = module.getContext();
    MLIRLowering lowering(*mlirContext);
    
    auto llvmModuleResult = lowering.lower(module, *context_);
    if (llvmModuleResult.isErr()) {
        return Result<void*>::Err(llvmModuleResult.error());
    }
    
    auto llvmModule = std::move(llvmModuleResult.value());
    
    // 2. 创建ThreadSafeModule
    auto tsm = llvm::orc::ThreadSafeModule(
        std::move(llvmModule),
        std::make_unique<llvm::LLVMContext>()
    );
    
    // 3. 添加模块到JIT
    auto err = jit_->addIRModule(std::move(tsm));
    if (err) {
        std::string errMsg;
        llvm::raw_string_ostream os(errMsg);
        os << err;
        llvm::consumeError(std::move(err));
        return Result<void*>::Err(
            makeError("添加模块到JIT失败: " + errMsg));
    }
    
    // 4. 查找函数
    auto funcSymbol = jit_->lookup(functionName);
    if (!funcSymbol) {
        llvm::consumeError(funcSymbol.takeError());
        return Result<void*>::Err(
            makeError("找不到函数: " + functionName));
    }
    
    // 5. 获取函数指针
    auto funcAddr = funcSymbol->getAddress();
    void* funcPtr = reinterpret_cast<void*>(funcAddr);
    
    return Result<void*>::Ok(funcPtr);
}

Result<void> JITCompiler::setupCommandLineArgs(
    llvm::Module& module,
    const std::vector<std::string>& args) {
    
    // 创建argc全局变量
    if (!args.empty()) {
        llvm::LLVMContext& context = module.getContext();
        
        // 创建argc
        auto* argcType = llvm::IntegerType::getInt32Ty(context);
        auto* argcValue = llvm::ConstantInt::get(argcType, args.size());
        auto* argcVar = new llvm::GlobalVariable(
            module,
            argcType,
            true,  // isConstant
            llvm::GlobalValue::ExternalLinkage,
            argcValue,
            "argc"
        );
        
        // 创建argv数组
        std::vector<llvm::Constant*> argValues;
        for (const auto& arg : args) {
            // 创建字符串常量
            auto* strConstant = llvm::ConstantDataArray::getString(context, arg, true);
            auto* strGlobal = new llvm::GlobalVariable(
                module,
                strConstant->getType(),
                true,  // isConstant
                llvm::GlobalValue::PrivateLinkage,
                strConstant
            );
            
            // 获取字符串的第一个元素的指针
            std::vector<llvm::Constant*> indices(2, llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), 0));
            auto* strPtr = llvm::ConstantExpr::getInBoundsGetElementPtr(
                strConstant->getType(),
                strGlobal,
                indices
            );
            
            argValues.push_back(strPtr);
        }
        
        // 添加null终止符
        argValues.push_back(llvm::ConstantPointerNull::get(
            llvm::Type::getInt8PtrTy(context)));
        
        // 创建argv数组
        auto* argvType = llvm::ArrayType::get(
            llvm::Type::getInt8PtrTy(context),
            argValues.size()
        );
        auto* argvArray = llvm::ConstantArray::get(argvType, argValues);
        auto* argvVar = new llvm::GlobalVariable(
            module,
            argvArray->getType(),
            true,  // isConstant
            llvm::GlobalValue::ExternalLinkage,
            argvArray,
            "argv"
        );
    }
    
    return Result<void>::Ok();
}

CompileError JITCompiler::makeError(const std::string& message) {
    return CompileError{
        ErrorKind::UnknownError,
        message,
        0, 0, ""
    };
}

} // namespace backend
} // namespace az