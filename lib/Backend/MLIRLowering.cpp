// AZ编译器 - MLIR降级模块实现

#include "AZ/Backend/MLIRLowering.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

namespace az {
namespace backend {

MLIRLowering::MLIRLowering(mlir::MLIRContext& context)
    : context_(context) {
    registerDialects();
}

Result<std::unique_ptr<llvm::Module>> MLIRLowering::lower(
    mlir::ModuleOp module,
    llvm::LLVMContext& llvmContext) {
    
    // 克隆模块以避免修改原始模块
    mlir::OwningOpRef<mlir::ModuleOp> moduleClone = module.clone();
    if (!moduleClone) {
        return Result<std::unique_ptr<llvm::Module>>::Err(
            makeError("无法克隆MLIR模块"));
    }

    // 创建Pass管理器
    mlir::PassManager pm(&context_);
    buildLoweringPipeline(pm);

    // 运行降级Pass
    if (mlir::failed(pm.run(moduleClone.get()))) {
        return Result<std::unique_ptr<llvm::Module>>::Err(
            makeError("MLIR降级Pass失败"));
    }

    // 转换为LLVM IR
    return translateToLLVM(moduleClone.get(), llvmContext);
}

void MLIRLowering::buildLoweringPipeline(mlir::PassManager& pm) {
    // 1. 高级方言降级
    pm.addPass(mlir::createConvertSCFToControlFlowPass());
    
    // 2. 标准方言到LLVM方言
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createConvertArithToLLVMPass());
    pm.addPass(mlir::createConvertMathToLLVMPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    
    // 3. 内存操作降级
    pm.addPass(mlir::createConvertMemRefToLLVMPass());
    
    // 4. 清理未实现的转换
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    
    // 5. 符号可见性处理
    pm.addPass(mlir::createSymbolDCEPass());
}

void MLIRLowering::registerDialects() {
    mlir::registerLLVMDialectTranslation(context_);
}

Result<std::unique_ptr<llvm::Module>> MLIRLowering::translateToLLVM(
    mlir::ModuleOp module,
    llvm::LLVMContext& llvmContext) {
    
    auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
    if (!llvmModule) {
        return Result<std::unique_ptr<llvm::Module>>::Err(
            makeError("无法生成LLVM Module"));
    }
    
    return Result<std::unique_ptr<llvm::Module>>::Ok(std::move(llvmModule));
}

CompileError MLIRLowering::makeError(const std::string& message) {
    return CompileError{
        ErrorKind::UnknownError,
        message,
        0, 0, ""
    };
}

} // namespace backend
} // namespace az
