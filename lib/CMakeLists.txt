# AZ编译器库

# 前端库
add_library(AZFrontend
    Frontend/Lexer.cpp
    Frontend/Parser.cpp
    Frontend/Token.cpp
    Frontend/Sema.cpp
    Frontend/AST.cpp
)

target_include_directories(AZFrontend PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(AZFrontend
    LLVMSupport
)

# 支持库
add_library(AZSupport
    Support/Result.cpp
)

target_include_directories(AZSupport PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# MLIR库
add_library(AZMLIR
    MLIR/MLIRGen.cpp
)

target_include_directories(AZMLIR PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(AZMLIR
    PUBLIC
    AZFrontend
    AZSupport
    MLIRAnalysis
    MLIRIR
    MLIRParser
    MLIRFuncDialect
    MLIRArithDialect
    MLIRSupport
)

# 后端库
add_library(AZBackend
    Backend/LLVMBackend.cpp
    Backend/MLIRLowering.cpp
    Backend/Optimizer.cpp
    Backend/CodeGenerator.cpp
    Backend/Linker.cpp
    Backend/DebugInfo.cpp
    Backend/JIT.cpp
    Backend/Cache.cpp
    Backend/TargetManager.cpp
)
target_include_directories(AZBackend PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
    ${OPENSSL_INCLUDE_DIRS}
)
target_link_libraries(AZBackend
    PUBLIC
    AZMLIR
    AZSupport
    ${OPENSSL_LIBRARIES}
    # MLIR组件
    MLIRLLVMToLLVMIRTranslation
    MLIRPass
    MLIRTransforms
    MLIRTargetLLVMIRExport
    MLIRConversionPasses
    MLIRFuncDialect
    MLIRArithDialect
    MLIRSCFDialect
    MLIRControlFlowDialect
    MLIRMemRefDialect
    # LLVM组件
    LLVMCore
    LLVMSupport
    LLVMIRReader
    LLVMTarget
    LLVMCodeGen
    LLVMPasses
    LLVMAnalysis
    LLVMTransformUtils
    LLVMScalarOpts
    LLVMInstCombine
    LLVMAggressiveInstCombine
    LLVMipo
    LLVMVectorize
    LLVMBitWriter
    LLVMBitReader
    LLVMOrcJIT
    LLVMExecutionEngine
    LLVMMC
    LLVMMCParser
    LLVMObject
    # lld组件
    lldCOFF
    lldCommon
    lldELF
    lldMachO
    lldMinGW
    lldWasm
    # 目标架构
    LLVMX86CodeGen
    LLVMX86AsmParser
    LLVMX86Desc
    LLVMX86Info
    LLVMAArch64CodeGen
    LLVMAArch64AsmParser
    LLVMAArch64Desc
    LLVMAArch64Info
)

# 链接器库（待实现）
add_library(AZLinker INTERFACE)
target_include_directories(AZLinker INTERFACE
    ${PROJECT_SOURCE_DIR}/include
)

# 调试库（待实现）
add_library(AZDebug INTERFACE)
target_include_directories(AZDebug INTERFACE
    ${PROJECT_SOURCE_DIR}/include
)