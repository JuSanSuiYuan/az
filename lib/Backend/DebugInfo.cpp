// AZ编译器 - 调试信息生成器实现

#include "AZ/Backend/DebugInfo.h"

#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"

namespace az {
namespace backend {

DebugInfoGenerator::DebugInfoGenerator(llvm::Module& module)
    : module_(module),
      builder_(std::make_unique<llvm::DIBuilder>(module)),
      compileUnit_(nullptr),
      file_(nullptr) {
}

DebugInfoGenerator::~DebugInfoGenerator() {
}

void DebugInfoGenerator::createCompileUnit(
    const std::string& filename,
    const std::string& directory,
    const std::string& producer) {
    
    // 创建文件
    file_ = builder_->createFile(filename, directory);

    // 创建编译单元
    compileUnit_ = builder_->createCompileUnit(
        llvm::dwarf::DW_LANG_C_plus_plus,  // 语言
        file_,                              // 文件
        producer,                           // 生产者
        false,                              // 是否优化
        "",                                 // 编译标志
        0                                   // 运行时版本
    );
}

void DebugInfoGenerator::createFunctionDebugInfo(
    llvm::Function* function,
    const std::string& name,
    size_t line,
    size_t column) {
    
    if (!compileUnit_ || !file_) {
        return;
    }

    // 创建函数类型
    llvm::SmallVector<llvm::Metadata*, 8> EltTys;
    auto FunctionTy = builder_->createSubroutineType(
        builder_->getOrCreateTypeArray(EltTys));

    // 创建函数调试信息
    auto SP = builder_->createFunction(
        file_,                              // 作用域
        name,                               // 名称
        function->getName(),                // 链接名称
        file_,                              // 文件
        line,                               // 行号
        FunctionTy,                         // 类型
        line,                               // 作用域行号
        llvm::DINode::FlagPrototyped,       // 标志
        llvm::DISubprogram::SPFlagDefinition  // SP标志
    );

    function->setSubprogram(SP);
}

void DebugInfoGenerator::createVariableDebugInfo(
    llvm::Value* value,
    const std::string& name,
    llvm::Type* type,
    size_t line,
    size_t column) {
    
    if (!compileUnit_ || !file_) {
        return;
    }

    // 获取或创建类型的调试信息
    llvm::DIType* diType = getOrCreateType(type);
    if (!diType) {
        return;
    }

    // 获取当前作用域（假设是当前函数）
    llvm::DIScope* scope = compileUnit_;
    
    // 如果value是指令，尝试获取其所在函数的作用域
    if (auto* inst = llvm::dyn_cast<llvm::Instruction>(value)) {
        if (auto* func = inst->getFunction()) {
            if (auto* sp = func->getSubprogram()) {
                scope = sp;
            }
        }
    }

    // 创建局部变量
    auto* localVar = builder_->createAutoVariable(
        scope,                              // 作用域
        name,                               // 变量名
        file_,                              // 文件
        line,                               // 行号
        diType,                             // 类型
        true                                // 总是保留
    );

    // 创建位置
    auto loc = llvm::DILocation::get(
        module_.getContext(),
        line,
        column,
        scope
    );

    // 插入dbg.declare或dbg.value
    if (auto* inst = llvm::dyn_cast<llvm::Instruction>(value)) {
        builder_->insertDeclare(
            value,                          // 变量
            localVar,                       // 变量信息
            builder_->createExpression(),   // 表达式
            loc,                            // 位置
            inst                            // 插入点
        );
    }
}

void DebugInfoGenerator::setLocation(
    llvm::Instruction* inst,
    size_t line,
    size_t column) {
    
    if (!compileUnit_ || !file_) {
        return;
    }

    auto loc = llvm::DILocation::get(
        module_.getContext(),
        line,
        column,
        compileUnit_
    );

    inst->setDebugLoc(loc);
}

void DebugInfoGenerator::finalize() {
    if (builder_) {
        builder_->finalize();
    }
}

llvm::DIType* DebugInfoGenerator::getOrCreateType(llvm::Type* type) {
    if (!type) {
        return nullptr;
    }

    // 生成类型的唯一键
    std::string typeKey = getTypeKey(type);
    
    // 检查缓存
    auto it = typeCache_.find(typeKey);
    if (it != typeCache_.end()) {
        return it->second;
    }

    // 创建新的调试类型
    llvm::DIType* diType = nullptr;

    if (type->isIntegerTy()) {
        // 整数类型
        unsigned bitWidth = type->getIntegerBitWidth();
        diType = builder_->createBasicType(
            "i" + std::to_string(bitWidth),
            bitWidth,
            llvm::dwarf::DW_ATE_signed
        );
    } else if (type->isFloatTy()) {
        // 单精度浮点
        diType = builder_->createBasicType(
            "float",
            32,
            llvm::dwarf::DW_ATE_float
        );
    } else if (type->isDoubleTy()) {
        // 双精度浮点
        diType = builder_->createBasicType(
            "double",
            64,
            llvm::dwarf::DW_ATE_float
        );
    } else if (type->isPointerTy()) {
        // 指针类型
        auto* pointeeType = type->getPointerElementType();
        auto* pointeeDIType = getOrCreateType(pointeeType);
        if (pointeeDIType) {
            diType = builder_->createPointerType(
                pointeeDIType,
                module_.getDataLayout().getPointerSizeInBits()
            );
        }
    } else if (type->isArrayTy()) {
        // 数组类型
        auto* arrayType = llvm::cast<llvm::ArrayType>(type);
        auto* elementType = arrayType->getElementType();
        auto* elementDIType = getOrCreateType(elementType);
        if (elementDIType) {
            llvm::SmallVector<llvm::Metadata*, 1> subscripts;
            subscripts.push_back(builder_->getOrCreateSubrange(
                0, arrayType->getNumElements()));
            diType = builder_->createArrayType(
                arrayType->getNumElements() * 
                    module_.getDataLayout().getTypeAllocSize(elementType) * 8,
                0,
                elementDIType,
                builder_->getOrCreateArray(subscripts)
            );
        }
    } else if (type->isStructTy()) {
        // 结构体类型
        auto* structType = llvm::cast<llvm::StructType>(type);
        std::string structName = structType->hasName() 
            ? structType->getName().str() 
            : "anonymous_struct";
        
        // 创建前向声明
        diType = builder_->createStructType(
            compileUnit_,
            structName,
            file_,
            0,
            module_.getDataLayout().getTypeAllocSize(structType) * 8,
            0,
            llvm::DINode::FlagZero,
            nullptr,
            llvm::DINodeArray()
        );
    } else if (type->isVoidTy()) {
        // void类型
        diType = nullptr;  // void没有调试类型
    }

    // 缓存类型
    if (diType) {
        typeCache_[typeKey] = diType;
    }

    return diType;
}

std::string DebugInfoGenerator::getTypeKey(llvm::Type* type) {
    std::string key;
    llvm::raw_string_ostream os(key);
    type->print(os);
    os.flush();
    return key;
}

} // namespace backend
} // namespace az
