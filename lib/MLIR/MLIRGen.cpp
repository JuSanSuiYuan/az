// AZ编译器 - MLIR生成器实现

#include "AZ/MLIR/MLIRGen.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/CF/IR/CFOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMOps.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringRef.h"

namespace az {
namespace mlirgen {

MLIRGenerator::MLIRGenerator(mlir::MLIRContext& context, SemanticAnalyzer& sema)
    : context_(context), builder_(&context), sema_(sema), filename_("") {

    // 加载必要的方言
    context_.getOrLoadDialect<mlir::func::FuncDialect>();
    context_.getOrLoadDialect<mlir::arith::ArithDialect>();
    context_.getOrLoadDialect<mlir::scf::SCFDialect>();
    context_.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
    context_.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
}

Result<mlir::OwningOpRef<mlir::ModuleOp>> 
MLIRGenerator::generate(Program* program) {
    // 创建模块
    auto loc = getLocation();
    module_ = mlir::ModuleOp::create(loc);
    builder_.setInsertionPointToEnd(module_.getBody());

    // 声明内置函数
    auto declareBuiltin = [&](llvm::StringRef name,
                              llvm::ArrayRef<mlir::Type> argTypes,
                              mlir::Type retType) {
        auto key = name.str();
        if (functionTable_.count(key)) {
            return;
        }
        auto funcType = builder_.getFunctionType(argTypes, retType);
        auto func = builder_.create<mlir::func::FuncOp>(loc, name, funcType);
        func.setPrivate();
        functionTable_[key] = func;
    };

    auto stringPtrType = convertTypeName("string");
    declareBuiltin("print", {stringPtrType}, builder_.getNoneType());
    declareBuiltin("println", {stringPtrType}, builder_.getNoneType());

    // 第一遍：声明所有函数
    for (auto& stmt : program->statements) {
        if (stmt->kind == StmtKind::FuncDecl) {
            auto* funcDecl = static_cast<FuncDeclStmt*>(stmt.get());
            
            // 构建函数类型
            llvm::SmallVector<mlir::Type, 4> argTypes;
            for (auto& param : funcDecl->params) {
                auto argType = convertTypeName(param.type);
                if (!argType) {
                    return Result<mlir::OwningOpRef<mlir::ModuleOp>>::Err(
                        makeError("不支持的参数类型: " + param.type)
                    );
                }
                argTypes.push_back(argType);
            }

            auto returnType = convertTypeName(funcDecl->returnType);
            if (!returnType) {
                return Result<mlir::OwningOpRef<mlir::ModuleOp>>::Err(
                    makeError("不支持的返回类型: " + funcDecl->returnType)
                );
            }
            
            auto funcType = builder_.getFunctionType(argTypes, returnType);
            
            // 创建函数
            auto func = builder_.create<mlir::func::FuncOp>(
                loc, funcDecl->name, funcType
            );
            
            functionTable_[funcDecl->name] = func;
        }
    }
    
    // 第二遍：生成函数体
    for (auto& stmt : program->statements) {
        auto result = genStmt(stmt.get());
        if (result.isErr()) {
            return Result<mlir::OwningOpRef<mlir::ModuleOp>>::Err(result.error());
        }
    }
    
    // 验证模块
    if (failed(mlir::verify(module_))) {
        return Result<mlir::OwningOpRef<mlir::ModuleOp>>::Err(
            makeError("MLIR模块验证失败")
        );
    }
    
    return Result<mlir::OwningOpRef<mlir::ModuleOp>>::Ok(
        mlir::OwningOpRef<mlir::ModuleOp>(module_)
    );
}

Result<void> MLIRGenerator::genStmt(Stmt* stmt) {
    switch (stmt->kind) {
        case StmtKind::VarDecl:
            return genVarDecl(static_cast<VarDeclStmt*>(stmt));
        case StmtKind::FuncDecl:
            return genFuncDecl(static_cast<FuncDeclStmt*>(stmt));
        case StmtKind::Return:
            return genReturn(static_cast<ReturnStmt*>(stmt));
        case StmtKind::If:
            return genIf(static_cast<IfStmt*>(stmt));
        case StmtKind::While:
            return genWhile(static_cast<WhileStmt*>(stmt));
        case StmtKind::Block:
            return genBlock(static_cast<BlockStmt*>(stmt));
        case StmtKind::Expression:
            return genExprStmt(static_cast<ExprStmt*>(stmt));
        default:
            return Result<void>::Ok();
    }
}

Result<void> MLIRGenerator::genVarDecl(VarDeclStmt* stmt) {
    auto loc = getLocation();
    
    // 生成初始化表达式
    if (stmt->initializer) {
        auto valueResult = genExpr(stmt->initializer.get());
        if (valueResult.isErr()) {
            return Result<void>::Err(valueResult.error());
        }
        symbolTable_[stmt->name] = valueResult.value();
    } else {
        auto typeName = stmt->type.empty() ? "int" : stmt->type;
        auto type = convertTypeName(typeName);
        if (!type) {
            return Result<void>::Err(makeError("不支持的变量类型: " + typeName));
        }
        auto zeroValue = createZeroValue(type, loc);
        if (!zeroValue) {
            return Result<void>::Err(makeError("无法为变量初始化默认值: " + stmt->name));
        }
        symbolTable_[stmt->name] = zeroValue;
    }

    return Result<void>::Ok();
}

Result<void> MLIRGenerator::genFuncDecl(FuncDeclStmt* stmt) {
    auto loc = getLocation();
    
    // 获取函数
    auto it = functionTable_.find(stmt->name);
    if (it == functionTable_.end()) {
        return Result<void>::Err(makeError("函数未声明: " + stmt->name));
    }
    
    auto func = it->second;
    
    // 创建入口块
    auto* entryBlock = func.addEntryBlock();
    builder_.setInsertionPointToStart(entryBlock);
    
    // 保存旧的符号表
    auto oldSymbolTable = symbolTable_;
    symbolTable_.clear();
    
    // 添加参数到符号表
    for (size_t i = 0; i < stmt->params.size(); ++i) {
        symbolTable_[stmt->params[i].name] = entryBlock->getArgument(i);
    }
    
    // 生成函数体
    auto result = genStmt(stmt->body.get());
    
    // 恢复符号表
    symbolTable_ = oldSymbolTable;

    if (result.isErr()) {
        return result;
    }

    auto* currentBlock = builder_.getBlock();
    auto needsTerminator = currentBlock &&
        (currentBlock->empty() || !currentBlock->back().hasTrait<mlir::OpTrait::IsTerminator>());

    if (needsTerminator) {
        if (stmt->returnType == "void") {
            builder_.create<mlir::func::ReturnOp>(loc);
        } else {
            auto retType = convertTypeName(stmt->returnType);
            auto zero = createZeroValue(retType, loc);
            if (!zero) {
                return Result<void>::Err(makeError("无法生成默认返回值: " + stmt->returnType));
            }
            builder_.create<mlir::func::ReturnOp>(loc, zero);
        }
    }

    builder_.setInsertionPointToEnd(module_.getBody());
    return Result<void>::Ok();
}

Result<void> MLIRGenerator::genReturn(ReturnStmt* stmt) {
    auto loc = getLocation();
    
    if (stmt->expr) {
        auto valueResult = genExpr(stmt->expr.get());
        if (valueResult.isErr()) {
            return Result<void>::Err(valueResult.error());
        }
        
        builder_.create<mlir::func::ReturnOp>(loc, valueResult.value());
    } else {
        builder_.create<mlir::func::ReturnOp>(loc);
    }
    
    return Result<void>::Ok();
}

Result<void> MLIRGenerator::genIf(IfStmt* stmt) {
    auto loc = getLocation();

    auto cond = genExpr(stmt->condition.get());
    if (cond.isErr()) return Result<void>::Err(cond.error());

    auto condBool = convertToBool(stmt->condition.get(), cond.value(), loc);
    if (condBool.isErr()) return Result<void>::Err(condBool.error());

    bool hasElse = stmt->elseBranch != nullptr;
    auto ifOp = builder_.create<mlir::scf::IfOp>(loc, mlir::TypeRange(), condBool.value(), hasElse);

    {
        mlir::OpBuilder::InsertionGuard guard(builder_);
        builder_.setInsertionPointToStart(ifOp.thenBlock());
        auto savedSymbols = symbolTable_;
        auto result = genStmt(stmt->thenBranch.get());
        symbolTable_ = savedSymbols;
        if (result.isErr()) return result;
        auto* block = builder_.getBlock();
        if (!block || block->empty() || !block->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
            builder_.create<mlir::scf::YieldOp>(loc);
        }
    }

    if (hasElse) {
        mlir::OpBuilder::InsertionGuard guard(builder_);
        builder_.setInsertionPointToStart(ifOp.elseBlock());
        auto savedSymbols = symbolTable_;
        auto result = genStmt(stmt->elseBranch.get());
        symbolTable_ = savedSymbols;
        if (result.isErr()) return result;
        auto* block = builder_.getBlock();
        if (!block || block->empty() || !block->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
            builder_.create<mlir::scf::YieldOp>(loc);
        }
    }

    return Result<void>::Ok();
}

Result<void> MLIRGenerator::genWhile(WhileStmt* stmt) {
    auto loc = getLocation();

    auto whileOp = builder_.create<mlir::scf::WhileOp>(loc, mlir::TypeRange(), mlir::ValueRange());

    {
        mlir::OpBuilder::InsertionGuard guard(builder_);
        builder_.setInsertionPointToStart(&whileOp.getBefore().front());

        auto cond = genExpr(stmt->condition.get());
        if (cond.isErr()) return Result<void>::Err(cond.error());
        auto condBool = convertToBool(stmt->condition.get(), cond.value(), loc);
        if (condBool.isErr()) return Result<void>::Err(condBool.error());

        builder_.create<mlir::scf::ConditionOp>(loc, condBool.value(), whileOp.getBeforeArguments());
    }

    {
        mlir::OpBuilder::InsertionGuard guard(builder_);
        builder_.setInsertionPointToStart(&whileOp.getAfter().front());
        auto savedSymbols = symbolTable_;
        auto bodyResult = genStmt(stmt->body.get());
        symbolTable_ = savedSymbols;
        if (bodyResult.isErr()) return bodyResult;
        auto* block = builder_.getBlock();
        if (!block || block->empty() || !block->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
            builder_.create<mlir::scf::YieldOp>(loc);
        }
    }

    return Result<void>::Ok();
}

Result<void> MLIRGenerator::genBlock(BlockStmt* stmt) {
    auto scopeBackup = symbolTable_;
    auto guard = llvm::make_scope_exit([&]() { symbolTable_ = scopeBackup; });

    for (auto& s : stmt->statements) {
        auto result = genStmt(s.get());
        if (result.isErr()) {
            return result;
        }
    }

    return Result<void>::Ok();
}

Result<void> MLIRGenerator::genExprStmt(ExprStmt* stmt) {
    auto result = genExpr(stmt->expr.get());
    if (result.isErr()) {
        return Result<void>::Err(result.error());
    }
    return Result<void>::Ok();
}

// 表达式生成
Result<mlir::Value> MLIRGenerator::genExpr(Expr* expr) {
    switch (expr->kind) {
        case ExprKind::IntLiteral:
            return genIntLiteral(static_cast<IntLiteralExpr*>(expr));
        case ExprKind::FloatLiteral:
            return genFloatLiteral(static_cast<FloatLiteralExpr*>(expr));
        case ExprKind::StringLiteral:
            return genStringLiteral(static_cast<StringLiteralExpr*>(expr));
        case ExprKind::Identifier:
            return genIdentifier(static_cast<IdentifierExpr*>(expr));
        case ExprKind::Binary:
            return genBinary(static_cast<BinaryExpr*>(expr));
        case ExprKind::Unary:
            return genUnary(static_cast<UnaryExpr*>(expr));
        case ExprKind::Call:
            return genCall(static_cast<CallExpr*>(expr));
        default:
            return Result<mlir::Value>::Err(makeError("未知表达式类型"));
    }
}

Result<mlir::Value> MLIRGenerator::genIntLiteral(IntLiteralExpr* expr) {
    auto loc = getLocation();
    auto type = builder_.getI32Type();
    auto attr = builder_.getI32IntegerAttr(expr->value);
    auto value = builder_.create<mlir::arith::ConstantOp>(loc, type, attr);
    return Result<mlir::Value>::Ok(value.getResult());
}

Result<mlir::Value> MLIRGenerator::genFloatLiteral(FloatLiteralExpr* expr) {
    auto loc = getLocation();
    auto type = builder_.getF64Type();
    auto attr = builder_.getF64FloatAttr(expr->value);
    auto value = builder_.create<mlir::arith::ConstantOp>(loc, type, attr);
    return Result<mlir::Value>::Ok(value.getResult());
}

Result<mlir::Value> MLIRGenerator::genStringLiteral(StringLiteralExpr* expr) {
    auto loc = getLocation();

    auto it = stringLiteralGlobals_.find(expr->value);
    if (it == stringLiteralGlobals_.end()) {
        auto globalName = "__az_str_" + std::to_string(stringLiteralCounter_++);
        auto charType = builder_.getI8Type();
        auto arrayType = mlir::LLVM::LLVMArrayType::get(charType, expr->value.size() + 1);

        auto bodyGuard = llvm::make_scope_exit([&]() { builder_.setInsertionPointToEnd(module_.getBody()); });
        auto insertPt = builder_.saveInsertionPoint();
        builder_.setInsertionPointToStart(module_.getBody());

        auto dataAttr = builder_.getStringAttr(expr->value + "\0");
        auto global = builder_.create<mlir::LLVM::GlobalOp>(
            loc,
            arrayType,
            true,
            mlir::LLVM::Linkage::Internal,
            globalName,
            dataAttr
        );

        auto symbol = builder_.getSymbolRefAttr(global);
        stringLiteralGlobals_[expr->value] = symbol;
        builder_.restoreInsertionPoint(insertPt);
        it = stringLiteralGlobals_.find(expr->value);
    }

    auto ptrType = mlir::LLVM::LLVMPointerType::get(builder_.getI8Type());
    auto addr = builder_.create<mlir::LLVM::AddressOfOp>(loc, ptrType, it->second);
    return Result<mlir::Value>::Ok(addr.getResult());
}

Result<mlir::Value> MLIRGenerator::genIdentifier(IdentifierExpr* expr) {
    auto it = symbolTable_.find(expr->name);
    if (it == symbolTable_.end()) {
        return Result<mlir::Value>::Err(
            makeError("未定义的变量: " + expr->name)
        );
    }
    return Result<mlir::Value>::Ok(it->second);
}

Result<mlir::Value> MLIRGenerator::genBinary(BinaryExpr* expr) {
    auto loc = getLocation();
    
    // 生成左右操作数
    auto leftResult = genExpr(expr->left.get());
    if (leftResult.isErr()) return leftResult;
    
    auto rightResult = genExpr(expr->right.get());
    if (rightResult.isErr()) return rightResult;
    
    auto left = leftResult.value();
    auto right = rightResult.value();
    
    // 根据运算符生成相应的MLIR操作
    Type* leftTypeInfo = sema_.getExprType(expr->left.get());
    Type* rightTypeInfo = sema_.getExprType(expr->right.get());
    bool leftFloat = leftTypeInfo && leftTypeInfo->isFloat();
    bool rightFloat = rightTypeInfo && rightTypeInfo->isFloat();
    bool resultFloat = (leftFloat || rightFloat);

    auto promoteBoth = [&](bool targetFloat) -> Result<std::pair<mlir::Value, mlir::Value>> {
        auto leftConverted = promoteNumeric(left, leftTypeInfo, targetFloat, loc);
        if (leftConverted.isErr()) return Result<std::pair<mlir::Value, mlir::Value>>::Err(leftConverted.error());
        auto rightConverted = promoteNumeric(right, rightTypeInfo, targetFloat, loc);
        if (rightConverted.isErr()) return Result<std::pair<mlir::Value, mlir::Value>>::Err(rightConverted.error());
        return Result<std::pair<mlir::Value, mlir::Value>>::Ok({leftConverted.value(), rightConverted.value()});
    };

    if (expr->op == "+" || expr->op == "-" || expr->op == "*" || expr->op == "/" || expr->op == "%") {
        bool useFloat = (expr->op == "/") ? resultFloat : (leftFloat || rightFloat);
        auto lhsPromoted = promoteNumeric(left, leftTypeInfo, useFloat, loc);
        if (lhsPromoted.isErr()) return lhsPromoted;
        auto rhsPromoted = promoteNumeric(right, rightTypeInfo, useFloat, loc);
        if (rhsPromoted.isErr()) return rhsPromoted;

        auto lhsVal = lhsPromoted.value();
        auto rhsVal = rhsPromoted.value();

        if (expr->op == "+") {
            if (useFloat) {
                auto op = builder_.create<mlir::arith::AddFOp>(loc, lhsVal, rhsVal);
                return Result<mlir::Value>::Ok(op.getResult());
            }
            auto op = builder_.create<mlir::arith::AddIOp>(loc, lhsVal, rhsVal);
            return Result<mlir::Value>::Ok(op.getResult());
        }
        if (expr->op == "-") {
            if (useFloat) {
                auto op = builder_.create<mlir::arith::SubFOp>(loc, lhsVal, rhsVal);
                return Result<mlir::Value>::Ok(op.getResult());
            }
            auto op = builder_.create<mlir::arith::SubIOp>(loc, lhsVal, rhsVal);
            return Result<mlir::Value>::Ok(op.getResult());
        }
        if (expr->op == "*") {
            if (useFloat) {
                auto op = builder_.create<mlir::arith::MulFOp>(loc, lhsVal, rhsVal);
                return Result<mlir::Value>::Ok(op.getResult());
            }
            auto op = builder_.create<mlir::arith::MulIOp>(loc, lhsVal, rhsVal);
            return Result<mlir::Value>::Ok(op.getResult());
        }
        if (expr->op == "/") {
            if (useFloat) {
                auto op = builder_.create<mlir::arith::DivFOp>(loc, lhsVal, rhsVal);
                return Result<mlir::Value>::Ok(op.getResult());
            }
            auto op = builder_.create<mlir::arith::DivSIOp>(loc, lhsVal, rhsVal);
            return Result<mlir::Value>::Ok(op.getResult());
        }

        auto lhsInt = promoteNumeric(left, leftTypeInfo, false, loc);
        if (lhsInt.isErr()) return lhsInt;
        auto rhsInt = promoteNumeric(right, rightTypeInfo, false, loc);
        if (rhsInt.isErr()) return rhsInt;
        auto op = builder_.create<mlir::arith::RemSIOp>(loc, lhsInt.value(), rhsInt.value());
        return Result<mlir::Value>::Ok(op.getResult());
    }

    if (expr->op == "==" || expr->op == "!=" || expr->op == ">" || expr->op == ">=" || expr->op == "<" || expr->op == "<=") {
        auto lhsPromoted = promoteNumeric(left, leftTypeInfo, resultFloat, loc);
        if (lhsPromoted.isErr()) return lhsPromoted;
        auto rhsPromoted = promoteNumeric(right, rightTypeInfo, resultFloat, loc);
        if (rhsPromoted.isErr()) return rhsPromoted;

        auto lhsVal = lhsPromoted.value();
        auto rhsVal = rhsPromoted.value();

        if (resultFloat) {
            mlir::arith::CmpFPredicate pred;
            if (expr->op == "==") pred = mlir::arith::CmpFPredicate::UEQ;
            else if (expr->op == "!=") pred = mlir::arith::CmpFPredicate::UNE;
            else if (expr->op == ">") pred = mlir::arith::CmpFPredicate::UGT;
            else if (expr->op == ">=") pred = mlir::arith::CmpFPredicate::UGE;
            else if (expr->op == "<") pred = mlir::arith::CmpFPredicate::ULT;
            else pred = mlir::arith::CmpFPredicate::ULE;
            auto op = builder_.create<mlir::arith::CmpFOp>(loc, pred, lhsVal, rhsVal);
            return Result<mlir::Value>::Ok(op.getResult());
        }

        mlir::arith::CmpIPredicate pred;
        if (expr->op == "==") pred = mlir::arith::CmpIPredicate::eq;
        else if (expr->op == "!=") pred = mlir::arith::CmpIPredicate::ne;
        else if (expr->op == ">") pred = mlir::arith::CmpIPredicate::sgt;
        else if (expr->op == ">=") pred = mlir::arith::CmpIPredicate::sge;
        else if (expr->op == "<") pred = mlir::arith::CmpIPredicate::slt;
        else pred = mlir::arith::CmpIPredicate::sle;
        auto op = builder_.create<mlir::arith::CmpIOp>(loc, pred, lhsVal, rhsVal);
        return Result<mlir::Value>::Ok(op.getResult());
    }

    if (expr->op == "&&" || expr->op == "||") {
        auto lhsBool = convertToBool(expr->left.get(), left, loc);
        if (lhsBool.isErr()) return Result<mlir::Value>::Err(lhsBool.error());
        auto rhsBool = convertToBool(expr->right.get(), right, loc);
        if (rhsBool.isErr()) return Result<mlir::Value>::Err(rhsBool.error());
        if (expr->op == "&&") {
            auto op = builder_.create<mlir::arith::AndIOp>(loc, lhsBool.value(), rhsBool.value());
            return Result<mlir::Value>::Ok(op.getResult());
        }
        auto op = builder_.create<mlir::arith::OrIOp>(loc, lhsBool.value(), rhsBool.value());
        return Result<mlir::Value>::Ok(op.getResult());
    }

    return Result<mlir::Value>::Err(makeError("不支持的运算符: " + expr->op));
}

Result<mlir::Value> MLIRGenerator::genUnary(UnaryExpr* expr) {
    auto loc = getLocation();

    auto operandResult = genExpr(expr->operand.get());
    if (operandResult.isErr()) return operandResult;

    auto operandType = sema_.getExprType(expr->operand.get());
    auto operand = operandResult.value();

    if (expr->op == "-") {
        if (operandType && operandType->isFloat()) {
            auto zero = builder_.create<mlir::arith::ConstantOp>(loc, builder_.getF64Type(), builder_.getF64FloatAttr(0.0));
            auto op = builder_.create<mlir::arith::SubFOp>(loc, zero.getResult(), operand);
            return Result<mlir::Value>::Ok(op.getResult());
        }
        auto zero = builder_.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
        auto op = builder_.create<mlir::arith::SubIOp>(loc, zero.getResult(), operand);
        return Result<mlir::Value>::Ok(op.getResult());
    }

    if (expr->op == "!") {
        auto boolVal = convertToBool(expr->operand.get(), operand, loc);
        if (boolVal.isErr()) return boolVal;
        auto one = builder_.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
        auto op = builder_.create<mlir::arith::XOrIOp>(loc, boolVal.value(), one.getResult());
        return Result<mlir::Value>::Ok(op.getResult());
    }

    return Result<mlir::Value>::Err(makeError("未知一元运算符: " + expr->op));
}

Result<mlir::Value> MLIRGenerator::genCall(CallExpr* expr) {
    auto loc = getLocation();
    
    // 获取函数名
    if (expr->callee->kind != ExprKind::Identifier) {
        return Result<mlir::Value>::Err(makeError("只支持直接函数调用"));
    }
    
    auto* identExpr = static_cast<IdentifierExpr*>(expr->callee.get());
    auto funcName = identExpr->name;
    
    if (funcName == "println" || funcName == "print") {
        if (expr->arguments.size() != 1) {
            return Result<mlir::Value>::Err(makeError(funcName + " 目前仅支持1个字符串参数"));
        }

        auto argValue = genExpr(expr->arguments[0].get());
        if (argValue.isErr()) return argValue;

        auto argType = sema_.getExprType(expr->arguments[0].get());
        if (!argType || !argType->isString()) {
            return Result<mlir::Value>::Err(makeError(funcName + " 目前仅支持字符串参数"));
        }

        auto builtinIt = functionTable_.find(funcName);
        if (builtinIt == functionTable_.end()) {
            return Result<mlir::Value>::Err(makeError("未声明内置函数: " + funcName));
        }

        builder_.create<mlir::func::CallOp>(loc, builtinIt->second.getName(), mlir::TypeRange(), argValue.value());
        return Result<mlir::Value>::Ok(mlir::Value());
    }
    
    // 查找函数
    auto it = functionTable_.find(funcName);
    if (it == functionTable_.end()) {
        return Result<mlir::Value>::Err(
            makeError("未定义的函数: " + funcName)
        );
    }
    
    // 生成参数
    llvm::SmallVector<mlir::Value, 4> args;
    for (auto& arg : expr->arguments) {
        auto argResult = genExpr(arg.get());
        if (argResult.isErr()) return argResult;
        args.push_back(argResult.value());
    }
    
    // 生成函数调用
    auto call = builder_.create<mlir::func::CallOp>(
        loc, it->second, args
    );
    
    if (call.getNumResults() > 0) {
        return Result<mlir::Value>::Ok(call.getResult(0));
    }
    
    return Result<mlir::Value>::Ok(mlir::Value());
}

// 类型转换
mlir::Type MLIRGenerator::convertType(Type* type) {
    if (type->isInt()) {
        return builder_.getI32Type();
    } else if (type->isFloat()) {
        return builder_.getF64Type();
    } else if (type->isVoid()) {
        return builder_.getNoneType();
    }
    return builder_.getNoneType();
}

mlir::Type MLIRGenerator::convertTypeName(const std::string& name) {
    if (name == "void") return builder_.getNoneType();
    if (name == "int") return builder_.getI32Type();
    if (name == "float") return builder_.getF64Type();
    if (name == "bool") return builder_.getI1Type();
    if (name == "string") return mlir::LLVM::LLVMPointerType::get(builder_.getI8Type());
    return {};
}

Result<mlir::Value> MLIRGenerator::promoteNumeric(mlir::Value value, Type* fromType, bool targetFloat, mlir::Location loc) {
    if (!fromType) return Result<mlir::Value>::Ok(value);
    auto type = value.getType();

    if (targetFloat) {
        if (type.isF64()) return Result<mlir::Value>::Ok(value);
        if (type.isInteger(32)) {
            auto op = builder_.create<mlir::arith::SIToFPOp>(loc, builder_.getF64Type(), value);
            return Result<mlir::Value>::Ok(op.getResult());
        }
    } else {
        if (type.isInteger(32)) return Result<mlir::Value>::Ok(value);
        if (type.isF64()) {
            auto op = builder_.create<mlir::arith::FPToSIOp>(loc, builder_.getI32Type(), value);
            return Result<mlir::Value>::Ok(op.getResult());
        }
    }

    return Result<mlir::Value>::Ok(value);
}

Result<mlir::Value> MLIRGenerator::convertToBool(Expr* expr, mlir::Value value, mlir::Location loc) {
    auto type = value.getType();
    if (type.isInteger(1)) return Result<mlir::Value>::Ok(value);
    if (type.isInteger(32) || type.isInteger(64)) {
        auto zero = builder_.create<mlir::arith::ConstantIntOp>(loc, 0, type.cast<mlir::IntegerType>().getWidth());
        auto op = builder_.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne, value, zero.getResult());
        return Result<mlir::Value>::Ok(op.getResult());
    }
    if (type.isF64()) {
        auto zero = builder_.create<mlir::arith::ConstantOp>(loc, builder_.getF64Type(), builder_.getF64FloatAttr(0.0));
        auto op = builder_.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::UNE, value, zero.getResult());
        return Result<mlir::Value>::Ok(op.getResult());
    }
    if (type.isa<mlir::LLVM::LLVMPointerType>()) {
        auto nullPtr = builder_.create<mlir::LLVM::NullOp>(loc, type);
        auto ptrToInt = builder_.create<mlir::LLVM::PtrToIntOp>(loc, builder_.getI64Type(), value);
        auto nullInt = builder_.create<mlir::LLVM::PtrToIntOp>(loc, builder_.getI64Type(), nullPtr.getResult());
        auto cmp = builder_.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne, ptrToInt.getResult(), nullInt.getResult());
        return Result<mlir::Value>::Ok(cmp.getResult());
    }
    return Result<mlir::Value>::Err(makeError("无法将表达式转换为bool"));
}

mlir::Value MLIRGenerator::createZeroValue(mlir::Type type, mlir::Location loc) {
    if (!type) type = builder_.getI32Type();
    if (type.isInteger(1)) {
        return builder_.create<mlir::arith::ConstantIntOp>(loc, 0, 1).getResult();
    }
    if (auto intType = type.dyn_cast<mlir::IntegerType>()) {
        return builder_.create<mlir::arith::ConstantIntOp>(loc, 0, intType.getWidth()).getResult();
    }
    if (type.isF64()) {
        return builder_.create<mlir::arith::ConstantOp>(loc, builder_.getF64Type(), builder_.getF64FloatAttr(0.0)).getResult();
    }
    if (type.isa<mlir::LLVM::LLVMPointerType>()) {
        return builder_.create<mlir::LLVM::NullOp>(loc, type).getResult();
    }
    if (type.isa<mlir::NoneType>()) return {};
    return builder_.create<mlir::arith::ConstantIntOp>(loc, 0, 32).getResult();
}

mlir::Value MLIRGenerator::createBoolConstant(bool value, mlir::Location loc) {
    return builder_.create<mlir::arith::ConstantIntOp>(loc, value ? 1 : 0, 1).getResult();
}

mlir::Location MLIRGenerator::getLocation(size_t, size_t) {
    return builder_.getUnknownLoc();
}

CompileError MLIRGenerator::makeError(const std::string& message) {
    return CompileError{ErrorKind::SemanticError, message, 0, 0, filename_};
}

} // namespace mlirgen
} // namespace az
