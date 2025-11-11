// AZ编译器 - 语义分析器实现

#include "AZ/Frontend/Sema.h"
#include <sstream>

namespace az {
namespace frontend {

// Type实现
std::string Type::toString() const {
    switch (kind) {
        case TypeKind::Void: return "void";
        case TypeKind::Int: return "int";
        case TypeKind::Float: return "float";
        case TypeKind::String: return "string";
        case TypeKind::Bool: return "bool";
        case TypeKind::Function: {
            std::string result = "(";
            for (size_t i = 0; i < paramTypes.size(); ++i) {
                if (i > 0) result += ", ";
                result += paramTypes[i]->toString();
            }
            result += ") -> ";
            result += returnType ? returnType->toString() : "void";
            return result;
        }
        case TypeKind::Array:
            return elementType->toString() + "[]";
        default:
            return "unknown";
    }
}

// SymbolTable实现
bool SymbolTable::addSymbol(const std::string& name, Symbol symbol) {
    if (symbols_.find(name) != symbols_.end()) {
        return false; // 符号已存在
    }
    symbols_[name] = std::move(symbol);
    return true;
}

Symbol* SymbolTable::findSymbol(const std::string& name) {
    auto it = symbols_.find(name);
    if (it != symbols_.end()) {
        return &it->second;
    }
    if (parent_) {
        return parent_->findSymbol(name);
    }
    return nullptr;
}

bool SymbolTable::hasSymbol(const std::string& name) const {
    return symbols_.find(name) != symbols_.end();
}

// SemanticAnalyzer实现
SemanticAnalyzer::SemanticAnalyzer()
    : currentFunction_(nullptr), filename_("") {
    
    // 创建内置类型
    voidType_ = new Type(TypeKind::Void, "void");
    intType_ = new Type(TypeKind::Int, "int");
    floatType_ = new Type(TypeKind::Float, "float");
    stringType_ = new Type(TypeKind::String, "string");
    boolType_ = new Type(TypeKind::Bool, "bool");
    
    types_.emplace_back(voidType_);
    types_.emplace_back(intType_);
    types_.emplace_back(floatType_);
    types_.emplace_back(stringType_);
    types_.emplace_back(boolType_);
    
    // 创建全局作用域
    globalScope_ = new SymbolTable();
    currentScope_ = globalScope_;
    scopes_.emplace_back(globalScope_);
    
    // 添加内置函数
    // println
    auto printlnType = createFunctionType({}, voidType_);
    globalScope_->addSymbol("println", Symbol("println", printlnType, false, true));
    
    // print
    auto printType = createFunctionType({}, voidType_);
    globalScope_->addSymbol("print", Symbol("print", printType, false, true));
}

SemanticAnalyzer::~SemanticAnalyzer() = default;

Result<void> SemanticAnalyzer::analyze(Program* program) {
    // 第一遍：收集所有函数声明
    for (auto& stmt : program->statements) {
        if (stmt->kind == StmtKind::FuncDecl) {
            auto* funcDecl = static_cast<FuncDeclStmt*>(stmt.get());
            
            // 创建函数类型
            std::vector<Type*> paramTypes;
            for (auto& param : funcDecl->params) {
                auto* paramType = getType(param.type);
                if (!paramType) {
                    return Result<void>::Err(makeError(
                        ErrorKind::TypeError,
                        "未知类型: " + param.type
                    ));
                }
                paramTypes.push_back(paramType);
            }
            
            auto* returnType = getType(funcDecl->returnType);
            if (!returnType) {
                return Result<void>::Err(makeError(
                    ErrorKind::TypeError,
                    "未知返回类型: " + funcDecl->returnType
                ));
            }
            
            auto* funcType = createFunctionType(paramTypes, returnType);
            
            // 添加到符号表
            if (!globalScope_->addSymbol(funcDecl->name, 
                Symbol(funcDecl->name, funcType, false, true))) {
                return Result<void>::Err(makeError(
                    ErrorKind::SemanticError,
                    "函数重复定义: " + funcDecl->name
                ));
            }
        }
    }
    
    // 第二遍：分析所有语句
    for (auto& stmt : program->statements) {
        auto result = analyzeStmt(stmt.get());
        if (result.isErr()) {
            return result;
        }
    }
    
    // 检查是否有main函数
    auto* mainSym = globalScope_->findSymbol("main");
    if (!mainSym || !mainSym->isFunction) {
        return Result<void>::Err(makeError(
            ErrorKind::SemanticError,
            "未找到main函数"
        ));
    }
    
    return Result<void>::Ok();
}

Result<void> SemanticAnalyzer::analyzeStmt(Stmt* stmt) {
    switch (stmt->kind) {
        case StmtKind::VarDecl:
            return analyzeVarDecl(static_cast<VarDeclStmt*>(stmt));
        case StmtKind::FuncDecl:
            return analyzeFuncDecl(static_cast<FuncDeclStmt*>(stmt));
        case StmtKind::Return:
            return analyzeReturn(static_cast<ReturnStmt*>(stmt));
        case StmtKind::If:
            return analyzeIf(static_cast<IfStmt*>(stmt));
        case StmtKind::While:
            return analyzeWhile(static_cast<WhileStmt*>(stmt));
        case StmtKind::Block:
            return analyzeBlock(static_cast<BlockStmt*>(stmt));
        case StmtKind::Expression:
            return analyzeExprStmt(static_cast<ExprStmt*>(stmt));
        default:
            return Result<void>::Ok();
    }
}

Result<void> SemanticAnalyzer::analyzeVarDecl(VarDeclStmt* stmt) {
    // 获取类型
    Type* varType = nullptr;
    if (!stmt->type.empty()) {
        varType = getType(stmt->type);
        if (!varType) {
            return Result<void>::Err(makeError(
                ErrorKind::TypeError,
                "未知类型: " + stmt->type
            ));
        }
    }
    
    // 分析初始化表达式
    if (stmt->initializer) {
        auto exprTypeResult = analyzeExpr(stmt->initializer.get());
        if (exprTypeResult.isErr()) {
            return Result<void>::Err(exprTypeResult.error());
        }
        
        auto* exprType = exprTypeResult.value();
        
        // 类型推导
        if (!varType) {
            varType = exprType;
        } else {
            // 类型检查
            if (!isCompatible(varType, exprType)) {
                return Result<void>::Err(makeError(
                    ErrorKind::TypeError,
                    "类型不匹配: 期望 " + varType->toString() + 
                    ", 得到 " + exprType->toString()
                ));
            }
        }
    }
    
    if (!varType) {
        varType = intType_; // 默认类型
    }
    
    // 添加到符号表
    if (!currentScope_->addSymbol(stmt->name, 
        Symbol(stmt->name, varType, stmt->isMutable, false))) {
        return Result<void>::Err(makeError(
            ErrorKind::SemanticError,
            "变量重复定义: " + stmt->name
        ));
    }
    
    return Result<void>::Ok();
}

Result<void> SemanticAnalyzer::analyzeFuncDecl(FuncDeclStmt* stmt) {
    // 保存当前函数
    auto* prevFunction = currentFunction_;
    currentFunction_ = stmt;
    
    // 创建函数作用域
    pushScope();
    
    // 添加参数到作用域
    for (auto& param : stmt->params) {
        auto* paramType = getType(param.type);
        if (!paramType) {
            popScope();
            currentFunction_ = prevFunction;
            return Result<void>::Err(makeError(
                ErrorKind::TypeError,
                "未知参数类型: " + param.type
            ));
        }
        
        if (!currentScope_->addSymbol(param.name, 
            Symbol(param.name, paramType, false, false))) {
            popScope();
            currentFunction_ = prevFunction;
            return Result<void>::Err(makeError(
                ErrorKind::SemanticError,
                "参数重复定义: " + param.name
            ));
        }
    }
    
    // 分析函数体
    auto result = analyzeStmt(stmt->body.get());
    
    // 恢复作用域
    popScope();
    currentFunction_ = prevFunction;
    
    return result;
}

Result<void> SemanticAnalyzer::analyzeReturn(ReturnStmt* stmt) {
    if (!currentFunction_) {
        return Result<void>::Err(makeError(
            ErrorKind::SemanticError,
            "return语句必须在函数内"
        ));
    }
    
    auto* expectedType = getType(currentFunction_->returnType);
    
    if (stmt->expr) {
        auto exprTypeResult = analyzeExpr(stmt->expr.get());
        if (exprTypeResult.isErr()) {
            return Result<void>::Err(exprTypeResult.error());
        }
        
        auto* exprType = exprTypeResult.value();
        if (!isCompatible(expectedType, exprType)) {
            return Result<void>::Err(makeError(
                ErrorKind::TypeError,
                "返回类型不匹配: 期望 " + expectedType->toString() + 
                ", 得到 " + exprType->toString()
            ));
        }
    } else {
        if (!expectedType->isVoid()) {
            return Result<void>::Err(makeError(
                ErrorKind::TypeError,
                "函数应该返回 " + expectedType->toString()
            ));
        }
    }
    
    return Result<void>::Ok();
}

Result<void> SemanticAnalyzer::analyzeIf(IfStmt* stmt) {
    // 分析条件
    auto condTypeResult = analyzeExpr(stmt->condition.get());
    if (condTypeResult.isErr()) {
        return Result<void>::Err(condTypeResult.error());
    }
    
    auto* condType = condTypeResult.value();
    if (!condType->isBool() && !condType->isInt()) {
        return Result<void>::Err(makeError(
            ErrorKind::TypeError,
            "if条件必须是bool或int类型"
        ));
    }
    
    // 分析then分支
    auto thenResult = analyzeStmt(stmt->thenBranch.get());
    if (thenResult.isErr()) {
        return thenResult;
    }
    
    // 分析else分支
    if (stmt->elseBranch) {
        return analyzeStmt(stmt->elseBranch.get());
    }
    
    return Result<void>::Ok();
}

Result<void> SemanticAnalyzer::analyzeWhile(WhileStmt* stmt) {
    // 分析条件
    auto condTypeResult = analyzeExpr(stmt->condition.get());
    if (condTypeResult.isErr()) {
        return Result<void>::Err(condTypeResult.error());
    }
    
    auto* condType = condTypeResult.value();
    if (!condType->isBool() && !condType->isInt()) {
        return Result<void>::Err(makeError(
            ErrorKind::TypeError,
            "while条件必须是bool或int类型"
        ));
    }
    
    // 分析循环体
    return analyzeStmt(stmt->body.get());
}

Result<void> SemanticAnalyzer::analyzeBlock(BlockStmt* stmt) {
    pushScope();
    
    for (auto& s : stmt->statements) {
        auto result = analyzeStmt(s.get());
        if (result.isErr()) {
            popScope();
            return result;
        }
    }
    
    popScope();
    return Result<void>::Ok();
}

Result<void> SemanticAnalyzer::analyzeExprStmt(ExprStmt* stmt) {
    auto result = analyzeExpr(stmt->expr.get());
    if (result.isErr()) {
        return Result<void>::Err(result.error());
    }
    return Result<void>::Ok();
}

// 表达式分析
Result<Type*> SemanticAnalyzer::analyzeExpr(Expr* expr) {
    // 检查缓存
    auto it = exprTypes_.find(expr);
    if (it != exprTypes_.end()) {
        return Result<Type*>::Ok(it->second);
    }
    
    Result<Type*> result;
    
    switch (expr->kind) {
        case ExprKind::IntLiteral:
            result = analyzeIntLiteral(static_cast<IntLiteralExpr*>(expr));
            break;
        case ExprKind::FloatLiteral:
            result = analyzeFloatLiteral(static_cast<FloatLiteralExpr*>(expr));
            break;
        case ExprKind::StringLiteral:
            result = analyzeStringLiteral(static_cast<StringLiteralExpr*>(expr));
            break;
        case ExprKind::Identifier:
            result = analyzeIdentifier(static_cast<IdentifierExpr*>(expr));
            break;
        case ExprKind::Binary:
            result = analyzeBinary(static_cast<BinaryExpr*>(expr));
            break;
        case ExprKind::Unary:
            result = analyzeUnary(static_cast<UnaryExpr*>(expr));
            break;
        case ExprKind::Call:
            result = analyzeCall(static_cast<CallExpr*>(expr));
            break;
        case ExprKind::Member:
            result = analyzeMember(static_cast<MemberExpr*>(expr));
            break;
        default:
            return Result<Type*>::Err(makeError(
                ErrorKind::SemanticError,
                "未知表达式类型"
            ));
    }
    
    // 缓存结果
    if (result.isOk()) {
        exprTypes_[expr] = result.value();
    }
    
    return result;
}

Result<Type*> SemanticAnalyzer::analyzeIntLiteral(IntLiteralExpr* expr) {
    return Result<Type*>::Ok(intType_);
}

Result<Type*> SemanticAnalyzer::analyzeFloatLiteral(FloatLiteralExpr* expr) {
    return Result<Type*>::Ok(floatType_);
}

Result<Type*> SemanticAnalyzer::analyzeStringLiteral(StringLiteralExpr* expr) {
    return Result<Type*>::Ok(stringType_);
}

Result<Type*> SemanticAnalyzer::analyzeIdentifier(IdentifierExpr* expr) {
    auto* symbol = currentScope_->findSymbol(expr->name);
    if (!symbol) {
        return Result<Type*>::Err(makeError(
            ErrorKind::SemanticError,
            "未定义的标识符: " + expr->name
        ));
    }
    return Result<Type*>::Ok(symbol->type);
}

Result<Type*> SemanticAnalyzer::analyzeBinary(BinaryExpr* expr) {
    auto leftResult = analyzeExpr(expr->left.get());
    if (leftResult.isErr()) {
        return leftResult;
    }
    
    auto rightResult = analyzeExpr(expr->right.get());
    if (rightResult.isErr()) {
        return rightResult;
    }
    
    auto* leftType = leftResult.value();
    auto* rightType = rightResult.value();
    
    // 算术运算符
    if (expr->op == "+" || expr->op == "-" || expr->op == "*" || 
        expr->op == "/" || expr->op == "%") {
        
        if (leftType->isInt() && rightType->isInt()) {
            return Result<Type*>::Ok(intType_);
        }
        if ((leftType->isInt() || leftType->isFloat()) && 
            (rightType->isInt() || rightType->isFloat())) {
            return Result<Type*>::Ok(floatType_);
        }
        if (expr->op == "+" && leftType->isString() && rightType->isString()) {
            return Result<Type*>::Ok(stringType_);
        }
        
        return Result<Type*>::Err(makeError(
            ErrorKind::TypeError,
            "运算符 " + expr->op + " 不支持类型 " + 
            leftType->toString() + " 和 " + rightType->toString()
        ));
    }
    
    // 比较运算符
    if (expr->op == "==" || expr->op == "!=" || 
        expr->op == "<" || expr->op == "<=" || 
        expr->op == ">" || expr->op == ">=") {
        
        if (!isCompatible(leftType, rightType)) {
            return Result<Type*>::Err(makeError(
                ErrorKind::TypeError,
                "无法比较类型 " + leftType->toString() + 
                " 和 " + rightType->toString()
            ));
        }
        return Result<Type*>::Ok(boolType_);
    }
    
    // 逻辑运算符
    if (expr->op == "&&" || expr->op == "||") {
        if (!leftType->isBool() || !rightType->isBool()) {
            return Result<Type*>::Err(makeError(
                ErrorKind::TypeError,
                "逻辑运算符需要bool类型"
            ));
        }
        return Result<Type*>::Ok(boolType_);
    }
    
    return Result<Type*>::Err(makeError(
        ErrorKind::SemanticError,
        "未知运算符: " + expr->op
    ));
}

Result<Type*> SemanticAnalyzer::analyzeUnary(UnaryExpr* expr) {
    auto operandResult = analyzeExpr(expr->operand.get());
    if (operandResult.isErr()) {
        return operandResult;
    }
    
    auto* operandType = operandResult.value();
    
    if (expr->op == "-") {
        if (operandType->isInt() || operandType->isFloat()) {
            return Result<Type*>::Ok(operandType);
        }
        return Result<Type*>::Err(makeError(
            ErrorKind::TypeError,
            "一元负号需要数字类型"
        ));
    }
    
    if (expr->op == "!") {
        if (operandType->isBool()) {
            return Result<Type*>::Ok(boolType_);
        }
        return Result<Type*>::Err(makeError(
            ErrorKind::TypeError,
            "逻辑非需要bool类型"
        ));
    }
    
    return Result<Type*>::Err(makeError(
        ErrorKind::SemanticError,
        "未知一元运算符: " + expr->op
    ));
}

Result<Type*> SemanticAnalyzer::analyzeCall(CallExpr* expr) {
    // 获取被调用的函数
    if (expr->callee->kind != ExprKind::Identifier) {
        return Result<Type*>::Err(makeError(
            ErrorKind::SemanticError,
            "只支持直接函数调用"
        ));
    }
    
    auto* identExpr = static_cast<IdentifierExpr*>(expr->callee.get());
    auto* symbol = currentScope_->findSymbol(identExpr->name);
    
    if (!symbol) {
        return Result<Type*>::Err(makeError(
            ErrorKind::SemanticError,
            "未定义的函数: " + identExpr->name
        ));
    }
    
    if (!symbol->isFunction) {
        return Result<Type*>::Err(makeError(
            ErrorKind::SemanticError,
            identExpr->name + " 不是函数"
        ));
    }
    
    auto* funcType = symbol->type;
    
    // 内置函数特殊处理
    if (identExpr->name == "println" || identExpr->name == "print") {
        // 分析参数
        for (auto& arg : expr->arguments) {
            auto argResult = analyzeExpr(arg.get());
            if (argResult.isErr()) {
                return Result<Type*>::Err(argResult.error());
            }
        }
        return Result<Type*>::Ok(voidType_);
    }
    
    // 检查参数数量
    if (expr->arguments.size() != funcType->paramTypes.size()) {
        return Result<Type*>::Err(makeError(
            ErrorKind::SemanticError,
            "参数数量不匹配: 期望 " + 
            std::to_string(funcType->paramTypes.size()) + 
            ", 得到 " + std::to_string(expr->arguments.size())
        ));
    }
    
    // 检查参数类型
    for (size_t i = 0; i < expr->arguments.size(); ++i) {
        auto argResult = analyzeExpr(expr->arguments[i].get());
        if (argResult.isErr()) {
            return argResult;
        }
        
        auto* argType = argResult.value();
        auto* paramType = funcType->paramTypes[i];
        
        if (!isCompatible(paramType, argType)) {
            return Result<Type*>::Err(makeError(
                ErrorKind::TypeError,
                "参数 " + std::to_string(i + 1) + " 类型不匹配: 期望 " + 
                paramType->toString() + ", 得到 " + argType->toString()
            ));
        }
    }
    
    return Result<Type*>::Ok(funcType->returnType);
}

Result<Type*> SemanticAnalyzer::analyzeMember(MemberExpr* expr) {
    auto objectResult = analyzeExpr(expr->object.get());
    if (objectResult.isErr()) {
        return objectResult;
    }
    
    // TODO: 实现结构体成员访问
    return Result<Type*>::Err(makeError(
        ErrorKind::SemanticError,
        "结构体尚未实现"
    ));
}

// 类型检查辅助函数
bool SemanticAnalyzer::isCompatible(Type* t1, Type* t2) {
    if (t1 == t2) return true;
    if (t1->kind == t2->kind) return true;
    
    // int和float可以互相转换
    if ((t1->isInt() && t2->isFloat()) || (t1->isFloat() && t2->isInt())) {
        return true;
    }
    
    return false;
}

Type* SemanticAnalyzer::getCommonType(Type* t1, Type* t2) {
    if (t1 == t2) return t1;
    if (t1->isFloat() || t2->isFloat()) return floatType_;
    if (t1->isInt() && t2->isInt()) return intType_;
    return nullptr;
}

Type* SemanticAnalyzer::getType(const std::string& name) {
    if (name == "void") return voidType_;
    if (name == "int") return intType_;
    if (name == "float") return floatType_;
    if (name == "string") return stringType_;
    if (name == "bool") return boolType_;
    return nullptr;
}

Type* SemanticAnalyzer::createFunctionType(const std::vector<Type*>& params, Type* ret) {
    auto* funcType = new Type(TypeKind::Function);
    funcType->paramTypes = params;
    funcType->returnType = ret;
    types_.emplace_back(funcType);
    return funcType;
}

void SemanticAnalyzer::pushScope() {
    auto* newScope = new SymbolTable(currentScope_);
    scopes_.emplace_back(newScope);
    currentScope_ = newScope;
}

void SemanticAnalyzer::popScope() {
    if (currentScope_->parent()) {
        currentScope_ = currentScope_->parent();
    }
}

Type* SemanticAnalyzer::getExprType(Expr* expr) {
    auto it = exprTypes_.find(expr);
    if (it != exprTypes_.end()) {
        return it->second;
    }
    return nullptr;
}

CompileError SemanticAnalyzer::makeError(ErrorKind kind, const std::string& message) {
    return CompileError{kind, message, 0, 0, filename_};
}

} // namespace frontend
} // namespace az
