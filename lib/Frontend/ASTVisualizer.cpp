// AZ编译器 - AST可视化器实现

#include "AZ/Frontend/ASTVisualizer.h"
#include <iostream>
#include <map>

namespace az {
namespace frontend {

ASTVisualizer::ASTVisualizer() {
    // 构造函数
}

ASTVisualizer::~ASTVisualizer() {
    // 析构函数
}

bool ASTVisualizer::visualizeAST(const Program* program, const std::string& outputFile) {
    std::ofstream out(outputFile);
    if (!out.is_open()) {
        return false;
    }
    
    writeGraphHeader(out, "AST");
    
    int nodeId = 0;
    
    // 遍历所有语句
    for (const auto& stmt : program->statements) {
        visualizeStmt(stmt.get(), out, nodeId);
    }
    
    writeGraphFooter(out);
    out.close();
    
    return true;
}

bool ASTVisualizer::visualizeCallGraph(const Program* program, const std::string& outputFile) {
    std::ofstream out(outputFile);
    if (!out.is_open()) {
        return false;
    }
    
    writeGraphHeader(out, "CallGraph");
    
    // 这里应该实现函数调用图的生成逻辑
    // 为简化起见，我们只添加一个示例节点
    
    out << "  main [label=\"main\"];\n";
    out << "  func1 [label=\"function1\"];\n";
    out << "  main -> func1 [label=\"calls\"];\n";
    
    writeGraphFooter(out);
    out.close();
    
    return true;
}

bool ASTVisualizer::visualizeControlFlow(const Stmt* stmt, const std::string& outputFile) {
    std::ofstream out(outputFile);
    if (!out.is_open()) {
        return false;
    }
    
    writeGraphHeader(out, "ControlFlow");
    
    // 这里应该实现控制流图的生成逻辑
    // 为简化起见，我们只添加一个示例节点
    
    out << "  start [label=\"Start\"];\n";
    out << "  end [label=\"End\"];\n";
    out << "  start -> end [label=\"flow\"];\n";
    
    writeGraphFooter(out);
    out.close();
    
    return true;
}

void ASTVisualizer::visualizeExpr(const Expr* expr, std::ofstream& out, int& nodeId) {
    if (!expr) return;
    
    int currentNodeId = nodeId++;
    std::string currentNode = generateNodeId(currentNodeId);
    
    // 创建当前节点
    out << "  " << currentNode << " [label=\"" << getNodeLabel(expr) << "\"];\n";
    
    // 递归处理子节点
    switch (expr->kind) {
        case ExprKind::Binary:
            if (expr->left) {
                int leftNodeId = nodeId;
                visualizeExpr(expr->left.get(), out, nodeId);
                out << "  " << currentNode << " -> " << generateNodeId(leftNodeId) << " [label=\"left\"];\n";
            }
            if (expr->right) {
                int rightNodeId = nodeId;
                visualizeExpr(expr->right.get(), out, nodeId);
                out << "  " << currentNode << " -> " << generateNodeId(rightNodeId) << " [label=\"right\"];\n";
            }
            break;
            
        case ExprKind::Unary:
            if (expr->operand) {
                int operandNodeId = nodeId;
                visualizeExpr(expr->operand.get(), out, nodeId);
                out << "  " << currentNode << " -> " << generateNodeId(operandNodeId) << " [label=\"operand\"];\n";
            }
            break;
            
        case ExprKind::Call:
            if (expr->callee) {
                int calleeNodeId = nodeId;
                visualizeExpr(expr->callee.get(), out, nodeId);
                out << "  " << currentNode << " -> " << generateNodeId(calleeNodeId) << " [label=\"callee\"];\n";
            }
            for (const auto& arg : expr->arguments) {
                if (arg) {
                    int argNodeId = nodeId;
                    visualizeExpr(arg.get(), out, nodeId);
                    out << "  " << currentNode << " -> " << generateNodeId(argNodeId) << " [label=\"arg\"];\n";
                }
            }
            break;
            
        default:
            // 其他表达式类型没有子节点
            break;
    }
}

void ASTVisualizer::visualizeStmt(const Stmt* stmt, std::ofstream& out, int& nodeId) {
    if (!stmt) return;
    
    int currentNodeId = nodeId++;
    std::string currentNode = generateNodeId(currentNodeId);
    
    // 创建当前节点
    out << "  " << currentNode << " [label=\"" << getNodeLabel(stmt) << "\"];\n";
    
    // 递归处理子节点
    switch (stmt->kind) {
        case StmtKind::Block:
            for (const auto& s : stmt->statements) {
                if (s) {
                    int childNodeId = nodeId;
                    visualizeStmt(s.get(), out, nodeId);
                    out << "  " << currentNode << " -> " << generateNodeId(childNodeId) << ";\n";
                }
            }
            break;
            
        case StmtKind::If:
            if (stmt->condition) {
                int conditionNodeId = nodeId;
                visualizeExpr(stmt->condition.get(), out, nodeId);
                out << "  " << currentNode << " -> " << generateNodeId(conditionNodeId) << " [label=\"condition\"];\n";
            }
            if (stmt->then_branch) {
                int thenNodeId = nodeId;
                visualizeStmt(stmt->then_branch.get(), out, nodeId);
                out << "  " << currentNode << " -> " << generateNodeId(thenNodeId) << " [label=\"then\"];\n";
            }
            if (stmt->else_branch) {
                int elseNodeId = nodeId;
                visualizeStmt(stmt->else_branch.get(), out, nodeId);
                out << "  " << currentNode << " -> " << generateNodeId(elseNodeId) << " [label=\"else\"];\n";
            }
            break;
            
        case StmtKind::While:
            if (stmt->condition) {
                int conditionNodeId = nodeId;
                visualizeExpr(stmt->condition.get(), out, nodeId);
                out << "  " << currentNode << " -> " << generateNodeId(conditionNodeId) << " [label=\"condition\"];\n";
            }
            if (stmt->body) {
                int bodyNodeId = nodeId;
                visualizeStmt(stmt->body.get(), out, nodeId);
                out << "  " << currentNode << " -> " << generateNodeId(bodyNodeId) << " [label=\"body\"];\n";
            }
            break;
            
        case StmtKind::FuncDecl:
            if (stmt->body) {
                int bodyNodeId = nodeId;
                visualizeStmt(stmt->body.get(), out, nodeId);
                out << "  " << currentNode << " -> " << generateNodeId(bodyNodeId) << " [label=\"body\"];\n";
            }
            break;
            
        default:
            // 其他语句类型处理
            if (stmt->expr) {
                int exprNodeId = nodeId;
                visualizeExpr(stmt->expr.get(), out, nodeId);
                out << "  " << currentNode << " -> " << generateNodeId(exprNodeId) << " [label=\"expr\"];\n";
            }
            break;
    }
}

std::string ASTVisualizer::getNodeLabel(const Expr* expr) {
    if (!expr) return "null";
    
    switch (expr->kind) {
        case ExprKind::IntLiteral:
            return "Int: " + std::to_string(expr->int_value);
        case ExprKind::FloatLiteral:
            return "Float: " + std::to_string(expr->float_value);
        case ExprKind::StringLiteral:
            return "String: " + expr->string_value;
        case ExprKind::BoolLiteral:
            return "Bool: " + std::string(expr->bool_value ? "true" : "false");
        case ExprKind::Identifier:
            return "Id: " + expr->name;
        case ExprKind::Binary:
            return "Binary: " + expr->op;
        case ExprKind::Unary:
            return "Unary: " + expr->op;
        case ExprKind::Call:
            return "Call";
        default:
            return "Expr";
    }
}

std::string ASTVisualizer::getNodeLabel(const Stmt* stmt) {
    if (!stmt) return "null";
    
    switch (stmt->kind) {
        case StmtKind::Expression:
            return "ExpressionStmt";
        case StmtKind::VarDecl:
            return "VarDecl: " + stmt->name;
        case StmtKind::FuncDecl:
            return "FuncDecl: " + stmt->name;
        case StmtKind::Return:
            return "Return";
        case StmtKind::If:
            return "If";
        case StmtKind::While:
            return "While";
        case StmtKind::Block:
            return "Block";
        default:
            return "Stmt";
    }
}

std::string ASTVisualizer::generateNodeId(int id) {
    return "node" + std::to_string(id);
}

void ASTVisualizer::writeGraphHeader(std::ofstream& out, const std::string& graphName) {
    out << "digraph " << graphName << " {\n";
    out << "  node [shape=box, style=rounded];\n";
    out << "  rankdir=TB;\n";
}

void ASTVisualizer::writeGraphFooter(std::ofstream& out) {
    out << "}\n";
}

} // namespace frontend
} // namespace az