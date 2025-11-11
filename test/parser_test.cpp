// AZ编译器 - 语法分析器测试

#include "AZ/Frontend/Lexer.h"
#include "AZ/Frontend/Parser.h"
#include <cassert>
#include <iostream>

using namespace az::frontend;

void testFunctionDeclaration() {
    std::cout << "测试函数声明..." << std::endl;
    
    const char* source = R"(
        fn add(a: int, b: int) int {
            return a + b;
        }
    )";
    
    Lexer lexer(source, "test.az");
    auto tokensResult = lexer.tokenize();
    assert(tokensResult.isOk());
    
    Parser parser(std::move(tokensResult.value()), "test.az");
    auto astResult = parser.parse();
    assert(astResult.isOk());
    
    auto& program = astResult.value();
    assert(program->statements.size() == 1);
    
    auto* funcDecl = dynamic_cast<FuncDeclStmt*>(program->statements[0].get());
    assert(funcDecl != nullptr);
    assert(funcDecl->name == "add");
    assert(funcDecl->params.size() == 2);
    assert(funcDecl->params[0].name == "a");
    assert(funcDecl->params[0].type == "int");
    assert(funcDecl->returnType == "int");
    
    std::cout << "  通过!" << std::endl;
}

void testVariableDeclaration() {
    std::cout << "测试变量声明..." << std::endl;
    
    const char* source = R"(
        let x = 10;
        var y: int = 20;
    )";
    
    Lexer lexer(source, "test.az");
    auto tokensResult = lexer.tokenize();
    assert(tokensResult.isOk());
    
    Parser parser(std::move(tokensResult.value()), "test.az");
    auto astResult = parser.parse();
    assert(astResult.isOk());
    
    auto& program = astResult.value();
    assert(program->statements.size() == 2);
    
    auto* varDecl1 = dynamic_cast<VarDeclStmt*>(program->statements[0].get());
    assert(varDecl1 != nullptr);
    assert(varDecl1->name == "x");
    assert(!varDecl1->isMutable);
    
    auto* varDecl2 = dynamic_cast<VarDeclStmt*>(program->statements[1].get());
    assert(varDecl2 != nullptr);
    assert(varDecl2->name == "y");
    assert(varDecl2->isMutable);
    assert(varDecl2->type == "int");
    
    std::cout << "  通过!" << std::endl;
}

void testBinaryExpression() {
    std::cout << "测试二元表达式..." << std::endl;
    
    const char* source = "let result = 1 + 2 * 3;";
    
    Lexer lexer(source, "test.az");
    auto tokensResult = lexer.tokenize();
    assert(tokensResult.isOk());
    
    Parser parser(std::move(tokensResult.value()), "test.az");
    auto astResult = parser.parse();
    assert(astResult.isOk());
    
    auto& program = astResult.value();
    assert(program->statements.size() == 1);
    
    auto* varDecl = dynamic_cast<VarDeclStmt*>(program->statements[0].get());
    assert(varDecl != nullptr);
    assert(varDecl->initializer != nullptr);
    
    // 检查运算符优先级: 1 + (2 * 3)
    auto* addExpr = dynamic_cast<BinaryExpr*>(varDecl->initializer.get());
    assert(addExpr != nullptr);
    assert(addExpr->op == "+");
    
    auto* mulExpr = dynamic_cast<BinaryExpr*>(addExpr->right.get());
    assert(mulExpr != nullptr);
    assert(mulExpr->op == "*");
    
    std::cout << "  通过!" << std::endl;
}

void testIfStatement() {
    std::cout << "测试if语句..." << std::endl;
    
    const char* source = R"(
        if (x > 0) {
            return 1;
        } else {
            return 0;
        }
    )";
    
    Lexer lexer(source, "test.az");
    auto tokensResult = lexer.tokenize();
    assert(tokensResult.isOk());
    
    Parser parser(std::move(tokensResult.value()), "test.az");
    auto astResult = parser.parse();
    assert(astResult.isOk());
    
    auto& program = astResult.value();
    assert(program->statements.size() == 1);
    
    auto* ifStmt = dynamic_cast<IfStmt*>(program->statements[0].get());
    assert(ifStmt != nullptr);
    assert(ifStmt->condition != nullptr);
    assert(ifStmt->thenBranch != nullptr);
    assert(ifStmt->elseBranch != nullptr);
    
    std::cout << "  通过!" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "运行语法分析器测试..." << std::endl;
    std::cout << std::endl;
    
    testFunctionDeclaration();
    testVariableDeclaration();
    testBinaryExpression();
    testIfStatement();
    
    std::cout << std::endl;
    std::cout << "所有测试通过!" << std::endl;
    
    return 0;
}
