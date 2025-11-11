// AZ编译器 - 语义分析器测试

#include "AZ/Frontend/Lexer.h"
#include "AZ/Frontend/Parser.h"
#include "AZ/Frontend/Sema.h"
#include <cassert>
#include <iostream>

using namespace az::frontend;

void testTypeChecking() {
    std::cout << "测试类型检查..." << std::endl;
    
    const char* source = R"(
        fn main() int {
            let x: int = 10;
            let y: int = 20;
            let sum: int = x + y;
            return sum;
        }
    )";
    
    Lexer lexer(source, "test.az");
    auto tokensResult = lexer.tokenize();
    assert(tokensResult.isOk());
    
    Parser parser(std::move(tokensResult.value()), "test.az");
    auto astResult = parser.parse();
    assert(astResult.isOk());
    
    SemanticAnalyzer sema;
    auto semaResult = sema.analyze(astResult.value().get());
    assert(semaResult.isOk());
    
    std::cout << "  通过!" << std::endl;
}

void testTypeInference() {
    std::cout << "测试类型推导..." << std::endl;
    
    const char* source = R"(
        fn main() int {
            let x = 10;
            let y = 20;
            let sum = x + y;
            return sum;
        }
    )";
    
    Lexer lexer(source, "test.az");
    auto tokensResult = lexer.tokenize();
    assert(tokensResult.isOk());
    
    Parser parser(std::move(tokensResult.value()), "test.az");
    auto astResult = parser.parse();
    assert(astResult.isOk());
    
    SemanticAnalyzer sema;
    auto semaResult = sema.analyze(astResult.value().get());
    assert(semaResult.isOk());
    
    std::cout << "  通过!" << std::endl;
}

void testFunctionCall() {
    std::cout << "测试函数调用..." << std::endl;
    
    const char* source = R"(
        fn add(a: int, b: int) int {
            return a + b;
        }
        
        fn main() int {
            let result = add(10, 20);
            return result;
        }
    )";
    
    Lexer lexer(source, "test.az");
    auto tokensResult = lexer.tokenize();
    assert(tokensResult.isOk());
    
    Parser parser(std::move(tokensResult.value()), "test.az");
    auto astResult = parser.parse();
    assert(astResult.isOk());
    
    SemanticAnalyzer sema;
    auto semaResult = sema.analyze(astResult.value().get());
    assert(semaResult.isOk());
    
    std::cout << "  通过!" << std::endl;
}

void testTypeError() {
    std::cout << "测试类型错误检测..." << std::endl;
    
    const char* source = R"(
        fn main() int {
            let x: int = "hello";
            return 0;
        }
    )";
    
    Lexer lexer(source, "test.az");
    auto tokensResult = lexer.tokenize();
    assert(tokensResult.isOk());
    
    Parser parser(std::move(tokensResult.value()), "test.az");
    auto astResult = parser.parse();
    assert(astResult.isOk());
    
    SemanticAnalyzer sema;
    auto semaResult = sema.analyze(astResult.value().get());
    assert(semaResult.isErr()); // 应该报错
    
    std::cout << "  通过!" << std::endl;
}

void testUndefinedVariable() {
    std::cout << "测试未定义变量检测..." << std::endl;
    
    const char* source = R"(
        fn main() int {
            return x;
        }
    )";
    
    Lexer lexer(source, "test.az");
    auto tokensResult = lexer.tokenize();
    assert(tokensResult.isOk());
    
    Parser parser(std::move(tokensResult.value()), "test.az");
    auto astResult = parser.parse();
    assert(astResult.isOk());
    
    SemanticAnalyzer sema;
    auto semaResult = sema.analyze(astResult.value().get());
    assert(semaResult.isErr()); // 应该报错
    
    std::cout << "  通过!" << std::endl;
}

void testReturnTypeCheck() {
    std::cout << "测试返回类型检查..." << std::endl;
    
    const char* source = R"(
        fn getNumber() int {
            return 42;
        }
        
        fn main() int {
            return getNumber();
        }
    )";
    
    Lexer lexer(source, "test.az");
    auto tokensResult = lexer.tokenize();
    assert(tokensResult.isOk());
    
    Parser parser(std::move(tokensResult.value()), "test.az");
    auto astResult = parser.parse();
    assert(astResult.isOk());
    
    SemanticAnalyzer sema;
    auto semaResult = sema.analyze(astResult.value().get());
    assert(semaResult.isOk());
    
    std::cout << "  通过!" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "运行语义分析器测试..." << std::endl;
    std::cout << std::endl;
    
    testTypeChecking();
    testTypeInference();
    testFunctionCall();
    testTypeError();
    testUndefinedVariable();
    testReturnTypeCheck();
    
    std::cout << std::endl;
    std::cout << "所有测试通过!" << std::endl;
    
    return 0;
}
