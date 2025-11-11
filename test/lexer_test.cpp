// AZ编译器 - 词法分析器测试

#include "AZ/Frontend/Lexer.h"
#include <cassert>
#include <iostream>

using namespace az::frontend;

void testBasicTokens() {
    std::cout << "测试基本token..." << std::endl;
    
    Lexer lexer("fn main() int { return 0; }", "test.az");
    auto result = lexer.tokenize();
    
    assert(result.isOk());
    auto& tokens = result.value();
    
    assert(tokens.size() == 9); // fn main ( ) int { return 0 ; EOF
    assert(tokens[0].type == TokenType::Fn);
    assert(tokens[1].type == TokenType::Identifier);
    assert(tokens[1].lexeme == "main");
    assert(tokens[2].type == TokenType::LeftParen);
    assert(tokens[3].type == TokenType::RightParen);
    assert(tokens[4].type == TokenType::Identifier);
    assert(tokens[4].lexeme == "int");
    assert(tokens[5].type == TokenType::LeftBrace);
    assert(tokens[6].type == TokenType::Return);
    assert(tokens[7].type == TokenType::IntLiteral);
    assert(tokens[7].lexeme == "0");
    assert(tokens[8].type == TokenType::Semicolon);
    
    std::cout << "  通过!" << std::endl;
}

void testOperators() {
    std::cout << "测试运算符..." << std::endl;
    
    Lexer lexer("+ - * / % == != < <= > >= && ||", "test.az");
    auto result = lexer.tokenize();
    
    assert(result.isOk());
    auto& tokens = result.value();
    
    assert(tokens[0].type == TokenType::Plus);
    assert(tokens[1].type == TokenType::Minus);
    assert(tokens[2].type == TokenType::Star);
    assert(tokens[3].type == TokenType::Slash);
    assert(tokens[4].type == TokenType::Percent);
    assert(tokens[5].type == TokenType::EqualEqual);
    assert(tokens[6].type == TokenType::BangEqual);
    assert(tokens[7].type == TokenType::Less);
    assert(tokens[8].type == TokenType::LessEqual);
    assert(tokens[9].type == TokenType::Greater);
    assert(tokens[10].type == TokenType::GreaterEqual);
    assert(tokens[11].type == TokenType::AmpAmp);
    assert(tokens[12].type == TokenType::PipePipe);
    
    std::cout << "  通过!" << std::endl;
}

void testStringLiterals() {
    std::cout << "测试字符串字面量..." << std::endl;
    
    Lexer lexer(R"("Hello, World!" "测试中文")", "test.az");
    auto result = lexer.tokenize();
    
    assert(result.isOk());
    auto& tokens = result.value();
    
    assert(tokens[0].type == TokenType::StringLiteral);
    assert(tokens[0].lexeme == "Hello, World!");
    assert(tokens[1].type == TokenType::StringLiteral);
    assert(tokens[1].lexeme == "测试中文");
    
    std::cout << "  通过!" << std::endl;
}

void testComments() {
    std::cout << "测试注释..." << std::endl;
    
    Lexer lexer("fn main() // 这是注释\n{ return 0; }", "test.az");
    auto result = lexer.tokenize();
    
    assert(result.isOk());
    auto& tokens = result.value();
    
    // 注释应该被忽略
    assert(tokens[0].type == TokenType::Fn);
    assert(tokens[1].type == TokenType::Identifier);
    assert(tokens[2].type == TokenType::LeftParen);
    assert(tokens[3].type == TokenType::RightParen);
    assert(tokens[4].type == TokenType::LeftBrace);
    
    std::cout << "  通过!" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "运行词法分析器测试..." << std::endl;
    std::cout << std::endl;
    
    testBasicTokens();
    testOperators();
    testStringLiterals();
    testComments();
    
    std::cout << std::endl;
    std::cout << "所有测试通过!" << std::endl;
    
    return 0;
}
