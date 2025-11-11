// AZ编译器 - 语法分析器头文件

#ifndef AZ_FRONTEND_PARSER_H
#define AZ_FRONTEND_PARSER_H

#include "AZ/Frontend/Token.h"
#include "AZ/Frontend/AST.h"
#include "AZ/Support/Result.h"
#include "AZ/Frontend/Error.h"

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

namespace az {
namespace frontend {

class Parser {
public:
    Parser(std::vector<Token> tokens, std::string filename);
    
    /// 解析整个程序
    Result<std::unique_ptr<Program>> parse();
    
private:
    std::vector<Token> tokens_;
    std::string filename_;
    size_t current_;
    
    // 解析语句
    Result<std::unique_ptr<Stmt>> parseStatement();
    Result<std::unique_ptr<Stmt>> parseImport();
    Result<std::unique_ptr<Stmt>> parseFunction();
    Result<std::unique_ptr<Stmt>> parseVarDeclaration();
    Result<std::unique_ptr<Stmt>> parseReturn();
    Result<std::unique_ptr<Stmt>> parseIf();
    Result<std::unique_ptr<Stmt>> parseWhile();
    Result<std::unique_ptr<Stmt>> parseFor();
    Result<std::unique_ptr<Stmt>> parseMatch();
    Result<std::unique_ptr<Stmt>> parseBlock();
    Result<std::unique_ptr<Stmt>> parseExpressionStatement();
    Result<std::unique_ptr<Stmt>> parseComptime();
    
    // 新增：解析结构体和枚举
    Result<std::unique_ptr<Stmt>> parseStruct();
    Result<std::unique_ptr<Stmt>> parseEnum();
    
    // 解析表达式
    Result<std::unique_ptr<Expr>> parseExpression();
    Result<std::unique_ptr<Expr>> parseLogicalOr();
    Result<std::unique_ptr<Expr>> parseLogicalAnd();
    Result<std::unique_ptr<Expr>> parseEquality();
    Result<std::unique_ptr<Expr>> parseComparison();
    Result<std::unique_ptr<Expr>> parseTerm();
    Result<std::unique_ptr<Expr>> parseFactor();
    Result<std::unique_ptr<Expr>> parseUnary();
    Result<std::unique_ptr<Expr>> parsePrimary();
    Result<std::unique_ptr<Expr>> parseCall(std::unique_ptr<Expr> expr);
    Result<std::unique_ptr<Expr>> parseIndex(std::unique_ptr<Expr> expr);
    Result<std::unique_ptr<Expr>> parseMember(std::unique_ptr<Expr> expr);
    
    // 解析模式匹配
    Result<std::unique_ptr<Pattern>> parsePattern();
    Result<std::unique_ptr<Pattern>> parseOrPattern();
    Result<std::unique_ptr<Pattern>> parsePrimaryPattern();
    
    // 工具函数
    bool match(TokenType type);
    bool match(std::initializer_list<TokenType> types);
    bool check(TokenType type) const;
    const Token& advance();
    const Token& peek() const;
    const Token& previous() const;
    bool isAtEnd() const;
    
    Result<Token> consume(TokenType type, const std::string& message);
    CompileError error(const Token& token, const std::string& message);
};

} // namespace frontend
} // namespace az

#endif // AZ_FRONTEND_PARSER_H