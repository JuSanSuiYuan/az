// AZ编译器 - 语法分析器实现

#include "AZ/Frontend/Parser.h"
#include <sstream>

namespace az {
namespace frontend {

Parser::Parser(std::vector<Token> tokens, std::string filename)
    : tokens_(std::move(tokens)), filename_(std::move(filename)), current_(0) {}

Result<std::unique_ptr<Program>> Parser::parse() {
    std::vector<std::unique_ptr<Stmt>> statements;
    
    while (!isAtEnd()) {
        auto stmt = parseStatement();
        if (stmt.isErr()) {
            return Result<std::unique_ptr<Program>>::Err(stmt.error());
        }
        statements.push_back(std::move(stmt.value()));
    }
    
    return Result<std::unique_ptr<Program>>::Ok(
        std::make_unique<Program>(std::move(statements))
    );
}

Result<std::unique_ptr<Stmt>> Parser::parseStatement() {
    // import语句
    if (match(TokenType::Import)) {
        return parseImport();
    }
    
    // 编译时执行语句
    if (match(TokenType::Comptime)) {
        return parseComptime();
    }
    
    // 函数声明
    if (match(TokenType::Fn)) {
        return parseFunction();
    }
    
    // 结构体声明
    bool isPublic = match(TokenType::Pub);
    if (match(TokenType::Struct)) {
        auto result = parseStruct();
        // 如果是public的结构体声明，设置isPublic标志
        if (result.isOk() && isPublic) {
            result.value()->is_public = true;
        }
        return result;
    }
    // 如果之前匹配了pub但不是struct，则回退
    if (isPublic && !check(TokenType::Struct)) {
        current_--; // 回退一个token
    }
    
    // 枚举声明
    isPublic = match(TokenType::Pub); // 重新检查pub
    if (match(TokenType::Enum)) {
        auto result = parseEnum();
        // 如果是public的枚举声明，设置isPublic标志
        if (result.isOk() && isPublic) {
            result.value()->is_public = true;
        }
        return result;
    }
    // 如果之前匹配了pub但不是enum，则回退
    if (isPublic && !check(TokenType::Enum)) {
        current_--; // 回退一个token
    }
    
    // 变量声明
    if (check(TokenType::Let) || check(TokenType::Var)) {
        return parseVarDeclaration();
    }
    
    // return语句
    if (match(TokenType::Return)) {
        return parseReturn();
    }
    
    // if语句
    if (match(TokenType::If)) {
        return parseIf();
    }
    
    // while语句
    if (match(TokenType::While)) {
        return parseWhile();
    }
    
    // for语句
    if (match(TokenType::For)) {
        return parseFor();
    }
    
    // match语句
    if (match(TokenType::Match)) {
        return parseMatch();
    }
    
    // 代码块
    if (check(TokenType::LeftBrace)) {
        return parseBlock();
    }
    
    // 表达式语句
    return parseExpressionStatement();
}

Result<std::unique_ptr<Stmt>> Parser::parseImport() {
    std::vector<std::string> parts;
    
    auto name = consume(TokenType::Identifier, "期望模块名");
    if (name.isErr()) {
        return Result<std::unique_ptr<Stmt>>::Err(name.error());
    }
    parts.push_back(name.value().lexeme);
    
    while (match(TokenType::Dot)) {
        auto part = consume(TokenType::Identifier, "期望模块名");
        if (part.isErr()) {
            return Result<std::unique_ptr<Stmt>>::Err(part.error());
        }
        parts.push_back(part.value().lexeme);
    }
    
    auto semi = consume(TokenType::Semicolon, "期望 ';'");
    if (semi.isErr()) {
        return Result<std::unique_ptr<Stmt>>::Err(semi.error());
    }
    
    std::string path;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) path += ".";
        path += parts[i];
    }
    
    return Result<std::unique_ptr<Stmt>>::Ok(
        std::make_unique<ImportStmt>(path)
    );
}

Result<std::unique_ptr<Stmt>> Parser::parseComptime() {
    auto body = parseStatement();
    if (body.isErr()) {
        return Result<std::unique_ptr<Stmt>>::Err(body.error());
    }
    
    return Result<std::unique_ptr<Stmt>>::Ok(
        std::make_unique<ComptimeStmt>(std::move(body.value()))
    );
}

Result<std::unique_ptr<Stmt>> Parser::parseFunction() {
    auto name = consume(TokenType::Identifier, "期望函数名");
    if (name.isErr()) {
        return Result<std::unique_ptr<Stmt>>::Err(name.error());
    }
    
    auto lparen = consume(TokenType::LeftParen, "期望 '('");
    if (lparen.isErr()) {
        return Result<std::unique_ptr<Stmt>>::Err(lparen.error());
    }
    
    std::vector<Param> params;
    if (!check(TokenType::RightParen)) {
        do {
            auto paramName = consume(TokenType::Identifier, "期望参数名");
            if (paramName.isErr()) {
                return Result<std::unique_ptr<Stmt>>::Err(paramName.error());
            }
            
            auto colon = consume(TokenType::Colon, "期望 ':'");
            if (colon.isErr()) {
                return Result<std::unique_ptr<Stmt>>::Err(colon.error());
            }
            
            auto paramType = consume(TokenType::Identifier, "期望类型名");
            if (paramType.isErr()) {
                return Result<std::unique_ptr<Stmt>>::Err(paramType.error());
            }
            
            params.emplace_back(paramName.value().lexeme, paramType.value().lexeme);
        } while (match(TokenType::Comma));
    }
    
    auto rparen = consume(TokenType::RightParen, "期望 ')'");
    if (rparen.isErr()) {
        return Result<std::unique_ptr<Stmt>>::Err(rparen.error());
    }
    
    std::string returnType = "void";
    if (!check(TokenType::LeftBrace)) {
        auto type = consume(TokenType::Identifier, "期望返回类型");
        if (type.isErr()) {
            return Result<std::unique_ptr<Stmt>>::Err(type.error());
        }
        returnType = type.value().lexeme;
    }
    
    auto body = parseBlock();
    if (body.isErr()) {
        return Result<std::unique_ptr<Stmt>>::Err(body.error());
    }
    
    return Result<std::unique_ptr<Stmt>>::Ok(
        std::make_unique<FuncDeclStmt>(
            name.value().lexeme,
            std::move(params),
            returnType,
            std::move(body.value())
        )
    );
}

Result<std::unique_ptr<Stmt>> Parser::parseStruct() {
    auto name = consume(TokenType::Identifier, "期望结构体名");
    if (name.isErr()) {
        return Result<std::unique_ptr<Stmt>>::Err(name.error());
    }
    
    auto lbrace = consume(TokenType::LeftBrace, "期望 '{'");
    if (lbrace.isErr()) {
        return Result<std::unique_ptr<Stmt>>::Err(lbrace.error());
    }
    
    std::vector<StructField> fields;
    while (!check(TokenType::RightBrace) && !isAtEnd()) {
        bool fieldPublic = match(TokenType::Pub);
        
        auto fieldName = consume(TokenType::Identifier, "期望字段名");
        if (fieldName.isErr()) {
            return Result<std::unique_ptr<Stmt>>::Err(fieldName.error());
        }
        
        auto colon = consume(TokenType::Colon, "期望 ':'");
        if (colon.isErr()) {
            return Result<std::unique_ptr<Stmt>>::Err(colon.error());
        }
        
        auto fieldType = consume(TokenType::Identifier, "期望字段类型");
        if (fieldType.isErr()) {
            return Result<std::unique_ptr<Stmt>>::Err(fieldType.error());
        }
        
        fields.emplace_back(StructField{
            fieldName.value().lexeme,
            fieldType.value().lexeme,
            fieldPublic
        });
        
        // 可选的分号
        match(TokenType::Semicolon);
    }
    
    auto rbrace = consume(TokenType::RightBrace, "期望 '}'");
    if (rbrace.isErr()) {
        return Result<std::unique_ptr<Stmt>>::Err(rbrace.error());
    }
    
    // 检查是否有pub关键字
    bool isPublic = false;
    // 这里需要根据调用上下文确定是否是public的
    
    return Result<std::unique_ptr<Stmt>>::Ok(
        make_struct_decl(name.value().lexeme, std::move(fields), isPublic)
    );
}

Result<std::unique_ptr<Stmt>> Parser::parseEnum() {
    auto name = consume(TokenType::Identifier, "期望枚举名");
    if (name.isErr()) {
        return Result<std::unique_ptr<Stmt>>::Err(name.error());
    }
    
    auto lbrace = consume(TokenType::LeftBrace, "期望 '{'");
    if (lbrace.isErr()) {
        return Result<std::unique_ptr<Stmt>>::Err(lbrace.error());
    }
    
    std::vector<EnumVariant> variants;
    while (!check(TokenType::RightBrace) && !isAtEnd()) {
        auto variantName = consume(TokenType::Identifier, "期望枚举变体名");
        if (variantName.isErr()) {
            return Result<std::unique_ptr<Stmt>>::Err(variantName.error());
        }
        
        std::unique_ptr<Expr> value;
        if (match(TokenType::Equal)) {
            auto expr = parseExpression();
            if (expr.isErr()) {
                return Result<std::unique_ptr<Stmt>>::Err(expr.error());
            }
            value = std::move(expr.value());
        }
        
        variants.emplace_back(EnumVariant{
            variantName.value().lexeme,
            std::move(value)
        });
        
        // 可选的分号或逗号
        match({TokenType::Semicolon, TokenType::Comma});
    }
    
    auto rbrace = consume(TokenType::RightBrace, "期望 '}'");
    if (rbrace.isErr()) {
        return Result<std::unique_ptr<Stmt>>::Err(rbrace.error());
    }
    
    // 检查是否有pub关键字
    bool isPublic = false;
    // 这里需要根据调用上下文确定是否是public的
    
    return Result<std::unique_ptr<Stmt>>::Ok(
        make_enum_decl(name.value().lexeme, std::move(variants), isPublic)
    );
}

Result<std::unique_ptr<Stmt>> Parser::parseVarDeclaration() {
    bool isMutable = match(TokenType::Var);
    if (!isMutable) {
        advance(); // consume 'let'
    }
    
    auto name = consume(TokenType::Identifier, "期望变量名");
    if (name.isErr()) {
        return Result<std::unique_ptr<Stmt>>::Err(name.error());
    }
    
    std::string type;
    if (match(TokenType::Colon)) {
        auto typeToken = consume(TokenType::Identifier, "期望类型名");
        if (typeToken.isErr()) {
            return Result<std::unique_ptr<Stmt>>::Err(typeToken.error());
        }
        type = typeToken.value().lexeme;
    }
    
    std::unique_ptr<Expr> initializer;
    if (match(TokenType::Equal)) {
        auto expr = parseExpression();
        if (expr.isErr()) {
            return Result<std::unique_ptr<Stmt>>::Err(expr.error());
        }
        initializer = std::move(expr.value());
    }
    
    auto semi = consume(TokenType::Semicolon, "期望 ';'");
    if (semi.isErr()) {
        return Result<std::unique_ptr<Stmt>>::Err(semi.error());
    }
    
    return Result<std::unique_ptr<Stmt>>::Ok(
        std::make_unique<VarDeclStmt>(
            name.value().lexeme,
            type,
            isMutable,
            std::move(initializer)
        )
    );
}

Result<std::unique_ptr<Stmt>> Parser::parseReturn() {
    std::unique_ptr<Expr> expr;
    if (!check(TokenType::Semicolon)) {
        auto exprResult = parseExpression();
        if (exprResult.isErr()) {
            return Result<std::unique_ptr<Stmt>>::Err(exprResult.error());
        }
        expr = std::move(exprResult.value());
    }
    
    auto semi = consume(TokenType::Semicolon, "期望 ';'");
    if (semi.isErr()) {
        return Result<std::unique_ptr<Stmt>>::Err(semi.error());
    }
    
    return Result<std::unique_ptr<Stmt>>::Ok(
        std::make_unique<ReturnStmt>(std::move(expr))
    );
}

Result<std::unique_ptr<Stmt>> Parser::parseIf() {
    auto lparen = consume(TokenType::LeftParen, "期望 '('");
    if (lparen.isErr()) {
        return Result<std::unique_ptr<Stmt>>::Err(lparen.error());
    }
    
    auto condition = parseExpression();
    if (condition.isErr()) {
        return Result<std::unique_ptr<Stmt>>::Err(condition.error());
    }
    
    auto rparen = consume(TokenType::RightParen, "期望 ')'");
    if (rparen.isErr()) {
        return Result<std::unique_ptr<Stmt>>::Err(rparen.error());
    }
    
    auto thenBranch = parseStatement();
    if (thenBranch.isErr()) {
        return Result<std::unique_ptr<Stmt>>::Err(thenBranch.error());
    }
    
    std::unique_ptr<Stmt> elseBranch;
    if (match(TokenType::Else)) {
        auto elseResult = parseStatement();
        if (elseResult.isErr()) {
            return Result<std::unique_ptr<Stmt>>::Err(elseResult.error());
        }
        elseBranch = std::move(elseResult.value());
    }
    
    return Result<std::unique_ptr<Stmt>>::Ok(
        std::make_unique<IfStmt>(
            std::move(condition.value()),
            std::move(thenBranch.value()),
            std::move(elseBranch)
        )
    );
}

Result<std::unique_ptr<Stmt>> Parser::parseWhile() {
    auto lparen = consume(TokenType::LeftParen, "期望 '('");
    if (lparen.isErr()) {
        return Result<std::unique_ptr<Stmt>>::Err(lparen.error());
    }
    
    auto condition = parseExpression();
    if (condition.isErr()) {
        return Result<std::unique_ptr<Stmt>>::Err(condition.error());
    }
    
    auto rparen = consume(TokenType::RightParen, "期望 ')'");
    if (rparen.isErr()) {
        return Result<std::unique_ptr<Stmt>>::Err(rparen.error());
    }
    
    auto body = parseStatement();
    if (body.isErr()) {
        return Result<std::unique_ptr<Stmt>>::Err(body.error());
    }
    
    return Result<std::unique_ptr<Stmt>>::Ok(
        std::make_unique<WhileStmt>(
            std::move(condition.value()),
            std::move(body.value())
        )
    );
}

Result<std::unique_ptr<Stmt>> Parser::parseFor() {
    auto lparen = consume(TokenType::LeftParen, "期望 '('");
    if (lparen.isErr()) {
        return Result<std::unique_ptr<Stmt>>::Err(lparen.error());
    }
    
    // 解析初始化部分
    std::unique_ptr<Stmt> initializer;
    if (!check(TokenType::Semicolon)) {
        if (check(TokenType::Let) || check(TokenType::Var)) {
            auto varDecl = parseVarDeclaration();
            if (varDecl.isErr()) {
                return Result<std::unique_ptr<Stmt>>::Err(varDecl.error());
            }
            initializer = std::move(varDecl.value());
        } else {
            auto expr = parseExpression();
            if (expr.isErr()) {
                return Result<std::unique_ptr<Stmt>>::Err(expr.error());
            }
            auto semi = consume(TokenType::Semicolon, "期望 ';'");
            if (semi.isErr()) {
                return Result<std::unique_ptr<Stmt>>::Err(semi.error());
            }
            initializer = std::make_unique<ExprStmt>(std::move(expr.value()));
        }
    } else {
        advance(); // consume ';'
    }
    
    // 解析条件部分
    std::unique_ptr<Expr> condition;
    if (!check(TokenType::Semicolon)) {
        auto cond = parseExpression();
        if (cond.isErr()) {
            return Result<std::unique_ptr<Stmt>>::Err(cond.error());
        }
        condition = std::move(cond.value());
    }
    auto semi2 = consume(TokenType::Semicolon, "期望 ';'");
    if (semi2.isErr()) {
        return Result<std::unique_ptr<Stmt>>::Err(semi2.error());
    }
    
    // 解析增量部分
    std::unique_ptr<Expr> increment;
    if (!check(TokenType::RightParen)) {
        auto incr = parseExpression();
        if (incr.isErr()) {
            return Result<std::unique_ptr<Stmt>>::Err(incr.error());
        }
        increment = std::move(incr.value());
    }
    
    auto rparen = consume(TokenType::RightParen, "期望 ')'");
    if (rparen.isErr()) {
        return Result<std::unique_ptr<Stmt>>::Err(rparen.error());
    }
    
    // 解析循环体
    auto body = parseStatement();
    if (body.isErr()) {
        return Result<std::unique_ptr<Stmt>>::Err(body.error());
    }
    
    return Result<std::unique_ptr<Stmt>>::Ok(
        std::make_unique<ForStmt>(
            std::move(initializer),
            std::move(condition),
            std::move(increment),
            std::move(body.value())
        )
    );
}

Result<std::unique_ptr<Stmt>> Parser::parseMatch() {
    auto lparen = consume(TokenType::LeftParen, "期望 '('");
    if (lparen.isErr()) {
        return Result<std::unique_ptr<Stmt>>::Err(lparen.error());
    }
    
    auto expr = parseExpression();
    if (expr.isErr()) {
        return Result<std::unique_ptr<Stmt>>::Err(expr.error());
    }
    
    auto rparen = consume(TokenType::RightParen, "期望 ')'");
    if (rparen.isErr()) {
        return Result<std::unique_ptr<Stmt>>::Err(rparen.error());
    }
    
    auto lbrace = consume(TokenType::LeftBrace, "期望 '{'");
    if (lbrace.isErr()) {
        return Result<std::unique_ptr<Stmt>>::Err(lbrace.error());
    }
    
    std::vector<MatchArm> arms;
    while (!check(TokenType::RightBrace) && !isAtEnd()) {
        // 解析模式
        auto pattern = parsePattern();
        if (pattern.isErr()) {
            return Result<std::unique_ptr<Stmt>>::Err(pattern.error());
        }
        
        // 可选的守卫条件
        std::unique_ptr<Expr> guard;
        if (match(TokenType::If)) {
            auto guardExpr = parseExpression();
            if (guardExpr.isErr()) {
                return Result<std::unique_ptr<Stmt>>::Err(guardExpr.error());
            }
            guard = std::move(guardExpr.value());
        }
        
        // =>
        auto arrow = consume(TokenType::Arrow, "期望 '=>'");
        if (arrow.isErr()) {
            return Result<std::unique_ptr<Stmt>>::Err(arrow.error());
        }
        
        // 分支体
        std::unique_ptr<Stmt> body;
        if (check(TokenType::LeftBrace)) {
            auto blockResult = parseBlock();
            if (blockResult.isErr()) {
                return Result<std::unique_ptr<Stmt>>::Err(blockResult.error());
            }
            body = std::move(blockResult.value());
        } else {
            // 单个表达式
            auto exprResult = parseExpression();
            if (exprResult.isErr()) {
                return Result<std::unique_ptr<Stmt>>::Err(exprResult.error());
            }
            body = std::make_unique<ExprStmt>(std::move(exprResult.value()));
        }
        
        // 可选的逗号
        match(TokenType::Comma);
        
        arms.emplace_back(std::move(pattern.value()), std::move(guard), std::move(body));
    }
    
    auto rbrace = consume(TokenType::RightBrace, "期望 '}'");
    if (rbrace.isErr()) {
        return Result<std::unique_ptr<Stmt>>::Err(rbrace.error());
    }
    
    return Result<std::unique_ptr<Stmt>>::Ok(
        std::make_unique<MatchStmt>(std::move(expr.value()), std::move(arms))
    );
}

Result<std::unique_ptr<Pattern>> Parser::parsePattern() {
    return parseOrPattern();
}

Result<std::unique_ptr<Pattern>> Parser::parseOrPattern() {
    auto pattern = parsePrimaryPattern();
    if (pattern.isErr()) return pattern;
    
    if (match(TokenType::Pipe)) {
        std::vector<std::unique_ptr<Pattern>> patterns;
        patterns.push_back(std::move(pattern.value()));
        
        do {
            auto nextPattern = parsePrimaryPattern();
            if (nextPattern.isErr()) return nextPattern;
            patterns.push_back(std::move(nextPattern.value()));
        } while (match(TokenType::Pipe));
        
        return Result<std::unique_ptr<Pattern>>::Ok(
            std::make_unique<OrPattern>(std::move(patterns))
        );
    }
    
    return pattern;
}

Result<std::unique_ptr<Pattern>> Parser::parsePrimaryPattern() {
    // 通配符模式: _
    if (match(TokenType::Identifier) && previous().lexeme == "_") {
        return Result<std::unique_ptr<Pattern>>::Ok(
            std::make_unique<WildcardPattern>()
        );
    }
    
    // 标识符模式
    if (check(TokenType::Identifier)) {
        std::string name = advance().lexeme;
        return Result<std::unique_ptr<Pattern>>::Ok(
            std::make_unique<IdentifierPattern>(name)
        );
    }
    
    // 字面量模式
    if (check(TokenType::IntLiteral) || check(TokenType::FloatLiteral) || 
        check(TokenType::StringLiteral)) {
        auto expr = parsePrimary();
        if (expr.isErr()) {
            return Result<std::unique_ptr<Pattern>>::Err(expr.error());
        }
        return Result<std::unique_ptr<Pattern>>::Ok(
            std::make_unique<LiteralPattern>(std::move(expr.value()))
        );
    }
    
    const Token& token = peek();
    return Result<std::unique_ptr<Pattern>>::Err(CompileError{
        ErrorKind::ParserError,
        "期望模式",
        token.line,
        token.column,
        filename_
    });
}

Result<std::unique_ptr<Stmt>> Parser::parseBlock() {
    auto lbrace = consume(TokenType::LeftBrace, "期望 '{'");
    if (lbrace.isErr()) {
        return Result<std::unique_ptr<Stmt>>::Err(lbrace.error());
    }
    
    std::vector<std::unique_ptr<Stmt>> statements;
    while (!check(TokenType::RightBrace) && !isAtEnd()) {
        auto stmt = parseStatement();
        if (stmt.isErr()) {
            return Result<std::unique_ptr<Stmt>>::Err(stmt.error());
        }
        statements.push_back(std::move(stmt.value()));
    }
    
    auto rbrace = consume(TokenType::RightBrace, "期望 '}'");
    if (rbrace.isErr()) {
        return Result<std::unique_ptr<Stmt>>::Err(rbrace.error());
    }
    
    return Result<std::unique_ptr<Stmt>>::Ok(
        std::make_unique<BlockStmt>(std::move(statements))
    );
}

Result<std::unique_ptr<Stmt>> Parser::parseExpressionStatement() {
    auto expr = parseExpression();
    if (expr.isErr()) {
        return Result<std::unique_ptr<Stmt>>::Err(expr.error());
    }
    
    auto semi = consume(TokenType::Semicolon, "期望 ';'");
    if (semi.isErr()) {
        return Result<std::unique_ptr<Stmt>>::Err(semi.error());
    }
    
    return Result<std::unique_ptr<Stmt>>::Ok(
        std::make_unique<ExprStmt>(std::move(expr.value()))
    );
}

// 表达式解析实现
Result<std::unique_ptr<Expr>> Parser::parseExpression() {
    return parseLogicalOr();
}

Result<std::unique_ptr<Expr>> Parser::parseLogicalOr() {
    auto left = parseLogicalAnd();
    if (left.isErr()) return left;
    
    while (match(TokenType::PipePipe)) {
        std::string op = previous().lexeme;
        auto right = parseLogicalAnd();
        if (right.isErr()) return right;
        
        left = Result<std::unique_ptr<Expr>>::Ok(
            std::make_unique<BinaryExpr>(op, std::move(left.value()), std::move(right.value()))
        );
    }
    
    return left;
}

Result<std::unique_ptr<Expr>> Parser::parseLogicalAnd() {
    auto left = parseEquality();
    if (left.isErr()) return left;
    
    while (match(TokenType::AmpAmp)) {
        std::string op = previous().lexeme;
        auto right = parseEquality();
        if (right.isErr()) return right;
        
        left = Result<std::unique_ptr<Expr>>::Ok(
            std::make_unique<BinaryExpr>(op, std::move(left.value()), std::move(right.value()))
        );
    }
    
    return left;
}

Result<std::unique_ptr<Expr>> Parser::parseEquality() {
    auto left = parseComparison();
    if (left.isErr()) return left;
    
    while (match({TokenType::EqualEqual, TokenType::BangEqual})) {
        std::string op = previous().lexeme;
        auto right = parseComparison();
        if (right.isErr()) return right;
        
        left = Result<std::unique_ptr<Expr>>::Ok(
            std::make_unique<BinaryExpr>(op, std::move(left.value()), std::move(right.value()))
        );
    }
    
    return left;
}

Result<std::unique_ptr<Expr>> Parser::parseComparison() {
    auto left = parseTerm();
    if (left.isErr()) return left;
    
    while (match({TokenType::Less, TokenType::LessEqual, 
                  TokenType::Greater, TokenType::GreaterEqual})) {
        std::string op = previous().lexeme;
        auto right = parseTerm();
        if (right.isErr()) return right;
        
        left = Result<std::unique_ptr<Expr>>::Ok(
            std::make_unique<BinaryExpr>(op, std::move(left.value()), std::move(right.value()))
        );
    }
    
    return left;
}

Result<std::unique_ptr<Expr>> Parser::parseTerm() {
    auto left = parseFactor();
    if (left.isErr()) return left;
    
    while (match({TokenType::Plus, TokenType::Minus})) {
        std::string op = previous().lexeme;
        auto right = parseFactor();
        if (right.isErr()) return right;
        
        left = Result<std::unique_ptr<Expr>>::Ok(
            std::make_unique<BinaryExpr>(op, std::move(left.value()), std::move(right.value()))
        );
    }
    
    return left;
}

Result<std::unique_ptr<Expr>> Parser::parseFactor() {
    auto left = parseUnary();
    if (left.isErr()) return left;
    
    while (match({TokenType::Star, TokenType::Slash, TokenType::Percent})) {
        std::string op = previous().lexeme;
        auto right = parseUnary();
        if (right.isErr()) return right;
        
        left = Result<std::unique_ptr<Expr>>::Ok(
            std::make_unique<BinaryExpr>(op, std::move(left.value()), std::move(right.value()))
        );
    }
    
    return left;
}

Result<std::unique_ptr<Expr>> Parser::parseUnary() {
    if (match({TokenType::Bang, TokenType::Minus})) {
        std::string op = previous().lexeme;
        auto operand = parseUnary();
        if (operand.isErr()) return operand;
        
        return Result<std::unique_ptr<Expr>>::Ok(
            std::make_unique<UnaryExpr>(op, std::move(operand.value()))
        );
    }
    
    return parsePostfix();
}

Result<std::unique_ptr<Expr>> Parser::parsePostfix() {
    auto expr = parsePrimary();
    if (expr.isErr()) return expr;
    
    while (true) {
        if (match(TokenType::LeftParen)) {
            // 函数调用
            std::vector<std::unique_ptr<Expr>> arguments;
            if (!check(TokenType::RightParen)) {
                do {
                    auto arg = parseExpression();
                    if (arg.isErr()) return arg;
                    arguments.push_back(std::move(arg.value()));
                } while (match(TokenType::Comma));
            }
            
            auto rparen = consume(TokenType::RightParen, "期望 ')'");
            if (rparen.isErr()) {
                return Result<std::unique_ptr<Expr>>::Err(rparen.error());
            }
            
            expr = Result<std::unique_ptr<Expr>>::Ok(
                std::make_unique<CallExpr>(std::move(expr.value()), std::move(arguments))
            );
        } else if (match(TokenType::Dot)) {
            auto member = consume(TokenType::Identifier, "期望成员名");
            if (member.isErr()) {
                return Result<std::unique_ptr<Expr>>::Err(member.error());
            }
            
            expr = Result<std::unique_ptr<Expr>>::Ok(
                std::make_unique<MemberExpr>(std::move(expr.value()), member.value().lexeme)
            );
        } else {
            break;
        }
    }
    
    return expr;
}

Result<std::unique_ptr<Expr>> Parser::parsePrimary() {
    // 数组字面量
    if (match(TokenType::LeftBracket)) {
        std::vector<std::unique_ptr<Expr>> elements;
        
        // 空数组
        if (!check(TokenType::RightBracket)) {
            do {
                auto element = parseExpression();
                if (element.isErr()) return element;
                elements.push_back(std::move(element.value()));
            } while (match(TokenType::Comma));
        }
        
        auto rbracket = consume(TokenType::RightBracket, "期望 ']'");
        if (rbracket.isErr()) {
            return Result<std::unique_ptr<Expr>>::Err(rbracket.error());
        }
        
        return Result<std::unique_ptr<Expr>>::Ok(
            std::make_unique<ArrayLiteralExpr>(std::move(elements))
        );
    }
    
    // 整数字面量
    if (match(TokenType::IntLiteral)) {
        int64_t value = std::stoll(previous().lexeme);
        return Result<std::unique_ptr<Expr>>::Ok(
            std::make_unique<IntLiteralExpr>(value)
        );
    }
    
    // 浮点数字面量
    if (match(TokenType::FloatLiteral)) {
        double value = std::stod(previous().lexeme);
        return Result<std::unique_ptr<Expr>>::Ok(
            std::make_unique<FloatLiteralExpr>(value)
        );
    }
    
    // 字符串字面量
    if (match(TokenType::StringLiteral)) {
        return Result<std::unique_ptr<Expr>>::Ok(
            std::make_unique<StringLiteralExpr>(previous().lexeme)
        );
    }
    
    // 标识符
    if (match(TokenType::Identifier)) {
        return Result<std::unique_ptr<Expr>>::Ok(
            std::make_unique<IdentifierExpr>(previous().lexeme)
        );
    }
    
    // 括号表达式
    if (match(TokenType::LeftParen)) {
        auto expr = parseExpression();
        if (expr.isErr()) return expr;
        
        auto rparen = consume(TokenType::RightParen, "期望 ')'");
        if (rparen.isErr()) {
            return Result<std::unique_ptr<Expr>>::Err(rparen.error());
        }
        
        return expr;
    }
    
    const Token& token = peek();
    return Result<std::unique_ptr<Expr>>::Err(CompileError{
        ErrorKind::ParserError,
        "期望表达式",
        token.line,
        token.column,
        filename_
    });
}

// 辅助函数实现
bool Parser::isAtEnd() const {
    return peek().type == TokenType::Eof;
}

const Token& Parser::peek() const {
    return tokens_[current_];
}

const Token& Parser::previous() const {
    return tokens_[current_ - 1];
}

const Token& Parser::advance() {
    if (!isAtEnd()) {
        current_++;
    }
    return previous();
}

bool Parser::check(TokenType type) const {
    if (isAtEnd()) return false;
    return peek().type == type;
}

bool Parser::match(TokenType type) {
    if (check(type)) {
        advance();
        return true;
    }
    return false;
}

bool Parser::match(std::initializer_list<TokenType> types) {
    for (TokenType type : types) {
        if (check(type)) {
            advance();
            return true;
        }
    }
    return false;
}

Result<Token> Parser::consume(TokenType type, const std::string& message) {
    if (check(type)) {
        return Result<Token>::Ok(advance());
    }
    
    const Token& token = peek();
    return Result<Token>::Err(CompileError{
        ErrorKind::ParserError,
        message,
        token.line,
        token.column,
        filename_
    });
}

} // namespace frontend
} // namespace az
