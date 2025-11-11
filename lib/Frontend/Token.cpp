// AZ编译器 - Token实现

#include "AZ/Frontend/Token.h"

namespace az {
namespace frontend {

const char* getTokenTypeName(TokenType type) {
    switch (type) {
        case TokenType::Fn: return "fn";
        case TokenType::Return: return "return";
        case TokenType::If: return "if";
        case TokenType::Else: return "else";
        case TokenType::For: return "for";
        case TokenType::While: return "while";
        case TokenType::Let: return "let";
        case TokenType::Var: return "var";
        case TokenType::Const: return "const";
        case TokenType::Import: return "import";
        case TokenType::Struct: return "struct";
        case TokenType::Enum: return "enum";
        case TokenType::Match: return "match";
        case TokenType::Comptime: return "comptime";
        case TokenType::Identifier: return "identifier";
        case TokenType::IntLiteral: return "int_literal";
        case TokenType::FloatLiteral: return "float_literal";
        case TokenType::StringLiteral: return "string_literal";
        case TokenType::Plus: return "+";
        case TokenType::Minus: return "-";
        case TokenType::Star: return "*";
        case TokenType::Slash: return "/";
        case TokenType::Percent: return "%";
        case TokenType::Equal: return "=";
        case TokenType::EqualEqual: return "==";
        case TokenType::BangEqual: return "!=";
        case TokenType::Less: return "<";
        case TokenType::LessEqual: return "<=";
        case TokenType::Greater: return ">";
        case TokenType::GreaterEqual: return ">=";
        case TokenType::AmpAmp: return "&&";
        case TokenType::PipePipe: return "||";
        case TokenType::Bang: return "!";
        case TokenType::LeftParen: return "(";
        case TokenType::RightParen: return ")";
        case TokenType::LeftBrace: return "{";
        case TokenType::RightBrace: return "}";
        case TokenType::LeftBracket: return "[";
        case TokenType::RightBracket: return "]";
        case TokenType::Comma: return ",";
        case TokenType::Semicolon: return ";";
        case TokenType::Colon: return ":";
        case TokenType::Dot: return ".";
        case TokenType::Arrow: return "->";
        case TokenType::Pipe: return "|";
        case TokenType::Eof: return "EOF";
        case TokenType::Unknown: return "unknown";
        default: return "unknown";
    }
}

} // namespace frontend
} // namespace az
