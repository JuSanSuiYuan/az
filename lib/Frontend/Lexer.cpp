// AZ编译器词法分析器实现
// 支持多编码（UTF-8, GBK, GB2312, GB18030等）

#include "AZ/Frontend/Lexer.h"
#include "AZ/Support/Result.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Unicode.h"
#include <unicode/ucnv.h>  // ICU for encoding conversion

namespace az {
namespace frontend {

using namespace az::support;

// 编码检测和转换
class EncodingConverter {
public:
    // 检测文件编码
    static Result<std::string> detectEncoding(llvm::StringRef content) {
        // 检查UTF-8 BOM
        if (content.startswith("\xEF\xBB\xBF")) {
            return Result<std::string>::Ok("UTF-8");
        }
        
        // 检查UTF-16 LE BOM
        if (content.startswith("\xFF\xFE")) {
            return Result<std::string>::Ok("UTF-16LE");
        }
        
        // 检查UTF-16 BE BOM
        if (content.startswith("\xFE\xFF")) {
            return Result<std::string>::Ok("UTF-16BE");
        }
        
        // 尝试UTF-8验证
        if (llvm::isLegalUTF8String(
                reinterpret_cast<const llvm::UTF8*>(content.data()),
                reinterpret_cast<const llvm::UTF8*>(content.data() + content.size()))) {
            return Result<std::string>::Ok("UTF-8");
        }
        
        // 默认尝试GBK（中文环境常见）
        return Result<std::string>::Ok("GBK");
    }
    
    // 转换到UTF-8
    static Result<std::string> convertToUTF8(llvm::StringRef content, 
                                              const std::string& encoding) {
        if (encoding == "UTF-8") {
            return Result<std::string>::Ok(content.str());
        }
        
        UErrorCode status = U_ZERO_ERROR;
        UConverter* conv = ucnv_open(encoding.c_str(), &status);
        if (U_FAILURE(status)) {
            return Result<std::string>::Err(CompileError{
                ErrorKind::LexerError,
                "不支持的编码: " + encoding,
                0, 0, ""
            });
        }
        
        // 转换到UTF-8
        int32_t destLen = content.size() * 4;  // 最坏情况
        std::string utf8(destLen, '\0');
        
        destLen = ucnv_toAlgorithmic(UCNV_UTF8, conv,
                                      &utf8[0], destLen,
                                      content.data(), content.size(),
                                      &status);
        
        ucnv_close(conv);
        
        if (U_FAILURE(status)) {
            return Result<std::string>::Err(CompileError{
                ErrorKind::LexerError,
                "编码转换失败",
                0, 0, ""
            });
        }
        
        utf8.resize(destLen);
        return Result<std::string>::Ok(utf8);
    }
};

// 词法分析器实现
Lexer::Lexer(llvm::StringRef source, llvm::StringRef filename)
    : source_(source), filename_(filename), current_(0), line_(1), column_(1) {
    
    // 检测并转换编码
    auto encoding = EncodingConverter::detectEncoding(source);
    if (encoding.isOk()) {
        auto utf8 = EncodingConverter::convertToUTF8(source, encoding.value());
        if (utf8.isOk()) {
            source_ = utf8.value();
        }
    }
}

Result<std::vector<Token>> Lexer::tokenize() {
    std::vector<Token> tokens;
    
    while (!isAtEnd()) {
        auto result = scanToken();
        if (!result.isOk()) {
            return Result<std::vector<Token>>::Err(result.error());
        }
        
        if (result.value().has_value()) {
            tokens.push_back(result.value().value());
        }
    }
    
    // 添加EOF token
    tokens.push_back(Token{TokenType::Eof, "", line_, column_});
    
    return Result<std::vector<Token>>::Ok(tokens);
}

Result<std::optional<Token>> Lexer::scanToken() {
    skipWhitespace();
    
    if (isAtEnd()) {
        return Result<std::optional<Token>>::Ok(std::nullopt);
    }
    
    size_t start = current_;
    size_t startLine = line_;
    size_t startColumn = column_;
    
    char c = advance();
    
    // 标识符和关键字（支持中文）
    if (isAlpha(c) || isChineseChar(c)) {
        return scanIdentifier(start, startLine, startColumn);
    }
    
    // 数字字面量
    if (isDigit(c)) {
        return scanNumber(start, startLine, startColumn);
    }
    
    // 字符串字面量
    if (c == '"') {
        return scanString(startLine, startColumn);
    }
    
    // 运算符和分隔符
    switch (c) {
        case '+': return makeToken(TokenType::Plus, "+", startLine, startColumn);
        case '-':
            if (match('>')) {
                return makeToken(TokenType::Arrow, "->", startLine, startColumn);
            }
            return makeToken(TokenType::Minus, "-", startLine, startColumn);
        case '*': return makeToken(TokenType::Star, "*", startLine, startColumn);
        case '/':
            if (match('/')) {
                skipLineComment();
                return Result<std::optional<Token>>::Ok(std::nullopt);
            }
            return makeToken(TokenType::Slash, "/", startLine, startColumn);
        case '%': return makeToken(TokenType::Percent, "%", startLine, startColumn);
        case '=':
            if (match('=')) {
                return makeToken(TokenType::EqualEqual, "==", startLine, startColumn);
            }
            return makeToken(TokenType::Equal, "=", startLine, startColumn);
        case '!':
            if (match('=')) {
                return makeToken(TokenType::BangEqual, "!=", startLine, startColumn);
            }
            return makeToken(TokenType::Bang, "!", startLine, startColumn);
        case '<':
            if (match('=')) {
                return makeToken(TokenType::LessEqual, "<=", startLine, startColumn);
            }
            return makeToken(TokenType::Less, "<", startLine, startColumn);
        case '>':
            if (match('=')) {
                return makeToken(TokenType::GreaterEqual, ">=", startLine, startColumn);
            }
            return makeToken(TokenType::Greater, ">", startLine, startColumn);
        case '&':
            if (match('&')) {
                return makeToken(TokenType::AmpAmp, "&&", startLine, startColumn);
            }
            break;
        case '|':
            if (match('|')) {
                return makeToken(TokenType::PipePipe, "||", startLine, startColumn);
            }
            return makeToken(TokenType::Pipe, "|", startLine, startColumn);
        case '(': return makeToken(TokenType::LeftParen, "(", startLine, startColumn);
        case ')': return makeToken(TokenType::RightParen, ")", startLine, startColumn);
        case '{': return makeToken(TokenType::LeftBrace, "{", startLine, startColumn);
        case '}': return makeToken(TokenType::RightBrace, "}", startLine, startColumn);
        case '[': return makeToken(TokenType::LeftBracket, "[", startLine, startColumn);
        case ']': return makeToken(TokenType::RightBracket, "]", startLine, startColumn);
        case ',': return makeToken(TokenType::Comma, ",", startLine, startColumn);
        case ';': return makeToken(TokenType::Semicolon, ";", startLine, startColumn);
        case ':': return makeToken(TokenType::Colon, ":", startLine, startColumn);
        case '.': return makeToken(TokenType::Dot, ".", startLine, startColumn);
    }
    
    // 未知字符
    return Result<std::optional<Token>>::Err(CompileError{
        ErrorKind::LexerError,
        std::string("未知字符: '") + c + "'",
        startLine,
        startColumn,
        filename_.str()
    });
}

bool Lexer::isChineseChar(char c) const {
    // 简单检查：UTF-8中文字符的第一个字节通常在0xE0-0xEF范围
    unsigned char uc = static_cast<unsigned char>(c);
    return uc >= 0xE0 && uc <= 0xEF;
}

Result<std::optional<Token>> Lexer::scanIdentifier(size_t start, 
                                                     size_t line, 
                                                     size_t column) {
    while (!isAtEnd() && (isAlnum(peek()) || peek() == '_' || isChineseChar(peek()))) {
        advance();
    }
    
    std::string lexeme = source_.substr(start, current_ - start).str();
    
    // 检查是否是关键字
    TokenType type = getKeywordType(lexeme);
    
    return makeToken(type, lexeme, line, column);
}

Result<std::optional<Token>> Lexer::scanNumber(size_t start, 
                                                 size_t line, 
                                                 size_t column) {
    while (!isAtEnd() && isDigit(peek())) {
        advance();
    }
    
    // 浮点数
    if (!isAtEnd() && peek() == '.' && isDigit(peekNext())) {
        advance();  // consume '.'
        while (!isAtEnd() && isDigit(peek())) {
            advance();
        }
        
        std::string lexeme = source_.substr(start, current_ - start).str();
        return makeToken(TokenType::FloatLiteral, lexeme, line, column);
    }
    
    std::string lexeme = source_.substr(start, current_ - start).str();
    return makeToken(TokenType::IntLiteral, lexeme, line, column);
}

Result<std::optional<Token>> Lexer::scanString(size_t line, size_t column) {
    std::string value;
    
    while (!isAtEnd() && peek() != '"') {
        if (peek() == '\n') {
            line_++;
            column_ = 0;
        }
        
        // 转义字符
        if (peek() == '\\') {
            advance();
            if (!isAtEnd()) {
                char escaped = advance();
                switch (escaped) {
                    case 'n': value += '\n'; break;
                    case 't': value += '\t'; break;
                    case 'r': value += '\r'; break;
                    case '\\': value += '\\'; break;
                    case '"': value += '"'; break;
                    default: value += escaped; break;
                }
            }
        } else {
            value += advance();
        }
    }
    
    if (isAtEnd()) {
        return Result<std::optional<Token>>::Err(CompileError{
            ErrorKind::LexerError,
            "未终止的字符串",
            line,
            column,
            filename_.str()
        });
    }
    
    advance();  // consume closing "
    
    return makeToken(TokenType::StringLiteral, value, line, column);
}

TokenType Lexer::getKeywordType(const std::string& word) const {
    static const std::unordered_map<std::string, TokenType> keywords = {
        {"fn", TokenType::Fn},
        {"return", TokenType::Return},
        {"if", TokenType::If},
        {"else", TokenType::Else},
        {"for", TokenType::For},
        {"while", TokenType::While},
        {"let", TokenType::Let},
        {"令", TokenType::Let},  // 中文关键字
        {"var", TokenType::Var},
        {"设", TokenType::Var},  // 中文关键字
        {"const", TokenType::Const},
        {"import", TokenType::Import},
        {"struct", TokenType::Struct},
        {"enum", TokenType::Enum},
        {"match", TokenType::Match},
        {"comptime", TokenType::Comptime},
    };
    
    auto it = keywords.find(word);
    if (it != keywords.end()) {
        return it->second;
    }
    
    return TokenType::Identifier;
}

void Lexer::skipWhitespace() {
    while (!isAtEnd()) {
        char c = peek();
        switch (c) {
            case ' ':
            case '\r':
            case '\t':
                advance();
                break;
            case '\n':
                line_++;
                column_ = 0;
                advance();
                break;
            default:
                return;
        }
    }
}

void Lexer::skipLineComment() {
    while (!isAtEnd() && peek() != '\n') {
        advance();
    }
}

} // namespace frontend
} // namespace az
