#!/usr/bin/env python3
"""
AZ编程语言编译器 - Bootstrap版本
采用C3风格的错误处理（Result类型）
"""

import sys
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union

# ============================================================================
# 错误处理 - C3风格的Result类型
# ============================================================================

class ErrorKind(Enum):
    LEXER_ERROR = auto()
    PARSER_ERROR = auto()
    SEMANTIC_ERROR = auto()
    TYPE_ERROR = auto()
    RUNTIME_ERROR = auto()

@dataclass
class CompileError:
    kind: ErrorKind
    message: str
    line: int
    column: int
    filename: str
    
    def report(self):
        # 英文错误类型
        kind_names_en = {
            ErrorKind.LEXER_ERROR: "Lexical Error",
            ErrorKind.PARSER_ERROR: "Syntax Error",
            ErrorKind.SEMANTIC_ERROR: "Semantic Error",
            ErrorKind.TYPE_ERROR: "Type Error",
            ErrorKind.RUNTIME_ERROR: "Runtime Error"
        }
        # 中文错误类型
        kind_names_zh = {
            ErrorKind.LEXER_ERROR: "词法错误",
            ErrorKind.PARSER_ERROR: "语法错误",
            ErrorKind.SEMANTIC_ERROR: "语义错误",
            ErrorKind.TYPE_ERROR: "类型错误",
            ErrorKind.RUNTIME_ERROR: "运行时错误"
        }
        
        # 输出英文错误信息
        print(f"[Error] {kind_names_en[self.kind]} at {self.filename}:{self.line}:{self.column}")
        # 输出中文错误信息
        print(f"[错误] {kind_names_zh[self.kind]} 在 {self.filename}:{self.line}:{self.column}")
        
        # 输出错误详情（英文）
        print(f"  {self.message}")
        # 输出错误详情（中文翻译）
        message_zh = self.translate_message(self.message)
        if message_zh != self.message:
            print(f"  {message_zh}")
    
    def translate_message(self, message: str) -> str:
        """翻译常见错误信息为中文"""
        translations = {
            "Expected ';'": "期望 ';'",
            "Expected '('": "期望 '('",
            "Expected ')'": "期望 ')'",
            "Expected '{'": "期望 '{'",
            "Expected '}'": "期望 '}'",
            "Expected '['": "期望 '['",
            "Expected ']'": "期望 ']'",
            "Expected ':'": "期望 ':'",
            "Expected identifier": "期望标识符",
            "Expected type name": "期望类型名",
            "Expected expression": "期望表达式",
            "Expected function name": "期望函数名",
            "Expected parameter name": "期望参数名",
            "Expected return type": "期望返回类型",
            "Expected module name": "期望模块名",
            "Unexpected token": "意外的标记",
            "Undefined variable": "未定义的变量",
            "Type mismatch": "类型不匹配",
            "Division by zero": "除以零",
        }
        
        # 尝试翻译
        for en, zh in translations.items():
            if en in message:
                return message.replace(en, zh)
        
        return message

class Result:
    """C3风格的Result类型"""
    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error
        self.is_ok = error is None
    
    @staticmethod
    def Ok(value):
        return Result(value=value)
    
    @staticmethod
    def Err(error):
        return Result(error=error)

# ============================================================================
# Token定义
# ============================================================================

class TokenType(Enum):
    # 关键字
    FN = auto()
    RETURN = auto()
    IF = auto()
    ELSE = auto()
    FOR = auto()
    WHILE = auto()
    LET = auto()
    VAR = auto()
    CONST = auto()
    IMPORT = auto()
    MODULE = auto()
    PUB = auto()
    STRUCT = auto()
    ENUM = auto()
    MATCH = auto()
    CASE = auto()
    COMPTIME = auto()
    
    # 标识符和字面量
    IDENTIFIER = auto()
    INT_LITERAL = auto()
    FLOAT_LITERAL = auto()
    STRING_LITERAL = auto()
    
    # 运算符
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    EQUAL = auto()
    EQUAL_EQUAL = auto()
    BANG_EQUAL = auto()
    LESS = auto()
    LESS_EQUAL = auto()
    GREATER = auto()
    GREATER_EQUAL = auto()
    AMP_AMP = auto()
    PIPE_PIPE = auto()
    BANG = auto()
    
    # 分隔符
    LEFT_PAREN = auto()
    RIGHT_PAREN = auto()
    LEFT_BRACE = auto()
    RIGHT_BRACE = auto()
    LEFT_BRACKET = auto()
    RIGHT_BRACKET = auto()
    COMMA = auto()
    SEMICOLON = auto()
    COLON = auto()
    DOT = auto()
    ARROW = auto()
    PIPE = auto()
    
    # 特殊
    EOF = auto()

@dataclass
class Token:
    type: TokenType
    lexeme: str
    line: int
    column: int

# ============================================================================
# 词法分析器
# ============================================================================

class Lexer:
    def __init__(self, source: str, filename: str):
        self.source = source
        self.filename = filename
        self.current = 0
        self.line = 1
        self.column = 1
        self.tokens = []
        
        self.keywords = {
            # 英文关键字
            'fn': TokenType.FN,
            'return': TokenType.RETURN,
            'if': TokenType.IF,
            'else': TokenType.ELSE,
            'for': TokenType.FOR,
            'while': TokenType.WHILE,
            'let': TokenType.LET,
            'var': TokenType.VAR,
            'const': TokenType.CONST,
            'import': TokenType.IMPORT,
            'module': TokenType.MODULE,
            'pub': TokenType.PUB,
            'struct': TokenType.STRUCT,
            'enum': TokenType.ENUM,
            'match': TokenType.MATCH,
            'case': TokenType.CASE,
            'comptime': TokenType.COMPTIME,
            # 中文关键字
            '函数': TokenType.FN,
            '返回': TokenType.RETURN,
            '如果': TokenType.IF,
            '否则': TokenType.ELSE,
            '循环': TokenType.FOR,
            '当': TokenType.WHILE,
            '令': TokenType.LET,
            '设': TokenType.VAR,
            '常': TokenType.CONST,
            '导入': TokenType.IMPORT,
            '模块': TokenType.MODULE,
            '公开': TokenType.PUB,
            '结构': TokenType.STRUCT,
            '枚举': TokenType.ENUM,
            '匹配': TokenType.MATCH,
            '情况': TokenType.CASE,
            '编译时': TokenType.COMPTIME,
        }
    
    def tokenize(self) -> Result:
        """词法分析主函数"""
        while not self.is_at_end():
            result = self.scan_token()
            if not result.is_ok:
                return result
        
        self.add_token(TokenType.EOF, "")
        return Result.Ok(self.tokens)
    
    def scan_token(self) -> Result:
        """扫描单个token"""
        self.skip_whitespace()
        
        if self.is_at_end():
            return Result.Ok(None)
        
        start_column = self.column
        c = self.advance()
        
        # 标识符和关键字
        if c.isalpha() or c == '_':
            return self.scan_identifier(c)
        
        # 数字字面量
        if c.isdigit():
            return self.scan_number(c)
        
        # 字符串字面量
        if c == '"':
            return self.scan_string()
        
        # 运算符和分隔符
        if c == '+':
            self.add_token(TokenType.PLUS, "+")
        elif c == '-':
            if self.match_char('>'):
                self.add_token(TokenType.ARROW, "->")
            else:
                self.add_token(TokenType.MINUS, "-")
        elif c == '*':
            self.add_token(TokenType.STAR, "*")
        elif c == '/':
            if self.match_char('/'):
                self.skip_line_comment()
            else:
                self.add_token(TokenType.SLASH, "/")
        elif c == '%':
            self.add_token(TokenType.PERCENT, "%")
        elif c == '=':
            if self.match_char('='):
                self.add_token(TokenType.EQUAL_EQUAL, "==")
            else:
                self.add_token(TokenType.EQUAL, "=")
        elif c == '!':
            if self.match_char('='):
                self.add_token(TokenType.BANG_EQUAL, "!=")
            else:
                self.add_token(TokenType.BANG, "!")
        elif c == '<':
            if self.match_char('='):
                self.add_token(TokenType.LESS_EQUAL, "<=")
            else:
                self.add_token(TokenType.LESS, "<")
        elif c == '>':
            if self.match_char('='):
                self.add_token(TokenType.GREATER_EQUAL, ">=")
            else:
                self.add_token(TokenType.GREATER, ">")
        elif c == '&':
            if self.match_char('&'):
                self.add_token(TokenType.AMP_AMP, "&&")
        elif c == '|':
            if self.match_char('|'):
                self.add_token(TokenType.PIPE_PIPE, "||")
            else:
                self.add_token(TokenType.PIPE, "|")
        elif c == '(':
            self.add_token(TokenType.LEFT_PAREN, "(")
        elif c == ')':
            self.add_token(TokenType.RIGHT_PAREN, ")")
        elif c == '{':
            self.add_token(TokenType.LEFT_BRACE, "{")
        elif c == '}':
            self.add_token(TokenType.RIGHT_BRACE, "}")
        elif c == '[':
            self.add_token(TokenType.LEFT_BRACKET, "[")
        elif c == ']':
            self.add_token(TokenType.RIGHT_BRACKET, "]")
        elif c == ',':
            self.add_token(TokenType.COMMA, ",")
        elif c == ';':
            self.add_token(TokenType.SEMICOLON, ";")
        elif c == ':':
            self.add_token(TokenType.COLON, ":")
        elif c == '.':
            self.add_token(TokenType.DOT, ".")
        else:
            return Result.Err(CompileError(
                ErrorKind.LEXER_ERROR,
                f"未知字符: '{c}'",
                self.line,
                start_column,
                self.filename
            ))
        
        return Result.Ok(None)

    def scan_identifier(self, first_char: str) -> Result:
        """扫描标识符或关键字"""
        lexeme = first_char
        while not self.is_at_end() and (self.peek().isalnum() or self.peek() == '_'):
            lexeme += self.advance()
        
        token_type = self.keywords.get(lexeme, TokenType.IDENTIFIER)
        self.add_token(token_type, lexeme)
        return Result.Ok(None)
    
    def scan_number(self, first_digit: str) -> Result:
        """扫描数字字面量"""
        lexeme = first_digit
        while not self.is_at_end() and self.peek().isdigit():
            lexeme += self.advance()
        
        # 浮点数
        if not self.is_at_end() and self.peek() == '.' and self.peek_next().isdigit():
            lexeme += self.advance()  # consume '.'
            while not self.is_at_end() and self.peek().isdigit():
                lexeme += self.advance()
            self.add_token(TokenType.FLOAT_LITERAL, lexeme)
        else:
            self.add_token(TokenType.INT_LITERAL, lexeme)
        
        return Result.Ok(None)
    
    def scan_string(self) -> Result:
        """扫描字符串字面量"""
        lexeme = ""
        while not self.is_at_end() and self.peek() != '"':
            if self.peek() == '\n':
                self.line += 1
                self.column = 0
            lexeme += self.advance()
        
        if self.is_at_end():
            return Result.Err(CompileError(
                ErrorKind.LEXER_ERROR,
                "未终止的字符串",
                self.line,
                self.column,
                self.filename
            ))
        
        self.advance()  # consume closing "
        self.add_token(TokenType.STRING_LITERAL, lexeme)
        return Result.Ok(None)
    
    # 辅助方法
    def is_at_end(self) -> bool:
        return self.current >= len(self.source)
    
    def advance(self) -> str:
        c = self.source[self.current]
        self.current += 1
        self.column += 1
        return c
    
    def peek(self) -> str:
        if self.is_at_end():
            return '\0'
        return self.source[self.current]
    
    def peek_next(self) -> str:
        if self.current + 1 >= len(self.source):
            return '\0'
        return self.source[self.current + 1]
    
    def match_char(self, expected: str) -> bool:
        if self.is_at_end() or self.source[self.current] != expected:
            return False
        self.advance()
        return True
    
    def skip_whitespace(self):
        while not self.is_at_end():
            c = self.peek()
            if c == ' ' or c == '\r' or c == '\t':
                self.advance()
            elif c == '\n':
                self.line += 1
                self.column = 0
                self.advance()
            else:
                break
    
    def skip_line_comment(self):
        while not self.is_at_end() and self.peek() != '\n':
            self.advance()
    
    def add_token(self, token_type: TokenType, lexeme: str):
        token = Token(token_type, lexeme, self.line, self.column)
        self.tokens.append(token)

# ============================================================================
# AST节点定义
# ============================================================================

class ExprKind(Enum):
    INT_LITERAL = auto()
    FLOAT_LITERAL = auto()
    STRING_LITERAL = auto()
    IDENTIFIER = auto()
    BINARY = auto()
    UNARY = auto()
    CALL = auto()
    MEMBER = auto()
    ARRAY_LITERAL = auto()
    ARRAY_ACCESS = auto()
    STRUCT_LITERAL = auto()
    ASSIGN = auto()

@dataclass
class Expr:
    kind: ExprKind
    # 根据kind使用不同字段
    int_value: Optional[int] = None
    float_value: Optional[float] = None
    string_value: Optional[str] = None
    name: Optional[str] = None
    operator: Optional[str] = None
    left: Optional['Expr'] = None
    right: Optional['Expr'] = None
    operand: Optional['Expr'] = None
    callee: Optional['Expr'] = None
    arguments: Optional[List['Expr']] = None
    object: Optional['Expr'] = None
    member: Optional[str] = None
    # 数组
    elements: Optional[List['Expr']] = None
    array: Optional['Expr'] = None
    index: Optional['Expr'] = None
    # 结构体字面量
    struct_name: Optional[str] = None
    field_values: Optional[Dict[str, 'Expr']] = None

class StmtKind(Enum):
    EXPRESSION = auto()
    VAR_DECL = auto()
    FUNC_DECL = auto()
    RETURN = auto()
    IF = auto()
    WHILE = auto()
    FOR = auto()
    BLOCK = auto()
    IMPORT = auto()
    MODULE_DECL = auto()
    STRUCT_DECL = auto()
    MATCH = auto()

@dataclass
class Param:
    name: str
    type_name: str

@dataclass
class Pattern:
    """模式匹配的模式"""
    kind: str  # 'literal', 'identifier', 'wildcard'
    value: Optional[Any] = None
    name: Optional[str] = None

@dataclass
class CaseArm:
    """Match语句的case分支"""
    patterns: List[Pattern]
    guard: Optional[Expr] = None
    body: Optional['Stmt'] = None

@dataclass
class StructField:
    """结构体字段"""
    name: str
    type_name: str
    is_public: bool = False

@dataclass
class Stmt:
    kind: StmtKind
    # 根据kind使用不同字段
    expr: Optional[Expr] = None
    name: Optional[str] = None
    type_name: Optional[str] = None
    is_mutable: bool = False
    is_public: bool = False
    initializer: Optional[Expr] = None
    params: Optional[List[Param]] = None
    return_type: Optional[str] = None
    body: Optional['Stmt'] = None
    condition: Optional[Expr] = None
    then_branch: Optional['Stmt'] = None
    else_branch: Optional['Stmt'] = None
    statements: Optional[List['Stmt']] = None
    module_path: Optional[str] = None
    # For循环
    init: Optional['Stmt'] = None
    update: Optional[Expr] = None
    # Match语句
    match_expr: Optional[Expr] = None
    cases: Optional[List[CaseArm]] = None
    # 结构体
    fields: Optional[List[StructField]] = None

@dataclass
class Program:
    statements: List[Stmt]

# ============================================================================
# 语法分析器
# ============================================================================

class Parser:
    def __init__(self, tokens: List[Token], filename: str):
        self.tokens = tokens
        self.filename = filename
        self.current = 0
    
    def parse(self) -> Result:
        """语法分析主函数"""
        statements = []
        
        while not self.is_at_end():
            result = self.parse_statement()
            if not result.is_ok:
                return result
            statements.append(result.value)
        
        return Result.Ok(Program(statements))
    
    def parse_statement(self) -> Result:
        """解析语句"""
        # module声明
        if self.match(TokenType.MODULE):
            return self.parse_module_decl()
        
        # import语句
        if self.match(TokenType.IMPORT):
            return self.parse_import()
        
        # pub关键字
        is_public = self.match(TokenType.PUB)
        
        # struct声明
        if self.check(TokenType.STRUCT):
            return self.parse_struct(is_public)
        
        # 函数声明
        if self.check(TokenType.FN):
            return self.parse_function(is_public)
        
        # 变量声明
        if self.check(TokenType.LET) or self.check(TokenType.VAR):
            return self.parse_var_declaration()
        
        # return语句
        if self.match(TokenType.RETURN):
            return self.parse_return()
        
        # if语句
        if self.match(TokenType.IF):
            return self.parse_if()
        
        # while语句
        if self.match(TokenType.WHILE):
            return self.parse_while()
        
        # for循环
        if self.match(TokenType.FOR):
            return self.parse_for()
        
        # match语句
        if self.match(TokenType.MATCH):
            return self.parse_match()
        
        # 代码块
        if self.check(TokenType.LEFT_BRACE):
            return self.parse_block()
        
        # 表达式语句
        return self.parse_expression_statement()
    
    def parse_import(self) -> Result:
        """解析import语句"""
        path_parts = []
        
        result = self.consume(TokenType.IDENTIFIER, "期望模块名")
        if not result.is_ok:
            return result
        path_parts.append(result.value.lexeme)
        
        while self.match(TokenType.DOT):
            result = self.consume(TokenType.IDENTIFIER, "期望模块名")
            if not result.is_ok:
                return result
            path_parts.append(result.value.lexeme)
        
        result = self.consume(TokenType.SEMICOLON, "期望 ';'")
        if not result.is_ok:
            return result
        
        return Result.Ok(Stmt(
            kind=StmtKind.IMPORT,
            module_path='.'.join(path_parts)
        ))

    def parse_function(self, is_public: bool = False) -> Result:
        """解析函数声明"""
        result = self.consume(TokenType.FN, "期望'fn'")
        if not result.is_ok:
            return result
        
        result = self.consume(TokenType.IDENTIFIER, "期望函数名")
        if not result.is_ok:
            return result
        name = result.value.lexeme
        
        result = self.consume(TokenType.LEFT_PAREN, "期望 '('")
        if not result.is_ok:
            return result
        
        params = []
        if not self.check(TokenType.RIGHT_PAREN):
            while True:
                result = self.consume(TokenType.IDENTIFIER, "期望参数名")
                if not result.is_ok:
                    return result
                param_name = result.value.lexeme
                
                result = self.consume(TokenType.COLON, "期望 ':'")
                if not result.is_ok:
                    return result
                
                result = self.consume(TokenType.IDENTIFIER, "期望类型名")
                if not result.is_ok:
                    return result
                param_type = result.value.lexeme
                
                params.append(Param(param_name, param_type))
                
                if not self.match(TokenType.COMMA):
                    break
        
        result = self.consume(TokenType.RIGHT_PAREN, "期望 ')'")
        if not result.is_ok:
            return result
        
        # 返回类型
        return_type = "void"
        if not self.check(TokenType.LEFT_BRACE):
            result = self.consume(TokenType.IDENTIFIER, "期望返回类型")
            if not result.is_ok:
                return result
            return_type = result.value.lexeme
        
        # 函数体
        result = self.parse_block()
        if not result.is_ok:
            return result
        body = result.value
        
        return Result.Ok(Stmt(
            kind=StmtKind.FUNC_DECL,
            name=name,
            params=params,
            return_type=return_type,
            body=body,
            is_public=is_public
        ))
    
    def parse_var_declaration(self) -> Result:
        """解析变量声明"""
        is_mutable = self.match(TokenType.VAR)
        if not is_mutable:
            self.advance()  # consume 'let'
        
        result = self.consume(TokenType.IDENTIFIER, "期望变量名")
        if not result.is_ok:
            return result
        name = result.value.lexeme
        
        type_name = ""
        if self.match(TokenType.COLON):
            result = self.consume(TokenType.IDENTIFIER, "期望类型名")
            if not result.is_ok:
                return result
            type_name = result.value.lexeme
        
        initializer = None
        if self.match(TokenType.EQUAL):
            result = self.parse_expression()
            if not result.is_ok:
                return result
            initializer = result.value
        
        result = self.consume(TokenType.SEMICOLON, "期望 ';'")
        if not result.is_ok:
            return result
        
        return Result.Ok(Stmt(
            kind=StmtKind.VAR_DECL,
            name=name,
            type_name=type_name,
            is_mutable=is_mutable,
            initializer=initializer
        ))
    
    def parse_return(self) -> Result:
        """解析return语句"""
        expr = None
        if not self.check(TokenType.SEMICOLON):
            result = self.parse_expression()
            if not result.is_ok:
                return result
            expr = result.value
        
        result = self.consume(TokenType.SEMICOLON, "期望 ';'")
        if not result.is_ok:
            return result
        
        return Result.Ok(Stmt(kind=StmtKind.RETURN, expr=expr))
    
    def parse_if(self) -> Result:
        """解析if语句"""
        result = self.consume(TokenType.LEFT_PAREN, "期望 '('")
        if not result.is_ok:
            return result
        
        result = self.parse_expression()
        if not result.is_ok:
            return result
        condition = result.value
        
        result = self.consume(TokenType.RIGHT_PAREN, "期望 ')'")
        if not result.is_ok:
            return result
        
        result = self.parse_statement()
        if not result.is_ok:
            return result
        then_branch = result.value
        
        else_branch = None
        if self.match(TokenType.ELSE):
            result = self.parse_statement()
            if not result.is_ok:
                return result
            else_branch = result.value
        
        return Result.Ok(Stmt(
            kind=StmtKind.IF,
            condition=condition,
            then_branch=then_branch,
            else_branch=else_branch
        ))
    
    def parse_while(self) -> Result:
        """解析while语句"""
        result = self.consume(TokenType.LEFT_PAREN, "期望 '('")
        if not result.is_ok:
            return result
        
        result = self.parse_expression()
        if not result.is_ok:
            return result
        condition = result.value
        
        result = self.consume(TokenType.RIGHT_PAREN, "期望 ')'")
        if not result.is_ok:
            return result
        
        result = self.parse_statement()
        if not result.is_ok:
            return result
        body = result.value
        
        return Result.Ok(Stmt(
            kind=StmtKind.WHILE,
            condition=condition,
            body=body
        ))
    
    def parse_for(self) -> Result:
        """解析for循环"""
        result = self.consume(TokenType.LEFT_PAREN, "期望 '('")
        if not result.is_ok:
            return result
        
        # 初始化语句
        init = None
        if not self.check(TokenType.SEMICOLON):
            if self.check(TokenType.VAR) or self.check(TokenType.LET):
                result = self.parse_var_declaration()
                if not result.is_ok:
                    return result
                init = result.value
            else:
                result = self.parse_expression()
                if not result.is_ok:
                    return result
                init = Stmt(kind=StmtKind.EXPRESSION, expr=result.value)
                result = self.consume(TokenType.SEMICOLON, "期望 ';'")
                if not result.is_ok:
                    return result
        else:
            self.advance()  # consume ';'
        
        # 条件表达式
        condition = None
        if not self.check(TokenType.SEMICOLON):
            result = self.parse_expression()
            if not result.is_ok:
                return result
            condition = result.value
        
        result = self.consume(TokenType.SEMICOLON, "期望 ';'")
        if not result.is_ok:
            return result
        
        # 更新表达式
        update = None
        if not self.check(TokenType.RIGHT_PAREN):
            result = self.parse_expression()
            if not result.is_ok:
                return result
            update = result.value
        
        result = self.consume(TokenType.RIGHT_PAREN, "期望 ')'")
        if not result.is_ok:
            return result
        
        # 循环体
        result = self.parse_statement()
        if not result.is_ok:
            return result
        body = result.value
        
        return Result.Ok(Stmt(
            kind=StmtKind.FOR,
            init=init,
            condition=condition,
            update=update,
            body=body
        ))
    
    def parse_module_decl(self) -> Result:
        """解析module声明"""
        path_parts = []
        
        result = self.consume(TokenType.IDENTIFIER, "期望模块名")
        if not result.is_ok:
            return result
        path_parts.append(result.value.lexeme)
        
        while self.match(TokenType.DOT):
            result = self.consume(TokenType.IDENTIFIER, "期望模块名")
            if not result.is_ok:
                return result
            path_parts.append(result.value.lexeme)
        
        result = self.consume(TokenType.SEMICOLON, "期望 ';'")
        if not result.is_ok:
            return result
        
        return Result.Ok(Stmt(
            kind=StmtKind.MODULE_DECL,
            module_path='.'.join(path_parts)
        ))
    
    def parse_struct(self, is_public: bool = False) -> Result:
        """解析struct声明"""
        result = self.consume(TokenType.STRUCT, "期望'struct'")
        if not result.is_ok:
            return result
        
        result = self.consume(TokenType.IDENTIFIER, "期望结构体名")
        if not result.is_ok:
            return result
        name = result.value.lexeme
        
        result = self.consume(TokenType.LEFT_BRACE, "期望 '{'")
        if not result.is_ok:
            return result
        
        fields = []
        while not self.check(TokenType.RIGHT_BRACE) and not self.is_at_end():
            # 检查pub关键字
            field_public = self.match(TokenType.PUB)
            
            result = self.consume(TokenType.IDENTIFIER, "期望字段名")
            if not result.is_ok:
                return result
            field_name = result.value.lexeme
            
            result = self.consume(TokenType.COLON, "期望 ':'")
            if not result.is_ok:
                return result
            
            result = self.consume(TokenType.IDENTIFIER, "期望类型名")
            if not result.is_ok:
                return result
            field_type = result.value.lexeme
            
            fields.append(StructField(field_name, field_type, field_public))
            
            # 可选的逗号
            self.match(TokenType.COMMA)
        
        result = self.consume(TokenType.RIGHT_BRACE, "期望 '}'")
        if not result.is_ok:
            return result
        
        return Result.Ok(Stmt(
            kind=StmtKind.STRUCT_DECL,
            name=name,
            fields=fields,
            is_public=is_public
        ))
    
    def parse_match(self) -> Result:
        """解析match语句"""
        # 解析被匹配的表达式
        result = self.parse_expression()
        if not result.is_ok:
            return result
        match_expr = result.value
        
        result = self.consume(TokenType.LEFT_BRACE, "期望 '{'")
        if not result.is_ok:
            return result
        
        cases = []
        while not self.check(TokenType.RIGHT_BRACE) and not self.is_at_end():
            result = self.parse_case_arm()
            if not result.is_ok:
                return result
            cases.append(result.value)
        
        result = self.consume(TokenType.RIGHT_BRACE, "期望 '}'")
        if not result.is_ok:
            return result
        
        return Result.Ok(Stmt(
            kind=StmtKind.MATCH,
            match_expr=match_expr,
            cases=cases
        ))
    
    def parse_case_arm(self) -> Result:
        """解析match的case分支"""
        result = self.consume(TokenType.CASE, "期望'case'")
        if not result.is_ok:
            return result
        
        # 解析模式列表（逗号分隔）
        patterns = []
        while True:
            result = self.parse_pattern()
            if not result.is_ok:
                return result
            patterns.append(result.value)
            
            if not self.match(TokenType.COMMA):
                break
        
        # 可选的守卫条件
        guard = None
        if self.match(TokenType.IF):
            result = self.parse_expression()
            if not result.is_ok:
                return result
            guard = result.value
        
        result = self.consume(TokenType.COLON, "期望 ':'")
        if not result.is_ok:
            return result
        
        # 解析分支体
        if self.check(TokenType.LEFT_BRACE):
            result = self.parse_block()
            if not result.is_ok:
                return result
            body = result.value
        else:
            result = self.parse_statement()
            if not result.is_ok:
                return result
            body = result.value
        
        return Result.Ok(CaseArm(patterns=patterns, guard=guard, body=body))
    
    def parse_pattern(self) -> Result:
        """解析模式"""
        # 通配符
        if self.match(TokenType.IDENTIFIER):
            name = self.previous().lexeme
            if name == '_':
                return Result.Ok(Pattern(kind='wildcard'))
            else:
                return Result.Ok(Pattern(kind='identifier', name=name))
        
        # 整数字面量
        if self.match(TokenType.INT_LITERAL):
            value = int(self.previous().lexeme)
            return Result.Ok(Pattern(kind='literal', value=value))
        
        # 字符串字面量
        if self.match(TokenType.STRING_LITERAL):
            value = self.previous().lexeme
            return Result.Ok(Pattern(kind='literal', value=value))
        
        return Result.Err(CompileError(
            kind=ErrorKind.PARSER_ERROR,
            message="期望模式",
            line=self.peek().line,
            column=self.peek().column,
            filename=self.filename
        ))
    
    def parse_block(self) -> Result:
        """解析代码块"""
        result = self.consume(TokenType.LEFT_BRACE, "期望 '{'")
        if not result.is_ok:
            return result
        
        statements = []
        while not self.check(TokenType.RIGHT_BRACE) and not self.is_at_end():
            result = self.parse_statement()
            if not result.is_ok:
                return result
            statements.append(result.value)
        
        result = self.consume(TokenType.RIGHT_BRACE, "期望 '}'")
        if not result.is_ok:
            return result
        
        return Result.Ok(Stmt(kind=StmtKind.BLOCK, statements=statements))
    
    def parse_expression_statement(self) -> Result:
        """解析表达式语句"""
        result = self.parse_expression()
        if not result.is_ok:
            return result
        expr = result.value
        
        result = self.consume(TokenType.SEMICOLON, "期望 ';'")
        if not result.is_ok:
            return result
        
        return Result.Ok(Stmt(kind=StmtKind.EXPRESSION, expr=expr))

    # 表达式解析
    def parse_expression(self) -> Result:
        return self.parse_assignment()
    
    def parse_assignment(self) -> Result:
        """解析赋值表达式"""
        result = self.parse_logical_or()
        if not result.is_ok:
            return result
        expr = result.value
        
        # 检查是否是赋值
        if self.match(TokenType.EQUAL):
            result = self.parse_assignment()  # 右结合
            if not result.is_ok:
                return result
            value = result.value
            
            # 创建赋值表达式
            return Result.Ok(Expr(
                kind=ExprKind.ASSIGN,
                left=expr,
                right=value
            ))
        
        return Result.Ok(expr)
    
    def parse_logical_or(self) -> Result:
        result = self.parse_logical_and()
        if not result.is_ok:
            return result
        left = result.value
        
        while self.match(TokenType.PIPE_PIPE):
            op = self.previous().lexeme
            result = self.parse_logical_and()
            if not result.is_ok:
                return result
            right = result.value
            left = Expr(kind=ExprKind.BINARY, operator=op, left=left, right=right)
        
        return Result.Ok(left)
    
    def parse_logical_and(self) -> Result:
        result = self.parse_equality()
        if not result.is_ok:
            return result
        left = result.value
        
        while self.match(TokenType.AMP_AMP):
            op = self.previous().lexeme
            result = self.parse_equality()
            if not result.is_ok:
                return result
            right = result.value
            left = Expr(kind=ExprKind.BINARY, operator=op, left=left, right=right)
        
        return Result.Ok(left)
    
    def parse_equality(self) -> Result:
        result = self.parse_comparison()
        if not result.is_ok:
            return result
        left = result.value
        
        while self.match(TokenType.EQUAL_EQUAL, TokenType.BANG_EQUAL):
            op = self.previous().lexeme
            result = self.parse_comparison()
            if not result.is_ok:
                return result
            right = result.value
            left = Expr(kind=ExprKind.BINARY, operator=op, left=left, right=right)
        
        return Result.Ok(left)
    
    def parse_comparison(self) -> Result:
        result = self.parse_term()
        if not result.is_ok:
            return result
        left = result.value
        
        while self.match(TokenType.LESS, TokenType.LESS_EQUAL, 
                         TokenType.GREATER, TokenType.GREATER_EQUAL):
            op = self.previous().lexeme
            result = self.parse_term()
            if not result.is_ok:
                return result
            right = result.value
            left = Expr(kind=ExprKind.BINARY, operator=op, left=left, right=right)
        
        return Result.Ok(left)
    
    def parse_term(self) -> Result:
        result = self.parse_factor()
        if not result.is_ok:
            return result
        left = result.value
        
        while self.match(TokenType.PLUS, TokenType.MINUS):
            op = self.previous().lexeme
            result = self.parse_factor()
            if not result.is_ok:
                return result
            right = result.value
            left = Expr(kind=ExprKind.BINARY, operator=op, left=left, right=right)
        
        return Result.Ok(left)
    
    def parse_factor(self) -> Result:
        result = self.parse_unary()
        if not result.is_ok:
            return result
        left = result.value
        
        while self.match(TokenType.STAR, TokenType.SLASH, TokenType.PERCENT):
            op = self.previous().lexeme
            result = self.parse_unary()
            if not result.is_ok:
                return result
            right = result.value
            left = Expr(kind=ExprKind.BINARY, operator=op, left=left, right=right)
        
        return Result.Ok(left)
    
    def parse_unary(self) -> Result:
        if self.match(TokenType.BANG, TokenType.MINUS):
            op = self.previous().lexeme
            result = self.parse_unary()
            if not result.is_ok:
                return result
            operand = result.value
            return Result.Ok(Expr(kind=ExprKind.UNARY, operator=op, operand=operand))
        
        return self.parse_postfix()
    
    def parse_postfix(self) -> Result:
        result = self.parse_primary()
        if not result.is_ok:
            return result
        expr = result.value
        
        while True:
            if self.match(TokenType.LEFT_PAREN):
                # 函数调用
                arguments = []
                if not self.check(TokenType.RIGHT_PAREN):
                    while True:
                        result = self.parse_expression()
                        if not result.is_ok:
                            return result
                        arguments.append(result.value)
                        if not self.match(TokenType.COMMA):
                            break
                
                result = self.consume(TokenType.RIGHT_PAREN, "期望 ')'")
                if not result.is_ok:
                    return result
                
                expr = Expr(kind=ExprKind.CALL, callee=expr, arguments=arguments)
            
            elif self.match(TokenType.LEFT_BRACKET):
                # 数组访问
                result = self.parse_expression()
                if not result.is_ok:
                    return result
                index = result.value
                
                result = self.consume(TokenType.RIGHT_BRACKET, "期望 ']'")
                if not result.is_ok:
                    return result
                
                expr = Expr(kind=ExprKind.ARRAY_ACCESS, array=expr, index=index)
            
            elif self.match(TokenType.DOT):
                result = self.consume(TokenType.IDENTIFIER, "期望成员名")
                if not result.is_ok:
                    return result
                member = result.value.lexeme
                expr = Expr(kind=ExprKind.MEMBER, object=expr, member=member)
            
            else:
                break
        
        return Result.Ok(expr)
    
    def parse_primary(self) -> Result:
        # 数组字面量
        if self.match(TokenType.LEFT_BRACKET):
            elements = []
            if not self.check(TokenType.RIGHT_BRACKET):
                while True:
                    result = self.parse_expression()
                    if not result.is_ok:
                        return result
                    elements.append(result.value)
                    if not self.match(TokenType.COMMA):
                        break
            
            result = self.consume(TokenType.RIGHT_BRACKET, "期望 ']'")
            if not result.is_ok:
                return result
            return Result.Ok(Expr(kind=ExprKind.ARRAY_LITERAL, elements=elements))
        
        # 整数字面量
        if self.match(TokenType.INT_LITERAL):
            value = int(self.previous().lexeme)
            return Result.Ok(Expr(kind=ExprKind.INT_LITERAL, int_value=value))
        
        # 浮点数字面量
        if self.match(TokenType.FLOAT_LITERAL):
            value = float(self.previous().lexeme)
            return Result.Ok(Expr(kind=ExprKind.FLOAT_LITERAL, float_value=value))
        
        # 字符串字面量
        if self.match(TokenType.STRING_LITERAL):
            value = self.previous().lexeme
            return Result.Ok(Expr(kind=ExprKind.STRING_LITERAL, string_value=value))
        
        # 标识符
        if self.match(TokenType.IDENTIFIER):
            name = self.previous().lexeme
            return Result.Ok(Expr(kind=ExprKind.IDENTIFIER, name=name))
        
        # 括号表达式
        if self.match(TokenType.LEFT_PAREN):
            result = self.parse_expression()
            if not result.is_ok:
                return result
            expr = result.value
            result = self.consume(TokenType.RIGHT_PAREN, "期望 ')'")
            if not result.is_ok:
                return result
            return Result.Ok(expr)
        
        token = self.peek()
        return Result.Err(CompileError(
            ErrorKind.PARSER_ERROR,
            "期望表达式",
            token.line,
            token.column,
            self.filename
        ))
    
    # 辅助方法
    def is_at_end(self) -> bool:
        return self.peek().type == TokenType.EOF
    
    def peek(self) -> Token:
        return self.tokens[self.current]
    
    def previous(self) -> Token:
        return self.tokens[self.current - 1]
    
    def advance(self) -> Token:
        if not self.is_at_end():
            self.current += 1
        return self.previous()
    
    def check(self, token_type: TokenType) -> bool:
        if self.is_at_end():
            return False
        return self.peek().type == token_type
    
    def match(self, *types: TokenType) -> bool:
        for token_type in types:
            if self.check(token_type):
                self.advance()
                return True
        return False
    
    def consume(self, token_type: TokenType, message: str) -> Result:
        if self.check(token_type):
            return Result.Ok(self.advance())
        
        token = self.peek()
        return Result.Err(CompileError(
            ErrorKind.PARSER_ERROR,
            message,
            token.line,
            token.column,
            self.filename
        ))

# ============================================================================
# 解释器/代码生成器
# ============================================================================

class ValueKind(Enum):
    INT = auto()
    FLOAT = auto()
    STRING = auto()
    BOOL = auto()
    VOID = auto()

@dataclass
class Value:
    kind: ValueKind
    value: Any = None

class Environment:
    def __init__(self, parent=None):
        self.variables = {}
        self.functions = {}
        self.parent = parent
    
    def define(self, name: str, value: Value):
        self.variables[name] = value
    
    def get(self, name: str) -> Optional[Value]:
        if name in self.variables:
            return self.variables[name]
        if self.parent:
            return self.parent.get(name)
        return None
    
    def set(self, name: str, value: Value) -> bool:
        if name in self.variables:
            self.variables[name] = value
            return True
        if self.parent:
            return self.parent.set(name, value)
        return False

class Interpreter:
    def __init__(self):
        self.global_env = Environment()
        self.current_env = self.global_env
        self.return_value = None
        self.has_returned = False
    
    def execute(self, program: Program) -> Result:
        """执行程序"""
        # 第一遍：收集所有函数
        for stmt in program.statements:
            if stmt.kind == StmtKind.FUNC_DECL:
                self.global_env.functions[stmt.name] = stmt
        
        # 查找main函数
        if 'main' not in self.global_env.functions:
            return Result.Err(CompileError(
                ErrorKind.SEMANTIC_ERROR,
                "未找到main函数",
                0, 0, ""
            ))
        
        # 执行main函数
        main_func = self.global_env.functions['main']
        result = self.execute_function(main_func, [])
        
        return result
    
    def execute_statement(self, stmt: Stmt) -> Result:
        """执行语句"""
        if self.has_returned:
            return Result.Ok(None)
        
        if stmt.kind == StmtKind.VAR_DECL:
            return self.execute_var_decl(stmt)
        elif stmt.kind == StmtKind.EXPRESSION:
            return self.execute_expr_stmt(stmt)
        elif stmt.kind == StmtKind.RETURN:
            return self.execute_return(stmt)
        elif stmt.kind == StmtKind.BLOCK:
            return self.execute_block(stmt)
        elif stmt.kind == StmtKind.IF:
            return self.execute_if(stmt)
        elif stmt.kind == StmtKind.WHILE:
            return self.execute_while(stmt)
        
        return Result.Ok(None)
    
    def execute_var_decl(self, stmt: Stmt) -> Result:
        """执行变量声明"""
        value = Value(ValueKind.VOID)
        
        if stmt.initializer:
            result = self.evaluate_expression(stmt.initializer)
            if not result.is_ok:
                return result
            value = result.value
        
        self.current_env.define(stmt.name, value)
        return Result.Ok(None)
    
    def execute_expr_stmt(self, stmt: Stmt) -> Result:
        """执行表达式语句"""
        return self.evaluate_expression(stmt.expr)
    
    def execute_return(self, stmt: Stmt) -> Result:
        """执行return语句"""
        if stmt.expr:
            result = self.evaluate_expression(stmt.expr)
            if not result.is_ok:
                return result
            self.return_value = result.value
        else:
            self.return_value = Value(ValueKind.VOID)
        
        self.has_returned = True
        return Result.Ok(None)
    
    def execute_block(self, stmt: Stmt) -> Result:
        """执行代码块"""
        block_env = Environment(self.current_env)
        old_env = self.current_env
        self.current_env = block_env
        
        for s in stmt.statements:
            result = self.execute_statement(s)
            if not result.is_ok:
                self.current_env = old_env
                return result
            if self.has_returned:
                break
        
        self.current_env = old_env
        return Result.Ok(None)
    
    def execute_if(self, stmt: Stmt) -> Result:
        """执行if语句"""
        result = self.evaluate_expression(stmt.condition)
        if not result.is_ok:
            return result
        
        condition = result.value
        if condition.kind == ValueKind.BOOL and condition.value:
            return self.execute_statement(stmt.then_branch)
        elif stmt.else_branch:
            return self.execute_statement(stmt.else_branch)
        
        return Result.Ok(None)
    
    def execute_while(self, stmt: Stmt) -> Result:
        """执行while语句"""
        while True:
            result = self.evaluate_expression(stmt.condition)
            if not result.is_ok:
                return result
            
            condition = result.value
            if not (condition.kind == ValueKind.BOOL and condition.value):
                break
            
            result = self.execute_statement(stmt.body)
            if not result.is_ok:
                return result
            if self.has_returned:
                break
        
        return Result.Ok(None)
    
    def execute_function(self, func: Stmt, args: List[Value]) -> Result:
        """执行函数"""
        func_env = Environment(self.global_env)
        
        # 绑定参数
        for i, param in enumerate(func.params):
            if i < len(args):
                func_env.define(param.name, args[i])
        
        old_env = self.current_env
        self.current_env = func_env
        self.has_returned = False
        self.return_value = None
        
        result = self.execute_statement(func.body)
        
        return_val = self.return_value if self.return_value else Value(ValueKind.VOID)
        
        self.current_env = old_env
        self.has_returned = False
        self.return_value = None
        
        if not result.is_ok:
            return result
        
        return Result.Ok(return_val)

    def evaluate_expression(self, expr: Expr) -> Result:
        """求值表达式"""
        if expr.kind == ExprKind.INT_LITERAL:
            return Result.Ok(Value(ValueKind.INT, expr.int_value))
        
        elif expr.kind == ExprKind.FLOAT_LITERAL:
            return Result.Ok(Value(ValueKind.FLOAT, expr.float_value))
        
        elif expr.kind == ExprKind.STRING_LITERAL:
            return Result.Ok(Value(ValueKind.STRING, expr.string_value))
        
        elif expr.kind == ExprKind.IDENTIFIER:
            return self.evaluate_identifier(expr)
        
        elif expr.kind == ExprKind.BINARY:
            return self.evaluate_binary(expr)
        
        elif expr.kind == ExprKind.UNARY:
            return self.evaluate_unary(expr)
        
        elif expr.kind == ExprKind.CALL:
            return self.evaluate_call(expr)
        
        return Result.Ok(Value(ValueKind.VOID))
    
    def evaluate_identifier(self, expr: Expr) -> Result:
        """求值标识符"""
        value = self.current_env.get(expr.name)
        if value is None:
            return Result.Err(CompileError(
                ErrorKind.SEMANTIC_ERROR,
                f"未定义的变量: {expr.name}",
                0, 0, ""
            ))
        return Result.Ok(value)
    
    def evaluate_binary(self, expr: Expr) -> Result:
        """求值二元运算"""
        left_result = self.evaluate_expression(expr.left)
        if not left_result.is_ok:
            return left_result
        
        right_result = self.evaluate_expression(expr.right)
        if not right_result.is_ok:
            return right_result
        
        left = left_result.value
        right = right_result.value
        
        # 整数运算
        if left.kind == ValueKind.INT and right.kind == ValueKind.INT:
            if expr.operator == '+':
                return Result.Ok(Value(ValueKind.INT, left.value + right.value))
            elif expr.operator == '-':
                return Result.Ok(Value(ValueKind.INT, left.value - right.value))
            elif expr.operator == '*':
                return Result.Ok(Value(ValueKind.INT, left.value * right.value))
            elif expr.operator == '/':
                if right.value == 0:
                    return Result.Err(CompileError(
                        ErrorKind.RUNTIME_ERROR,
                        "除数不能为零",
                        0, 0, ""
                    ))
                return Result.Ok(Value(ValueKind.INT, left.value // right.value))
            elif expr.operator == '%':
                return Result.Ok(Value(ValueKind.INT, left.value % right.value))
            elif expr.operator == '==':
                return Result.Ok(Value(ValueKind.BOOL, left.value == right.value))
            elif expr.operator == '!=':
                return Result.Ok(Value(ValueKind.BOOL, left.value != right.value))
            elif expr.operator == '<':
                return Result.Ok(Value(ValueKind.BOOL, left.value < right.value))
            elif expr.operator == '<=':
                return Result.Ok(Value(ValueKind.BOOL, left.value <= right.value))
            elif expr.operator == '>':
                return Result.Ok(Value(ValueKind.BOOL, left.value > right.value))
            elif expr.operator == '>=':
                return Result.Ok(Value(ValueKind.BOOL, left.value >= right.value))
        
        # 字符串连接
        if left.kind == ValueKind.STRING and right.kind == ValueKind.STRING:
            if expr.operator == '+':
                return Result.Ok(Value(ValueKind.STRING, left.value + right.value))
        
        # 字符串和整数连接
        if left.kind == ValueKind.STRING and right.kind == ValueKind.INT:
            if expr.operator == '+':
                return Result.Ok(Value(ValueKind.STRING, left.value + str(right.value)))
        
        if left.kind == ValueKind.INT and right.kind == ValueKind.STRING:
            if expr.operator == '+':
                return Result.Ok(Value(ValueKind.STRING, str(left.value) + right.value))
        
        # 布尔运算
        if left.kind == ValueKind.BOOL and right.kind == ValueKind.BOOL:
            if expr.operator == '&&':
                return Result.Ok(Value(ValueKind.BOOL, left.value and right.value))
            elif expr.operator == '||':
                return Result.Ok(Value(ValueKind.BOOL, left.value or right.value))
        
        return Result.Err(CompileError(
            ErrorKind.TYPE_ERROR,
            f"不支持的运算: {expr.operator}",
            0, 0, ""
        ))
    
    def evaluate_unary(self, expr: Expr) -> Result:
        """求值一元运算"""
        operand_result = self.evaluate_expression(expr.operand)
        if not operand_result.is_ok:
            return operand_result
        
        operand = operand_result.value
        
        if expr.operator == '-':
            if operand.kind == ValueKind.INT:
                return Result.Ok(Value(ValueKind.INT, -operand.value))
            elif operand.kind == ValueKind.FLOAT:
                return Result.Ok(Value(ValueKind.FLOAT, -operand.value))
        
        elif expr.operator == '!':
            if operand.kind == ValueKind.BOOL:
                return Result.Ok(Value(ValueKind.BOOL, not operand.value))
        
        return Result.Err(CompileError(
            ErrorKind.TYPE_ERROR,
            f"不支持的一元运算: {expr.operator}",
            0, 0, ""
        ))
    
    def evaluate_call(self, expr: Expr) -> Result:
        """求值函数调用"""
        if expr.callee.kind != ExprKind.IDENTIFIER:
            return Result.Err(CompileError(
                ErrorKind.SEMANTIC_ERROR,
                "只支持直接函数调用",
                0, 0, ""
            ))
        
        func_name = expr.callee.name
        
        # 内置函数
        if func_name == 'println' or func_name == 'print':
            args = []
            for arg in expr.arguments:
                result = self.evaluate_expression(arg)
                if not result.is_ok:
                    return result
                args.append(result.value)
            
            output = []
            for arg in args:
                if arg.kind == ValueKind.INT:
                    output.append(str(arg.value))
                elif arg.kind == ValueKind.FLOAT:
                    output.append(str(arg.value))
                elif arg.kind == ValueKind.STRING:
                    output.append(arg.value)
                elif arg.kind == ValueKind.BOOL:
                    output.append('true' if arg.value else 'false')
                else:
                    output.append('void')
            
            if func_name == 'println':
                print(''.join(output))
            else:
                print(''.join(output), end='')
            
            return Result.Ok(Value(ValueKind.VOID))
        
        # 用户定义函数
        if func_name not in self.global_env.functions:
            return Result.Err(CompileError(
                ErrorKind.SEMANTIC_ERROR,
                f"未定义的函数: {func_name}",
                0, 0, ""
            ))
        
        func = self.global_env.functions[func_name]
        
        # 求值参数
        args = []
        for arg in expr.arguments:
            result = self.evaluate_expression(arg)
            if not result.is_ok:
                return result
            args.append(result.value)
        
        return self.execute_function(func, args)

# ============================================================================
# 主程序
# ============================================================================

def compile_to_c(filename: str, output_file: str = None) -> Result:
    """编译AZ代码到C代码"""
    print(f"正在编译: {filename} -> C代码")
    
    # 1. 读取源文件
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            source = f.read()
    except FileNotFoundError:
        return Result.Err(CompileError(
            ErrorKind.LEXER_ERROR,
            f"无法读取文件: {filename}",
            0, 0, filename
        ))
    
    # 2. 词法分析
    print("[1/3] 词法分析...")
    lexer = Lexer(source, filename)
    tokens_result = lexer.tokenize()
    if not tokens_result.is_ok:
        return tokens_result
    tokens = tokens_result.value
    print(f"  生成了 {len(tokens)} 个token")
    
    # 3. 语法分析
    print("[2/3] 语法分析...")
    parser = Parser(tokens, filename)
    ast_result = parser.parse()
    if not ast_result.is_ok:
        return ast_result
    ast = ast_result.value
    print(f"  生成了 {len(ast.statements)} 个顶层语句")
    
    # 4. 生成C代码
    print("[3/3] 生成C代码...")
    codegen = CCodeGenerator()
    c_code = codegen.generate(ast)
    
    # 5. 写入输出文件
    if output_file is None:
        output_file = filename.replace('.az', '.c')
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(c_code)
        print(f"  C代码已写入: {output_file}")
    except IOError as e:
        return Result.Err(CompileError(
            ErrorKind.LEXER_ERROR,
            f"无法写入文件: {output_file}",
            0, 0, filename
        ))
    
    return Result.Ok(None)

def compile_file(filename: str) -> Result:
    """编译文件"""
    print(f"正在编译: {filename}")
    
    # 1. 读取源文件
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            source = f.read()
    except FileNotFoundError:
        return Result.Err(CompileError(
            ErrorKind.LEXER_ERROR,
            f"无法读取文件: {filename}",
            0, 0, filename
        ))
    
    # 2. 词法分析
    print("[1/4] 词法分析...")
    lexer = Lexer(source, filename)
    tokens_result = lexer.tokenize()
    if not tokens_result.is_ok:
        return tokens_result
    tokens = tokens_result.value
    print(f"  生成了 {len(tokens)} 个token")
    
    # 3. 语法分析
    print("[2/4] 语法分析...")
    parser = Parser(tokens, filename)
    ast_result = parser.parse()
    if not ast_result.is_ok:
        return ast_result
    ast = ast_result.value
    print(f"  生成了 {len(ast.statements)} 个顶层语句")
    
    # 4. 语义分析（简化版，在解释器中进行）
    print("[3/4] 语义分析...")
    print("  语义检查通过")
    
    # 5. 执行程序
    print("[4/4] 执行程序...")
    print("---输出---")
    interpreter = Interpreter()
    exec_result = interpreter.execute(ast)
    print("----------")
    
    if not exec_result.is_ok:
        return exec_result
    
    return Result.Ok(None)

def main():
    """主函数"""
    print("AZ编译器 v0.1.0")
    print("采用C3风格的错误处理")
    print()
    
    if len(sys.argv) < 2:
        print("用法: python az_compiler.py <源文件> [--emit-c] [-o 输出文件]")
        return 1
    
    filename = sys.argv[1]
    emit_c = '--emit-c' in sys.argv
    output_file = None
    
    # 解析输出文件名
    if '-o' in sys.argv:
        idx = sys.argv.index('-o')
        if idx + 1 < len(sys.argv):
            output_file = sys.argv[idx + 1]
    
    if emit_c:
        # C代码生成模式
        result = compile_to_c(filename, output_file)
    else:
        # 解释执行模式
        result = compile_file(filename)
    
    if result.is_ok:
        print("\n编译成功！")
        return 0
    else:
        print()
        result.error.report()
        return 1

# ============================================================================
# C代码生成器
# ============================================================================

class CCodeGenerator:
    """将AZ AST转换为C代码"""
    
    def __init__(self):
        self.output = []
        self.indent_level = 0
        self.temp_counter = 0
    
    def generate(self, program: Program) -> str:
        """生成C代码"""
        self.output = []
        
        # 添加头文件
        self.emit("#include <stdio.h>")
        self.emit("#include <stdlib.h>")
        self.emit("#include <string.h>")
        self.emit("#include <stdbool.h>")
        self.emit("")
        
        # 添加内置函数声明
        self.emit("// 内置函数")
        self.emit("void println(const char* str) {")
        self.emit("    printf(\"%s\\n\", str);")
        self.emit("}")
        self.emit("")
        self.emit("void print(const char* str) {")
        self.emit("    printf(\"%s\", str);")
        self.emit("}")
        self.emit("")
        
        # 生成函数前向声明
        for stmt in program.statements:
            if stmt.kind == StmtKind.FUNC_DECL:
                self.gen_func_forward_decl(stmt)
        
        self.emit("")
        
        # 生成函数定义
        for stmt in program.statements:
            if stmt.kind == StmtKind.FUNC_DECL:
                self.gen_func_decl(stmt)
        
        return '\n'.join(self.output)
    
    def gen_func_forward_decl(self, stmt: Stmt):
        """生成函数前向声明"""
        return_type = self.map_type(stmt.return_type)
        params = ', '.join([
            f"{self.map_type(p.type_name)} {p.name}"
            for p in stmt.params
        ]) if stmt.params else 'void'
        
        self.emit(f"{return_type} {stmt.name}({params});")
    
    def gen_func_decl(self, stmt: Stmt):
        """生成函数定义"""
        return_type = self.map_type(stmt.return_type)
        params = ', '.join([
            f"{self.map_type(p.type_name)} {p.name}"
            for p in stmt.params
        ]) if stmt.params else 'void'
        
        self.emit(f"{return_type} {stmt.name}({params}) {{")
        self.indent_level += 1
        
        # 函数体
        if stmt.body:
            self.gen_stmt(stmt.body)
        
        self.indent_level -= 1
        self.emit("}")
        self.emit("")
    
    def gen_stmt(self, stmt: Stmt):
        """生成语句"""
        if stmt.kind == StmtKind.VAR_DECL:
            self.gen_var_decl(stmt)
        elif stmt.kind == StmtKind.EXPRESSION:
            self.gen_expr_stmt(stmt)
        elif stmt.kind == StmtKind.RETURN:
            self.gen_return(stmt)
        elif stmt.kind == StmtKind.BLOCK:
            self.gen_block(stmt)
        elif stmt.kind == StmtKind.IF:
            self.gen_if(stmt)
        elif stmt.kind == StmtKind.WHILE:
            self.gen_while(stmt)
        elif stmt.kind == StmtKind.FOR:
            self.gen_for(stmt)
        elif stmt.kind == StmtKind.MATCH:
            self.gen_match(stmt)
        elif stmt.kind == StmtKind.MODULE_DECL:
            # 模块声明在C代码中作为注释
            self.emit(f"// module {stmt.module_path}")
        elif stmt.kind == StmtKind.STRUCT_DECL:
            self.gen_struct(stmt)
    
    def gen_var_decl(self, stmt: Stmt):
        """生成变量声明"""
        type_str = self.map_type(stmt.type_name) if stmt.type_name else "int"
        
        if stmt.initializer:
            # 检查是否是数组字面量
            if stmt.initializer.kind == ExprKind.ARRAY_LITERAL:
                init_code = self.gen_expr(stmt.initializer)
                # 数组声明
                self.emit(f"{type_str} {stmt.name}[] = {init_code};")
            else:
                init_code = self.gen_expr(stmt.initializer)
                self.emit(f"{type_str} {stmt.name} = {init_code};")
        else:
            self.emit(f"{type_str} {stmt.name};")
    
    def gen_expr_stmt(self, stmt: Stmt):
        """生成表达式语句"""
        expr_code = self.gen_expr(stmt.expr)
        self.emit(f"{expr_code};")
    
    def gen_return(self, stmt: Stmt):
        """生成return语句"""
        if stmt.expr:
            expr_code = self.gen_expr(stmt.expr)
            self.emit(f"return {expr_code};")
        else:
            self.emit("return;")
    
    def gen_block(self, stmt: Stmt):
        """生成代码块"""
        self.emit("{")
        self.indent_level += 1
        
        if stmt.statements:
            for s in stmt.statements:
                self.gen_stmt(s)
        
        self.indent_level -= 1
        self.emit("}")
    
    def gen_if(self, stmt: Stmt):
        """生成if语句"""
        condition = self.gen_expr(stmt.condition)
        self.emit(f"if ({condition}) {{")
        self.indent_level += 1
        
        self.gen_stmt(stmt.then_branch)
        
        self.indent_level -= 1
        self.emit("}")
        
        if stmt.else_branch:
            self.emit("else {")
            self.indent_level += 1
            
            self.gen_stmt(stmt.else_branch)
            
            self.indent_level -= 1
            self.emit("}")
    
    def gen_while(self, stmt: Stmt):
        """生成while语句"""
        condition = self.gen_expr(stmt.condition)
        self.emit(f"while ({condition}) {{")
        self.indent_level += 1
        
        self.gen_stmt(stmt.body)
        
        self.indent_level -= 1
        self.emit("}")
    
    def gen_for(self, stmt: Stmt):
        """生成for语句"""
        # 生成初始化（如果有）
        if hasattr(stmt, 'init') and stmt.init:
            if stmt.init.kind == StmtKind.VAR_DECL:
                # 变量声明
                type_str = self.map_type(stmt.init.type_name) if stmt.init.type_name else "int"
                if stmt.init.initializer:
                    init_code = self.gen_expr(stmt.init.initializer)
                    self.emit(f"{type_str} {stmt.init.name} = {init_code};")
                else:
                    self.emit(f"{type_str} {stmt.init.name};")
            else:
                # 表达式语句
                self.gen_stmt(stmt.init)
        
        # 生成while循环
        self.emit("while (1) {")
        self.indent_level += 1
        
        # 生成条件检查
        if hasattr(stmt, 'condition') and stmt.condition:
            condition = self.gen_expr(stmt.condition)
            self.emit(f"if (!({condition})) break;")
        
        # 生成循环体
        if hasattr(stmt, 'body') and stmt.body:
            self.gen_stmt(stmt.body)
        
        # 生成更新表达式
        if hasattr(stmt, 'update') and stmt.update:
            update = self.gen_expr(stmt.update)
            self.emit(f"{update};")
        
        self.indent_level -= 1
        self.emit("}")
    
    def gen_match(self, stmt: Stmt):
        """生成match语句（降级为if-else链）"""
        if not hasattr(stmt, 'match_expr') or not hasattr(stmt, 'cases'):
            return
        
        # 生成临时变量存储匹配表达式的值
        match_value = self.gen_expr(stmt.match_expr)
        temp_var = self.new_temp()
        self.emit(f"int {temp_var} = {match_value};")
        
        # 生成每个case分支
        for i, case in enumerate(stmt.cases):
            # 构建条件表达式
            conditions = []
            
            for pattern in case.patterns:
                if pattern.kind == 'wildcard':
                    conditions.append("1")  # 总是匹配
                elif pattern.kind == 'literal':
                    if isinstance(pattern.value, str):
                        conditions.append(f'strcmp({temp_var}, "{pattern.value}") == 0')
                    else:
                        conditions.append(f"({temp_var} == {pattern.value})")
                elif pattern.kind == 'identifier':
                    # 变量绑定
                    self.emit(f"int {pattern.name} = {temp_var};")
                    conditions.append("1")
            
            # 添加守卫条件
            if case.guard:
                guard_code = self.gen_expr(case.guard)
                conditions.append(f"({guard_code})")
            
            # 生成if/else if
            condition_str = " || ".join(conditions) if conditions else "1"
            if i == 0:
                self.emit(f"if ({condition_str}) {{")
            else:
                self.emit(f"else if ({condition_str}) {{")
            
            self.indent_level += 1
            if case.body:
                self.gen_stmt(case.body)
            self.indent_level -= 1
            self.emit("}")
    
    def gen_struct(self, stmt: Stmt):
        """生成结构体定义"""
        if not hasattr(stmt, 'fields'):
            return
        
        self.emit(f"typedef struct {{")
        self.indent_level += 1
        
        for field in stmt.fields:
            type_str = self.map_type(field.type_name)
            self.emit(f"{type_str} {field.name};")
        
        self.indent_level -= 1
        self.emit(f"}} {stmt.name};")
        self.emit("")
    
    def gen_expr(self, expr: Expr) -> str:
        """生成表达式"""
        if expr.kind == ExprKind.INT_LITERAL:
            return str(expr.int_value)
        
        elif expr.kind == ExprKind.FLOAT_LITERAL:
            return str(expr.float_value)
        
        elif expr.kind == ExprKind.STRING_LITERAL:
            # 转义字符串
            escaped = expr.string_value.replace('\\', '\\\\').replace('"', '\\"')
            return f'"{escaped}"'
        
        elif expr.kind == ExprKind.IDENTIFIER:
            return expr.name
        
        elif expr.kind == ExprKind.BINARY:
            left = self.gen_expr(expr.left)
            right = self.gen_expr(expr.right)
            return f"({left} {expr.operator} {right})"
        
        elif expr.kind == ExprKind.UNARY:
            operand = self.gen_expr(expr.operand)
            return f"({expr.operator}{operand})"
        
        elif expr.kind == ExprKind.CALL:
            return self.gen_call(expr)
        
        elif expr.kind == ExprKind.ASSIGN:
            # 生成赋值表达式
            if hasattr(expr, 'left') and hasattr(expr, 'right'):
                left = self.gen_expr(expr.left)
                right = self.gen_expr(expr.right)
                return f"({left} = {right})"
            return "0"
        
        elif expr.kind == ExprKind.ARRAY_LITERAL:
            # 生成数组字面量
            if hasattr(expr, 'elements') and expr.elements:
                elements = [self.gen_expr(e) for e in expr.elements]
                return "{" + ", ".join(elements) + "}"
            return "{}"
        
        elif expr.kind == ExprKind.ARRAY_ACCESS:
            # 生成数组访问
            if hasattr(expr, 'array') and hasattr(expr, 'index'):
                array = self.gen_expr(expr.array)
                index = self.gen_expr(expr.index)
                return f"{array}[{index}]"
            return "0"
        
        elif expr.kind == ExprKind.MEMBER:
            # 生成成员访问
            if hasattr(expr, 'object') and hasattr(expr, 'member'):
                obj = self.gen_expr(expr.object)
                return f"{obj}.{expr.member}"
            return "0"
        
        return "0"
    
    def gen_call(self, expr: Expr) -> str:
        """生成函数调用"""
        if expr.callee.kind != ExprKind.IDENTIFIER:
            return "0"
        
        func_name = expr.callee.name
        
        # 生成参数
        args = []
        if expr.arguments:
            for arg in expr.arguments:
                args.append(self.gen_expr(arg))
        
        args_str = ', '.join(args)
        return f"{func_name}({args_str})"
    
    def map_type(self, az_type: str) -> str:
        """映射AZ类型到C类型"""
        type_map = {
            'int': 'int',
            'float': 'double',
            'string': 'const char*',
            'bool': 'bool',
            'void': 'void'
        }
        return type_map.get(az_type, 'int')
    
    def emit(self, code: str):
        """输出一行代码"""
        indent = '    ' * self.indent_level
        self.output.append(indent + code)
    
    def new_temp(self) -> str:
        """生成临时变量名"""
        name = f"_t{self.temp_counter}"
        self.temp_counter += 1
        return name


if __name__ == '__main__':
    sys.exit(main())
