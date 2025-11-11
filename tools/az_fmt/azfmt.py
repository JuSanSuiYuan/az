#!/usr/bin/env python3
"""
AZ fmt - AZ语言代码格式化工具
类似于rustfmt，自动格式化AZ代码

用法:
    python tools/az_fmt/azfmt.py file.az                    # 格式化单个文件
    python tools/az_fmt/azfmt.py file1.az file2.az          # 格式化多个文件
    python tools/az_fmt/azfmt.py --check file.az            # 检查格式但不修改
    python tools/az_fmt/azfmt.py --config azfmt.toml file.az # 使用配置文件
    python tools/az_fmt/azfmt.py --help                     # 显示帮助
"""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

# 导入词法分析器
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "bootstrap"))
from az_compiler import Lexer, Token, TokenType


@dataclass
class FormatConfig:
    """格式化配置"""
    indent_size: int = 4
    max_line_length: int = 100
    use_spaces: bool = True
    space_before_brace: bool = True
    space_after_comma: bool = True
    space_around_operators: bool = True
    newline_before_brace: bool = False
    align_struct_fields: bool = True
    align_function_params: bool = False
    trailing_comma: bool = True
    
    @classmethod
    def from_file(cls, path: str) -> 'FormatConfig':
        """从配置文件加载"""
        # TODO: 实现TOML配置文件解析
        return cls()


class AZFormatter:
    """AZ代码格式化器"""
    
    def __init__(self, config: FormatConfig = None):
        self.config = config or FormatConfig()
        self.indent_level = 0
        self.output = []
        self.current_line = []
        self.line_length = 0
        
    def format_file(self, filepath: str) -> str:
        """格式化整个文件"""
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        
        return self.format_source(source)
    
    def format_source(self, source: str) -> str:
        """格式化源代码"""
        lexer = Lexer(source, "<format>")
        
        # 词法分析
        result = lexer.tokenize()
        if not result.is_ok:
            raise Exception(f"Lexer error: {result.error.message}")
        
        tokens = result.value
        
        # 格式化tokens
        self.output = []
        self.indent_level = 0
        self.format_tokens(tokens)
        
        return '\n'.join(self.output)
    
    def format_tokens(self, tokens: List[Token]):
        """格式化token列表"""
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            # 跳过EOF
            if hasattr(TokenType, 'EOF') and token.type == TokenType.EOF:
                break
            
            # 处理注释（如果存在）
            if hasattr(TokenType, 'COMMENT') and token.type == TokenType.COMMENT:
                self.add_comment(token.value)
                i += 1
                continue
            
            # 处理导入语句
            if hasattr(TokenType, 'IMPORT') and token.type == TokenType.IMPORT:
                i = self.format_import(tokens, i)
                continue
            
            # 处理函数定义
            if hasattr(TokenType, 'FN') and token.type == TokenType.FN:
                i = self.format_function(tokens, i)
                continue
            
            # 处理结构体定义
            if hasattr(TokenType, 'STRUCT') and token.type == TokenType.STRUCT:
                i = self.format_struct(tokens, i)
                continue
            
            # 处理枚举定义
            if hasattr(TokenType, 'ENUM') and token.type == TokenType.ENUM:
                i = self.format_enum(tokens, i)
                continue
            
            # 处理模块定义
            if hasattr(TokenType, 'MODULE') and token.type == TokenType.MODULE:
                i = self.format_module(tokens, i)
                continue
            
            i += 1
    
    def format_import(self, tokens: List[Token], start: int) -> int:
        """格式化import语句"""
        line = "import "
        i = start + 1
        
        # 收集导入路径
        while i < len(tokens) and tokens[i].type != TokenType.SEMICOLON:
            if tokens[i].type == TokenType.IDENTIFIER or tokens[i].type == TokenType.DOT:
                line += tokens[i].value
            i += 1
        
        line += ";"
        self.add_line(line)
        self.add_blank_line()
        
        return i + 1
    
    def format_function(self, tokens: List[Token], start: int) -> int:
        """格式化函数定义"""
        i = start
        line = ""
        
        # pub fn
        if i > 0 and tokens[i-1].type == TokenType.PUB:
            line = "pub "
        
        line += "fn "
        i += 1
        
        # 函数名
        if i < len(tokens) and tokens[i].type == TokenType.IDENTIFIER:
            line += tokens[i].value
            i += 1
        
        # 参数列表
        if i < len(tokens) and tokens[i].type == TokenType.LPAREN:
            params, i = self.format_parameters(tokens, i)
            line += params
        
        # 返回类型
        if i < len(tokens) and tokens[i].type == TokenType.IDENTIFIER:
            line += " " + tokens[i].value
            i += 1
        
        # 函数体
        if i < len(tokens) and tokens[i].type == TokenType.LBRACE:
            if self.config.space_before_brace:
                line += " {"
            else:
                line += "{"
            
            self.add_line(line)
            self.indent_level += 1
            
            # 格式化函数体
            i = self.format_block(tokens, i + 1)
            
            self.indent_level -= 1
            self.add_line("}")
            self.add_blank_line()
        
        return i
    
    def format_parameters(self, tokens: List[Token], start: int) -> tuple[str, int]:
        """格式化参数列表"""
        result = "("
        i = start + 1
        first = True
        
        while i < len(tokens) and tokens[i].type != TokenType.RPAREN:
            if tokens[i].type == TokenType.COMMA:
                if self.config.space_after_comma:
                    result += ", "
                else:
                    result += ","
                first = False
            elif tokens[i].type == TokenType.IDENTIFIER:
                if not first and not result.endswith(" "):
                    result += " "
                result += tokens[i].value
            elif tokens[i].type == TokenType.COLON:
                result += ": "
            i += 1
        
        result += ")"
        return result, i + 1
    
    def format_struct(self, tokens: List[Token], start: int) -> int:
        """格式化结构体定义"""
        i = start
        line = ""
        
        # pub struct
        if i > 0 and tokens[i-1].type == TokenType.PUB:
            line = "pub "
        
        line += "struct "
        i += 1
        
        # 结构体名
        if i < len(tokens) and tokens[i].type == TokenType.IDENTIFIER:
            line += tokens[i].value
            i += 1
        
        # 泛型参数
        if i < len(tokens) and tokens[i].type == TokenType.LT:
            generics, i = self.format_generics(tokens, i)
            line += generics
        
        # 结构体体
        if i < len(tokens) and tokens[i].type == TokenType.LBRACE:
            if self.config.space_before_brace:
                line += " {"
            else:
                line += "{"
            
            self.add_line(line)
            self.indent_level += 1
            
            # 格式化字段
            i = self.format_struct_fields(tokens, i + 1)
            
            self.indent_level -= 1
            self.add_line("}")
            self.add_blank_line()
        
        return i
    
    def format_struct_fields(self, tokens: List[Token], start: int) -> int:
        """格式化结构体字段"""
        i = start
        fields = []
        
        # 收集所有字段
        while i < len(tokens) and tokens[i].type != TokenType.RBRACE:
            if tokens[i].type == TokenType.IDENTIFIER:
                field_name = tokens[i].value
                i += 1
                
                if i < len(tokens) and tokens[i].type == TokenType.COLON:
                    i += 1
                    field_type = ""
                    
                    while i < len(tokens) and tokens[i].type not in [TokenType.COMMA, TokenType.RBRACE]:
                        field_type += tokens[i].value
                        i += 1
                    
                    fields.append((field_name, field_type))
                    
                    if i < len(tokens) and tokens[i].type == TokenType.COMMA:
                        i += 1
            else:
                i += 1
        
        # 对齐字段
        if self.config.align_struct_fields and fields:
            max_name_len = max(len(name) for name, _ in fields)
            for name, type_name in fields:
                line = f"{name.ljust(max_name_len)}: {type_name},"
                self.add_line(line)
        else:
            for name, type_name in fields:
                self.add_line(f"{name}: {type_name},")
        
        return i
    
    def format_enum(self, tokens: List[Token], start: int) -> int:
        """格式化枚举定义"""
        i = start
        line = ""
        
        # pub enum
        if i > 0 and tokens[i-1].type == TokenType.PUB:
            line = "pub "
        
        line += "enum "
        i += 1
        
        # 枚举名
        if i < len(tokens) and tokens[i].type == TokenType.IDENTIFIER:
            line += tokens[i].value
            i += 1
        
        # 泛型参数
        if i < len(tokens) and tokens[i].type == TokenType.LT:
            generics, i = self.format_generics(tokens, i)
            line += generics
        
        # 枚举体
        if i < len(tokens) and tokens[i].type == TokenType.LBRACE:
            if self.config.space_before_brace:
                line += " {"
            else:
                line += "{"
            
            self.add_line(line)
            self.indent_level += 1
            
            # 格式化枚举变体
            i = self.format_enum_variants(tokens, i + 1)
            
            self.indent_level -= 1
            self.add_line("}")
            self.add_blank_line()
        
        return i
    
    def format_enum_variants(self, tokens: List[Token], start: int) -> int:
        """格式化枚举变体"""
        i = start
        
        while i < len(tokens) and tokens[i].type != TokenType.RBRACE:
            if tokens[i].type == TokenType.IDENTIFIER:
                variant = tokens[i].value
                i += 1
                
                # 带参数的变体
                if i < len(tokens) and tokens[i].type == TokenType.LPAREN:
                    params, i = self.format_parameters(tokens, i)
                    variant += params
                
                variant += ","
                self.add_line(variant)
                
                if i < len(tokens) and tokens[i].type == TokenType.COMMA:
                    i += 1
            else:
                i += 1
        
        return i
    
    def format_module(self, tokens: List[Token], start: int) -> int:
        """格式化模块定义"""
        line = "module "
        i = start + 1
        
        # 收集模块路径
        while i < len(tokens) and tokens[i].type != TokenType.SEMICOLON:
            if tokens[i].type == TokenType.IDENTIFIER or tokens[i].type == TokenType.DOT:
                line += tokens[i].value
            i += 1
        
        line += ";"
        self.add_line(line)
        self.add_blank_line()
        
        return i + 1
    
    def format_block(self, tokens: List[Token], start: int) -> int:
        """格式化代码块"""
        i = start
        brace_count = 1
        
        while i < len(tokens) and brace_count > 0:
            token = tokens[i]
            
            if token.type == TokenType.LBRACE:
                brace_count += 1
            elif token.type == TokenType.RBRACE:
                brace_count -= 1
                if brace_count == 0:
                    break
            
            # 格式化语句
            if token.type in [TokenType.LET, TokenType.VAR]:
                i = self.format_variable_declaration(tokens, i)
            elif token.type == TokenType.RETURN:
                i = self.format_return_statement(tokens, i)
            elif token.type == TokenType.IF:
                i = self.format_if_statement(tokens, i)
            elif token.type == TokenType.WHILE:
                i = self.format_while_statement(tokens, i)
            elif token.type == TokenType.FOR:
                i = self.format_for_statement(tokens, i)
            elif token.type == TokenType.MATCH:
                i = self.format_match_statement(tokens, i)
            else:
                i += 1
        
        return i
    
    def format_variable_declaration(self, tokens: List[Token], start: int) -> int:
        """格式化变量声明"""
        line = tokens[start].value + " "
        i = start + 1
        
        # 变量名
        if i < len(tokens) and tokens[i].type == TokenType.IDENTIFIER:
            line += tokens[i].value
            i += 1
        
        # 类型注解
        if i < len(tokens) and tokens[i].type == TokenType.COLON:
            line += ": "
            i += 1
            
            while i < len(tokens) and tokens[i].type not in [TokenType.ASSIGN, TokenType.SEMICOLON]:
                line += tokens[i].value
                i += 1
        
        # 初始值
        if i < len(tokens) and tokens[i].type == TokenType.ASSIGN:
            if self.config.space_around_operators:
                line += " = "
            else:
                line += "="
            i += 1
            
            while i < len(tokens) and tokens[i].type != TokenType.SEMICOLON:
                line += tokens[i].value
                if tokens[i].type in [TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.SLASH]:
                    if self.config.space_around_operators:
                        line += " "
                i += 1
        
        line += ";"
        self.add_line(line)
        
        return i + 1
    
    def format_return_statement(self, tokens: List[Token], start: int) -> int:
        """格式化return语句"""
        line = "return"
        i = start + 1
        
        if i < len(tokens) and tokens[i].type != TokenType.SEMICOLON:
            line += " "
            
            while i < len(tokens) and tokens[i].type != TokenType.SEMICOLON:
                line += tokens[i].value
                if tokens[i].type in [TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.SLASH]:
                    if self.config.space_around_operators:
                        line += " "
                i += 1
        
        line += ";"
        self.add_line(line)
        
        return i + 1
    
    def format_if_statement(self, tokens: List[Token], start: int) -> int:
        """格式化if语句"""
        line = "if ("
        i = start + 1
        
        # 跳过左括号
        if i < len(tokens) and tokens[i].type == TokenType.LPAREN:
            i += 1
        
        # 条件表达式
        while i < len(tokens) and tokens[i].type != TokenType.RPAREN:
            line += tokens[i].value
            if tokens[i].type in [TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.SLASH,
                                  TokenType.EQ, TokenType.NE, TokenType.LT, TokenType.GT]:
                if self.config.space_around_operators:
                    line += " "
            i += 1
        
        line += ")"
        
        # 左大括号
        if i + 1 < len(tokens) and tokens[i + 1].type == TokenType.LBRACE:
            if self.config.space_before_brace:
                line += " {"
            else:
                line += "{"
            
            self.add_line(line)
            self.indent_level += 1
            
            i = self.format_block(tokens, i + 2)
            
            self.indent_level -= 1
            self.add_line("}")
        
        # else分支
        if i + 1 < len(tokens) and tokens[i + 1].type == TokenType.ELSE:
            i += 2
            line = "else"
            
            if i < len(tokens) and tokens[i].type == TokenType.LBRACE:
                if self.config.space_before_brace:
                    line += " {"
                else:
                    line += "{"
                
                self.add_line(line)
                self.indent_level += 1
                
                i = self.format_block(tokens, i + 1)
                
                self.indent_level -= 1
                self.add_line("}")
        
        return i + 1
    
    def format_while_statement(self, tokens: List[Token], start: int) -> int:
        """格式化while语句"""
        line = "while ("
        i = start + 1
        
        # 跳过左括号
        if i < len(tokens) and tokens[i].type == TokenType.LPAREN:
            i += 1
        
        # 条件表达式
        while i < len(tokens) and tokens[i].type != TokenType.RPAREN:
            line += tokens[i].value
            if tokens[i].type in [TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.SLASH]:
                if self.config.space_around_operators:
                    line += " "
            i += 1
        
        line += ")"
        
        # 左大括号
        if i + 1 < len(tokens) and tokens[i + 1].type == TokenType.LBRACE:
            if self.config.space_before_brace:
                line += " {"
            else:
                line += "{"
            
            self.add_line(line)
            self.indent_level += 1
            
            i = self.format_block(tokens, i + 2)
            
            self.indent_level -= 1
            self.add_line("}")
        
        return i + 1
    
    def format_for_statement(self, tokens: List[Token], start: int) -> int:
        """格式化for语句"""
        line = "for ("
        i = start + 1
        
        # 跳过左括号
        if i < len(tokens) and tokens[i].type == TokenType.LPAREN:
            i += 1
        
        # for循环头
        while i < len(tokens) and tokens[i].type != TokenType.RPAREN:
            line += tokens[i].value
            if tokens[i].type == TokenType.SEMICOLON:
                line += " "
            i += 1
        
        line += ")"
        
        # 左大括号
        if i + 1 < len(tokens) and tokens[i + 1].type == TokenType.LBRACE:
            if self.config.space_before_brace:
                line += " {"
            else:
                line += "{"
            
            self.add_line(line)
            self.indent_level += 1
            
            i = self.format_block(tokens, i + 2)
            
            self.indent_level -= 1
            self.add_line("}")
        
        return i + 1
    
    def format_match_statement(self, tokens: List[Token], start: int) -> int:
        """格式化match语句"""
        line = "match "
        i = start + 1
        
        # 匹配表达式
        while i < len(tokens) and tokens[i].type != TokenType.LBRACE:
            line += tokens[i].value
            i += 1
        
        if self.config.space_before_brace:
            line += " {"
        else:
            line += "{"
        
        self.add_line(line)
        self.indent_level += 1
        
        # 格式化match分支
        i = self.format_match_arms(tokens, i + 1)
        
        self.indent_level -= 1
        self.add_line("}")
        
        return i
    
    def format_match_arms(self, tokens: List[Token], start: int) -> int:
        """格式化match分支"""
        i = start
        
        while i < len(tokens) and tokens[i].type != TokenType.RBRACE:
            if tokens[i].type == TokenType.CASE:
                line = "case "
                i += 1
                
                # 模式
                while i < len(tokens) and tokens[i].type != TokenType.ARROW:
                    line += tokens[i].value
                    i += 1
                
                line += " =>"
                
                # 表达式
                if i + 1 < len(tokens):
                    i += 1
                    line += " "
                    
                    while i < len(tokens) and tokens[i].type not in [TokenType.COMMA, TokenType.RBRACE]:
                        line += tokens[i].value
                        i += 1
                    
                    if i < len(tokens) and tokens[i].type == TokenType.COMMA:
                        line += ","
                        i += 1
                
                self.add_line(line)
            else:
                i += 1
        
        return i
    
    def format_generics(self, tokens: List[Token], start: int) -> tuple[str, int]:
        """格式化泛型参数"""
        result = "<"
        i = start + 1
        
        while i < len(tokens) and tokens[i].type != TokenType.GT:
            if tokens[i].type == TokenType.COMMA:
                if self.config.space_after_comma:
                    result += ", "
                else:
                    result += ","
            else:
                result += tokens[i].value
            i += 1
        
        result += ">"
        return result, i + 1
    
    def add_line(self, line: str):
        """添加一行"""
        indent = self.get_indent()
        self.output.append(indent + line)
    
    def add_blank_line(self):
        """添加空行"""
        if self.output and self.output[-1] != "":
            self.output.append("")
    
    def add_comment(self, comment: str):
        """添加注释"""
        self.add_line(comment)
    
    def get_indent(self) -> str:
        """获取当前缩进"""
        if self.config.use_spaces:
            return " " * (self.indent_level * self.config.indent_size)
        else:
            return "\t" * self.indent_level


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="AZ fmt - AZ语言代码格式化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python tools/az_fmt/azfmt.py file.az                    格式化单个文件
  python tools/az_fmt/azfmt.py file1.az file2.az          格式化多个文件
  python tools/az_fmt/azfmt.py --check file.az            检查格式但不修改
  python tools/az_fmt/azfmt.py --config azfmt.toml file.az 使用配置文件
        """
    )
    
    parser.add_argument('files', nargs='+', help='要格式化的文件')
    parser.add_argument('--check', action='store_true', help='检查格式但不修改文件')
    parser.add_argument('--config', help='配置文件路径')
    parser.add_argument('--indent', type=int, default=4, help='缩进大小（默认4）')
    parser.add_argument('--max-width', type=int, default=100, help='最大行宽（默认100）')
    parser.add_argument('--version', action='version', version='AZ fmt 0.1.0')
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        config = FormatConfig.from_file(args.config)
    else:
        config = FormatConfig(
            indent_size=args.indent,
            max_line_length=args.max_width
        )
    
    # 创建格式化器
    formatter = AZFormatter(config)
    
    # 格式化文件
    exit_code = 0
    for filepath in args.files:
        if not os.path.exists(filepath):
            print(f"错误: 文件不存在: {filepath}", file=sys.stderr)
            exit_code = 1
            continue
        
        try:
            # 读取原始内容
            with open(filepath, 'r', encoding='utf-8') as f:
                original = f.read()
            
            # 格式化
            formatted = formatter.format_source(original)
            
            if args.check:
                # 检查模式
                if original != formatted:
                    print(f"需要格式化: {filepath}")
                    exit_code = 1
                else:
                    print(f"格式正确: {filepath}")
            else:
                # 写入模式
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(formatted)
                print(f"已格式化: {filepath}")
        
        except Exception as e:
            print(f"错误: 格式化 {filepath} 失败: {e}", file=sys.stderr)
            exit_code = 1
    
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
