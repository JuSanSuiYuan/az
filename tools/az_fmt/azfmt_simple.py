#!/usr/bin/env python3
"""
AZ fmt Simple - AZ语言代码格式化工具（简化版）
不依赖Lexer，直接基于正则表达式和字符串处理

用法:
    python tools/az_fmt/azfmt_simple.py file.az
"""

import sys
import os
import re
import argparse
from pathlib import Path
from dataclasses import dataclass


@dataclass
class FormatConfig:
    """格式化配置"""
    indent_size: int = 4
    max_line_length: int = 100
    use_spaces: bool = True
    space_before_brace: bool = True
    space_after_comma: bool = True
    space_around_operators: bool = True


class SimpleAZFormatter:
    """简单的AZ代码格式化器"""
    
    def __init__(self, config: FormatConfig = None):
        self.config = config or FormatConfig()
    
    def format_source(self, source: str) -> str:
        """格式化源代码"""
        # 首先分割成语句
        statements = self.split_statements(source)
        
        formatted_lines = []
        indent_level = 0
        
        for stmt in statements:
            stmt = stmt.strip()
            
            # 跳过空语句
            if not stmt:
                continue
            
            # 处理注释
            if stmt.startswith('//'):
                formatted_lines.append(self.indent(stmt, indent_level))
                continue
            
            # 减少缩进（右大括号）
            if stmt == '}':
                indent_level = max(0, indent_level - 1)
                formatted_lines.append(self.indent('}', indent_level))
                # 在}后添加空行（除非是最后一个）
                formatted_lines.append('')
                continue
            
            # 格式化语句
            formatted_stmt = self.format_line(stmt)
            
            # 处理包含大括号的语句
            if '{' in formatted_stmt:
                # 分割成多行
                parts = formatted_stmt.split('{')
                formatted_lines.append(self.indent(parts[0].strip() + ' {', indent_level))
                indent_level += 1
                
                # 处理大括号后的内容
                if len(parts) > 1 and parts[1].strip():
                    rest = parts[1].strip()
                    if rest == '}':
                        indent_level = max(0, indent_level - 1)
                        formatted_lines.append(self.indent('}', indent_level))
                        formatted_lines.append('')
                    else:
                        # 递归处理
                        sub_stmts = self.split_statements(rest)
                        for sub_stmt in sub_stmts:
                            if sub_stmt.strip():
                                formatted_lines.append(self.indent(self.format_line(sub_stmt.strip()), indent_level))
            else:
                formatted_lines.append(self.indent(formatted_stmt, indent_level))
                
                # 在import和module后添加空行
                if formatted_stmt.startswith('import ') or formatted_stmt.startswith('module '):
                    formatted_lines.append('')
        
        # 移除多余的空行
        result = []
        prev_empty = False
        for line in formatted_lines:
            is_empty = not line.strip()
            if is_empty and prev_empty:
                continue
            result.append(line)
            prev_empty = is_empty
        
        # 确保文件以换行结束
        if result and result[-1]:
            result.append('')
        
        return '\n'.join(result)
    
    def split_statements(self, source: str) -> list:
        """分割语句"""
        statements = []
        current = ''
        in_string = False
        brace_level = 0
        
        for char in source:
            if char == '"' and (not current or current[-1] != '\\'):
                in_string = not in_string
            
            if not in_string:
                if char == '{':
                    brace_level += 1
                elif char == '}':
                    brace_level -= 1
                    if brace_level == 0 and current.strip():
                        statements.append(current.strip())
                        statements.append('}')
                        current = ''
                        continue
                elif char == ';' and brace_level == 0:
                    current += char
                    if current.strip():
                        statements.append(current.strip())
                    current = ''
                    continue
            
            current += char
        
        if current.strip():
            statements.append(current.strip())
        
        return statements
    
    def format_line(self, line: str) -> str:
        """格式化单行"""
        # 移除多余空格
        line = ' '.join(line.split())
        
        # 格式化import语句
        if line.startswith('import '):
            return self.format_import(line)
        
        # 格式化module语句
        if line.startswith('module '):
            return self.format_module(line)
        
        # 格式化函数定义
        if 'fn ' in line and '(' in line:
            return self.format_function(line)
        
        # 格式化结构体定义
        if line.startswith('struct ') or line.startswith('pub struct '):
            return self.format_struct(line)
        
        # 格式化枚举定义
        if line.startswith('enum ') or line.startswith('pub enum '):
            return self.format_enum(line)
        
        # 格式化变量声明
        if line.startswith('let ') or line.startswith('var '):
            return self.format_variable(line)
        
        # 格式化return语句
        if line.startswith('return '):
            return self.format_return(line)
        
        # 格式化if语句
        if line.startswith('if ') or line.startswith('if('):
            return self.format_if(line)
        
        # 格式化else语句
        if line.startswith('else ') or line == 'else' or line.startswith('else{'):
            return self.format_else(line)
        
        # 格式化while语句
        if line.startswith('while ') or line.startswith('while('):
            return self.format_while(line)
        
        # 格式化for语句
        if line.startswith('for ') or line.startswith('for('):
            return self.format_for(line)
        
        # 格式化大括号
        line = self.format_braces(line)
        
        # 格式化运算符
        line = self.format_operators(line)
        
        # 格式化逗号
        line = self.format_commas(line)
        
        # 格式化冒号
        line = self.format_colons(line)
        
        return line
    
    def format_import(self, line: str) -> str:
        """格式化import语句"""
        # import std.io;
        line = re.sub(r'import\s+', 'import ', line)
        if not line.endswith(';'):
            line += ';'
        return line
    
    def format_module(self, line: str) -> str:
        """格式化module语句"""
        # module test.example;
        line = re.sub(r'module\s+', 'module ', line)
        if not line.endswith(';'):
            line += ';'
        return line
    
    def format_function(self, line: str) -> str:
        """格式化函数定义"""
        # pub fn add(a: int, b: int) int {
        
        # 处理pub关键字
        line = re.sub(r'pub\s+fn\s+', 'pub fn ', line)
        line = re.sub(r'fn\s+', 'fn ', line)
        
        # 处理参数列表
        line = re.sub(r'\(\s*', '(', line)
        line = re.sub(r'\s*\)', ')', line)
        line = re.sub(r',\s*', ', ', line)
        line = re.sub(r':\s*', ': ', line)
        
        # 处理大括号
        if self.config.space_before_brace:
            line = re.sub(r'\)\s*(\w+)\s*\{', r') \1 {', line)
            line = re.sub(r'\)\s*\{', r') {', line)
        else:
            line = re.sub(r'\)\s*(\w+)\s*\{', r') \1{', line)
        
        return line
    
    def format_struct(self, line: str) -> str:
        """格式化结构体定义"""
        # pub struct Point {
        line = re.sub(r'pub\s+struct\s+', 'pub struct ', line)
        line = re.sub(r'struct\s+', 'struct ', line)
        
        if self.config.space_before_brace:
            line = re.sub(r'\s*\{', ' {', line)
        else:
            line = re.sub(r'\s*\{', '{', line)
        
        # 处理字段
        if ':' in line:
            line = re.sub(r':\s*', ': ', line)
        if ',' in line:
            line = re.sub(r',\s*', ', ', line)
        
        return line
    
    def format_enum(self, line: str) -> str:
        """格式化枚举定义"""
        # pub enum Result<T, E> {
        line = re.sub(r'pub\s+enum\s+', 'pub enum ', line)
        line = re.sub(r'enum\s+', 'enum ', line)
        
        # 处理泛型
        line = re.sub(r'<\s*', '<', line)
        line = re.sub(r'\s*>', '>', line)
        line = re.sub(r',\s*', ', ', line)
        
        if self.config.space_before_brace:
            line = re.sub(r'\s*\{', ' {', line)
        else:
            line = re.sub(r'\s*\{', '{', line)
        
        return line
    
    def format_variable(self, line: str) -> str:
        """格式化变量声明"""
        # let x: int = 10;
        line = re.sub(r'let\s+', 'let ', line)
        line = re.sub(r'var\s+', 'var ', line)
        line = re.sub(r':\s*', ': ', line)
        
        if self.config.space_around_operators:
            line = re.sub(r'\s*=\s*', ' = ', line)
        else:
            line = re.sub(r'\s*=\s*', '=', line)
        
        if not line.endswith(';'):
            line += ';'
        
        return line
    
    def format_return(self, line: str) -> str:
        """格式化return语句"""
        # return a + b;
        line = re.sub(r'return\s+', 'return ', line)
        
        if not line.endswith(';'):
            line += ';'
        
        return line
    
    def format_if(self, line: str) -> str:
        """格式化if语句"""
        # if (x > 10) {
        line = re.sub(r'if\s*\(', 'if (', line)
        line = re.sub(r'\)\s*\{', ') {' if self.config.space_before_brace else '){', line)
        
        return line
    
    def format_else(self, line: str) -> str:
        """格式化else语句"""
        # else {
        if line == 'else':
            return 'else'
        
        line = re.sub(r'else\s*\{', 'else {' if self.config.space_before_brace else 'else{', line)
        
        return line
    
    def format_while(self, line: str) -> str:
        """格式化while语句"""
        # while (condition) {
        line = re.sub(r'while\s*\(', 'while (', line)
        line = re.sub(r'\)\s*\{', ') {' if self.config.space_before_brace else '){', line)
        
        return line
    
    def format_for(self, line: str) -> str:
        """格式化for语句"""
        # for (i = 0; i < 10; i = i + 1) {
        line = re.sub(r'for\s*\(', 'for (', line)
        line = re.sub(r'\)\s*\{', ') {' if self.config.space_before_brace else '){', line)
        
        return line
    
    def format_braces(self, line: str) -> str:
        """格式化大括号"""
        if self.config.space_before_brace:
            line = re.sub(r'(\w)\s*\{', r'\1 {', line)
        
        return line
    
    def format_operators(self, line: str) -> str:
        """格式化运算符"""
        if not self.config.space_around_operators:
            return line
        
        # 保护字符串中的内容
        # 简化版本：只处理基本运算符
        operators = ['+', '-', '*', '/', '%', '==', '!=', '<', '>', '<=', '>=', '&&', '||']
        
        for op in operators:
            # 避免影响已经正确格式化的
            pattern = r'(\w)\s*' + re.escape(op) + r'\s*(\w)'
            replacement = r'\1 ' + op + r' \2'
            line = re.sub(pattern, replacement, line)
        
        return line
    
    def format_commas(self, line: str) -> str:
        """格式化逗号"""
        if self.config.space_after_comma:
            line = re.sub(r',\s*', ', ', line)
        else:
            line = re.sub(r',\s*', ',', line)
        
        return line
    
    def format_colons(self, line: str) -> str:
        """格式化冒号"""
        # 在类型注解中
        line = re.sub(r'(\w)\s*:\s*(\w)', r'\1: \2', line)
        
        return line
    
    def indent(self, line: str, level: int) -> str:
        """添加缩进"""
        if not line.strip():
            return ''
        
        if self.config.use_spaces:
            indent = ' ' * (level * self.config.indent_size)
        else:
            indent = '\t' * level
        
        return indent + line


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="AZ fmt Simple - AZ语言代码格式化工具（简化版）"
    )
    
    parser.add_argument('files', nargs='+', help='要格式化的文件')
    parser.add_argument('--check', action='store_true', help='检查格式但不修改文件')
    parser.add_argument('--indent', type=int, default=4, help='缩进大小（默认4）')
    parser.add_argument('--version', action='version', version='AZ fmt-simple 0.1.0')
    
    args = parser.parse_args()
    
    # 创建配置
    config = FormatConfig(indent_size=args.indent)
    
    # 创建格式化器
    formatter = SimpleAZFormatter(config)
    
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
