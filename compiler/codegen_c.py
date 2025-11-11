#!/usr/bin/env python3
"""
AZ编译器 - C代码生成器
将AZ代码转译为C代码
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# ============================================================================
# C代码生成器
# ============================================================================

class CCodeGenerator:
    """C代码生成器"""
    
    def __init__(self):
        self.output = []
        self.indent_level = 0
        self.temp_var_counter = 0
        self.string_literals = []
        
    def generate(self, ast: Dict[str, Any]) -> str:
        """生成C代码"""
        self.output = []
        self.indent_level = 0
        
        # 生成头文件包含
        self.emit_includes()
        
        # 生成字符串字面量
        self.emit_string_literals()
        
        # 生成前向声明
        self.emit_forward_declarations(ast)
        
        # 生成函数定义
        for stmt in ast.get('statements', []):
            if stmt['type'] == 'function':
                self.generate_function(stmt)
        
        return '\n'.join(self.output)
    
    def emit_includes(self):
        """生成头文件包含"""
        self.emit('#include <stdio.h>')
        self.emit('#include <stdlib.h>')
        self.emit('#include <string.h>')
        self.emit('#include <stdbool.h>')
        self.emit('#include <stdint.h>')
        self.emit('')
        self.emit('// AZ Runtime')
        self.emit('#include "az_runtime.h"')
        self.emit('')
    
    def emit_string_literals(self):
        """生成字符串字面量"""
        if self.string_literals:
            self.emit('// String literals')
            for i, s in enumerate(self.string_literals):
                escaped = s.replace('\\', '\\\\').replace('"', '\\"')
                self.emit(f'static const char* __str_{i} = "{escaped}";')
            self.emit('')
    
    def emit_forward_declarations(self, ast: Dict[str, Any]):
        """生成前向声明"""
        self.emit('// Forward declarations')
        for stmt in ast.get('statements', []):
            if stmt['type'] == 'function':
                sig = self.generate_function_signature(stmt)
                self.emit(f'{sig};')
        self.emit('')
    
    def generate_function(self, func: Dict[str, Any]):
        """生成函数定义"""
        sig = self.generate_function_signature(func)
        self.emit(f'{sig} {{')
        self.indent_level += 1
        
        # 生成函数体
        if 'body' in func:
            for stmt in func['body']:
                self.generate_statement(stmt)
        
        self.indent_level -= 1
        self.emit('}')
        self.emit('')
    
    def generate_function_signature(self, func: Dict[str, Any]) -> str:
        """生成函数签名"""
        name = func.get('name', 'unknown')
        return_type = self.map_type(func.get('return_type', 'void'))
        
        # 参数
        params = []
        for param in func.get('parameters', []):
            param_type = self.map_type(param.get('type', 'int'))
            param_name = param.get('name', 'arg')
            params.append(f'{param_type} {param_name}')
        
        params_str = ', '.join(params) if params else 'void'
        return f'{return_type} {name}({params_str})'
    
    def generate_statement(self, stmt: Dict[str, Any]):
        """生成语句"""
        stmt_type = stmt.get('type')
        
        if stmt_type == 'variable_declaration':
            self.generate_variable_declaration(stmt)
        elif stmt_type == 'assignment':
            self.generate_assignment(stmt)
        elif stmt_type == 'return':
            self.generate_return(stmt)
        elif stmt_type == 'if':
            self.generate_if(stmt)
        elif stmt_type == 'while':
            self.generate_while(stmt)
        elif stmt_type == 'for':
            self.generate_for(stmt)
        elif stmt_type == 'expression_statement':
            expr = self.generate_expression(stmt.get('expression', {}))
            self.emit(f'{expr};')
        else:
            self.emit(f'// Unknown statement: {stmt_type}')
    
    def generate_variable_declaration(self, stmt: Dict[str, Any]):
        """生成变量声明"""
        var_type = self.map_type(stmt.get('var_type', 'int'))
        name = stmt.get('name', 'var')
        
        if 'value' in stmt:
            value = self.generate_expression(stmt['value'])
            self.emit(f'{var_type} {name} = {value};')
        else:
            self.emit(f'{var_type} {name};')
    
    def generate_assignment(self, stmt: Dict[str, Any]):
        """生成赋值语句"""
        target = stmt.get('target', 'var')
        value = self.generate_expression(stmt.get('value', {}))
        self.emit(f'{target} = {value};')
    
    def generate_return(self, stmt: Dict[str, Any]):
        """生成return语句"""
        if 'value' in stmt:
            value = self.generate_expression(stmt['value'])
            self.emit(f'return {value};')
        else:
            self.emit('return;')
    
    def generate_if(self, stmt: Dict[str, Any]):
        """生成if语句"""
        condition = self.generate_expression(stmt.get('condition', {}))
        self.emit(f'if ({condition}) {{')
        self.indent_level += 1
        
        for s in stmt.get('then_body', []):
            self.generate_statement(s)
        
        self.indent_level -= 1
        
        if 'else_body' in stmt:
            self.emit('} else {')
            self.indent_level += 1
            
            for s in stmt['else_body']:
                self.generate_statement(s)
            
            self.indent_level -= 1
        
        self.emit('}')
    
    def generate_while(self, stmt: Dict[str, Any]):
        """生成while语句"""
        condition = self.generate_expression(stmt.get('condition', {}))
        self.emit(f'while ({condition}) {{')
        self.indent_level += 1
        
        for s in stmt.get('body', []):
            self.generate_statement(s)
        
        self.indent_level -= 1
        self.emit('}')
    
    def generate_for(self, stmt: Dict[str, Any]):
        """生成for语句"""
        init = self.generate_statement_inline(stmt.get('init', {}))
        condition = self.generate_expression(stmt.get('condition', {}))
        update = self.generate_statement_inline(stmt.get('update', {}))
        
        self.emit(f'for ({init}; {condition}; {update}) {{')
        self.indent_level += 1
        
        for s in stmt.get('body', []):
            self.generate_statement(s)
        
        self.indent_level -= 1
        self.emit('}')
    
    def generate_expression(self, expr: Dict[str, Any]) -> str:
        """生成表达式"""
        expr_type = expr.get('type')
        
        if expr_type == 'integer':
            return str(expr.get('value', 0))
        elif expr_type == 'float':
            return str(expr.get('value', 0.0))
        elif expr_type == 'string':
            s = expr.get('value', '')
            idx = len(self.string_literals)
            self.string_literals.append(s)
            return f'__str_{idx}'
        elif expr_type == 'boolean':
            return 'true' if expr.get('value', False) else 'false'
        elif expr_type == 'identifier':
            return expr.get('name', 'var')
        elif expr_type == 'binary_op':
            return self.generate_binary_op(expr)
        elif expr_type == 'unary_op':
            return self.generate_unary_op(expr)
        elif expr_type == 'call':
            return self.generate_call(expr)
        else:
            return '0'
    
    def generate_binary_op(self, expr: Dict[str, Any]) -> str:
        """生成二元运算"""
        left = self.generate_expression(expr.get('left', {}))
        right = self.generate_expression(expr.get('right', {}))
        op = expr.get('operator', '+')
        
        return f'({left} {op} {right})'
    
    def generate_unary_op(self, expr: Dict[str, Any]) -> str:
        """生成一元运算"""
        operand = self.generate_expression(expr.get('operand', {}))
        op = expr.get('operator', '-')
        
        return f'({op}{operand})'
    
    def generate_call(self, expr: Dict[str, Any]) -> str:
        """生成函数调用"""
        func_name = expr.get('function', 'func')
        
        # 特殊处理标准库函数
        if func_name == 'println':
            args = expr.get('arguments', [])
            if args:
                arg = self.generate_expression(args[0])
                return f'az_println({arg})'
            return 'az_println("")'
        elif func_name == 'print':
            args = expr.get('arguments', [])
            if args:
                arg = self.generate_expression(args[0])
                return f'az_print({arg})'
            return 'az_print("")'
        
        # 普通函数调用
        args = [self.generate_expression(arg) for arg in expr.get('arguments', [])]
        args_str = ', '.join(args)
        
        return f'{func_name}({args_str})'
    
    def generate_statement_inline(self, stmt: Dict[str, Any]) -> str:
        """生成内联语句（用于for循环）"""
        if stmt.get('type') == 'variable_declaration':
            var_type = self.map_type(stmt.get('var_type', 'int'))
            name = stmt.get('name', 'var')
            if 'value' in stmt:
                value = self.generate_expression(stmt['value'])
                return f'{var_type} {name} = {value}'
            return f'{var_type} {name}'
        elif stmt.get('type') == 'assignment':
            target = stmt.get('target', 'var')
            value = self.generate_expression(stmt.get('value', {}))
            return f'{target} = {value}'
        return ''
    
    def map_type(self, az_type: str) -> str:
        """映射AZ类型到C类型"""
        type_map = {
            'int': 'int64_t',
            'float': 'double',
            'bool': 'bool',
            'string': 'const char*',
            'void': 'void',
            'char': 'char',
            'byte': 'uint8_t'
        }
        return type_map.get(az_type, 'void')
    
    def emit(self, code: str):
        """输出一行代码"""
        if code:
            indent = '    ' * self.indent_level
            self.output.append(indent + code)
        else:
            self.output.append('')


# ============================================================================
# 主函数
# ============================================================================

def generate_c_code(ast: Dict[str, Any]) -> str:
    """生成C代码"""
    generator = CCodeGenerator()
    return generator.generate(ast)


if __name__ == '__main__':
    # 测试
    test_ast = {
        'statements': [
            {
                'type': 'function',
                'name': 'main',
                'return_type': 'int',
                'parameters': [],
                'body': [
                    {
                        'type': 'expression_statement',
                        'expression': {
                            'type': 'call',
                            'function': 'println',
                            'arguments': [
                                {
                                    'type': 'string',
                                    'value': 'Hello, World!'
                                }
                            ]
                        }
                    },
                    {
                        'type': 'return',
                        'value': {
                            'type': 'integer',
                            'value': 0
                        }
                    }
                ]
            }
        ]
    }
    
    code = generate_c_code(test_ast)
    print(code)
