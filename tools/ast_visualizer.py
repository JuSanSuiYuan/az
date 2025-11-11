#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AZ语言AST可视化工具
使用Graphviz生成AST图形表示
"""

import sys
import os
import subprocess
import json
import argparse
from typing import Dict, Any, List

class ASTVisualizer:
    def __init__(self):
        self.node_id = 0
        
    def generate_dot(self, ast: Dict[str, Any], output_file: str):
        """生成Graphviz DOT文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("digraph AST {\n")
            f.write("  node [shape=box, style=rounded];\n")
            f.write("  rankdir=TB;\n")
            
            self.node_id = 0
            if 'statements' in ast:
                for stmt in ast['statements']:
                    self._visualize_stmt(stmt, f)
            
            f.write("}\n")
        
        print(f"✅ AST已可视化到文件: {output_file}")
        
    def _visualize_stmt(self, stmt: Dict[str, Any], f, parent_id: str = None):
        """可视化语句节点"""
        current_id = f"node{self.node_id}"
        self.node_id += 1
        
        # 创建节点
        label = self._get_stmt_label(stmt)
        f.write(f'  {current_id} [label="{label}"];\n')
        
        # 连接父节点
        if parent_id:
            f.write(f'  {parent_id} -> {current_id};\n')
        
        # 递归处理子节点
        if stmt.get('type') == 'block' and 'statements' in stmt:
            for child_stmt in stmt['statements']:
                self._visualize_stmt(child_stmt, f, current_id)
        elif stmt.get('type') == 'if':
            if 'condition' in stmt:
                self._visualize_expr(stmt['condition'], f, current_id)
            if 'then_branch' in stmt:
                self._visualize_stmt(stmt['then_branch'], f, current_id)
            if 'else_branch' in stmt:
                self._visualize_stmt(stmt['else_branch'], f, current_id)
        elif stmt.get('type') == 'while':
            if 'condition' in stmt:
                self._visualize_expr(stmt['condition'], f, current_id)
            if 'body' in stmt:
                self._visualize_stmt(stmt['body'], f, current_id)
        elif stmt.get('type') == 'function':
            if 'body' in stmt:
                self._visualize_stmt(stmt['body'], f, current_id)
        elif 'expression' in stmt:
            self._visualize_expr(stmt['expression'], f, current_id)
            
    def _visualize_expr(self, expr: Dict[str, Any], f, parent_id: str = None):
        """可视化表达式节点"""
        current_id = f"node{self.node_id}"
        self.node_id += 1
        
        # 创建节点
        label = self._get_expr_label(expr)
        f.write(f'  {current_id} [label="{label}"];\n')
        
        # 连接父节点
        if parent_id:
            f.write(f'  {parent_id} -> {current_id};\n')
        
        # 递归处理子节点
        if expr.get('type') == 'binary':
            if 'left' in expr:
                self._visualize_expr(expr['left'], f, current_id)
            if 'right' in expr:
                self._visualize_expr(expr['right'], f, current_id)
        elif expr.get('type') == 'unary':
            if 'operand' in expr:
                self._visualize_expr(expr['operand'], f, current_id)
        elif expr.get('type') == 'call':
            if 'function' in expr:
                # 函数名作为文本节点
                func_id = f"node{self.node_id}"
                self.node_id += 1
                f.write(f'  {func_id} [label="Function: {expr["function"]}"];\n')
                f.write(f'  {current_id} -> {func_id};\n')
            if 'arguments' in expr:
                for arg in expr['arguments']:
                    self._visualize_expr(arg, f, current_id)
                    
    def _get_stmt_label(self, stmt: Dict[str, Any]) -> str:
        """获取语句节点标签"""
        stmt_type = stmt.get('type', 'unknown')
        if stmt_type == 'variable':
            return f"VarDecl: {stmt.get('name', 'unknown')}"
        elif stmt_type == 'function':
            return f"FuncDecl: {stmt.get('name', 'unknown')}"
        elif stmt_type == 'return':
            return "Return"
        elif stmt_type == 'if':
            return "If"
        elif stmt_type == 'while':
            return "While"
        elif stmt_type == 'block':
            return "Block"
        elif stmt_type == 'expression_statement':
            return "ExprStmt"
        else:
            return stmt_type.capitalize()
            
    def _get_expr_label(self, expr: Dict[str, Any]) -> str:
        """获取表达式节点标签"""
        expr_type = expr.get('type', 'unknown')
        if expr_type == 'integer':
            return f"Int: {expr.get('value', 0)}"
        elif expr_type == 'float':
            return f"Float: {expr.get('value', 0.0)}"
        elif expr_type == 'string':
            return f"String: {expr.get('value', '')}"
        elif expr_type == 'boolean':
            return f"Bool: {expr.get('value', False)}"
        elif expr_type == 'identifier':
            return f"Id: {expr.get('name', 'unknown')}"
        elif expr_type == 'binary':
            return f"Binary: {expr.get('operator', 'unknown')}"
        elif expr_type == 'unary':
            return f"Unary: {expr.get('operator', 'unknown')}"
        elif expr_type == 'call':
            return "Call"
        else:
            return expr_type.capitalize()

def main():
    parser = argparse.ArgumentParser(description='AZ语言AST可视化工具')
    parser.add_argument('source_file', help='AZ源文件')
    parser.add_argument('-o', '--output', default='ast.dot', help='输出文件名 (默认: ast.dot)')
    parser.add_argument('--png', action='store_true', help='同时生成PNG图像')
    
    args = parser.parse_args()
    
    # 检查源文件是否存在
    if not os.path.exists(args.source_file):
        print(f"❌ 错误: 源文件不存在: {args.source_file}")
        return 1
        
    # 检查Graphviz是否已安装
    try:
        subprocess.run(['dot', '-V'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  警告: 未找到Graphviz工具 (dot命令)")
        print("   请从 https://graphviz.org/download/ 下载并安装Graphviz")
        print("   安装后请将Graphviz的bin目录添加到系统PATH环境变量中")
    
    # 读取源文件
    try:
        with open(args.source_file, 'r', encoding='utf-8') as f:
            source_code = f.read()
    except Exception as e:
        print(f"❌ 错误: 无法读取源文件: {e}")
        return 1
    
    # 这里应该调用AZ编译器生成AST
    # 为演示目的，我们创建一个简单的AST示例
    sample_ast = {
        "statements": [
            {
                "type": "function",
                "name": "main",
                "return_type": "int",
                "parameters": [],
                "body": {
                    "type": "block",
                    "statements": [
                        {
                            "type": "variable",
                            "name": "x",
                            "var_type": "int",
                            "value": {
                                "type": "integer",
                                "value": 5
                            }
                        },
                        {
                            "type": "variable",
                            "name": "result",
                            "var_type": "int",
                            "value": {
                                "type": "call",
                                "function": "factorial",
                                "arguments": [
                                    {
                                        "type": "identifier",
                                        "name": "x"
                                    }
                                ]
                            }
                        },
                        {
                            "type": "expression_statement",
                            "expression": {
                                "type": "call",
                                "function": "println",
                                "arguments": [
                                    {
                                        "type": "binary",
                                        "operator": "+",
                                        "left": {
                                            "type": "string",
                                            "value": "Factorial of "
                                        },
                                        "right": {
                                            "type": "binary",
                                            "operator": "+",
                                            "left": {
                                                "type": "identifier",
                                                "name": "x"
                                            },
                                            "right": {
                                                "type": "binary",
                                                "operator": "+",
                                                "left": {
                                                    "type": "string",
                                                    "value": " is "
                                                },
                                                "right": {
                                                    "type": "identifier",
                                                    "name": "result"
                                                }
                                            }
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            }
        ]
    }
    
    # 生成可视化
    visualizer = ASTVisualizer()
    visualizer.generate_dot(sample_ast, args.output)
    
    # 如果需要生成PNG
    if args.png:
        png_file = args.output.replace('.dot', '.png')
        try:
            subprocess.run(['dot', '-Tpng', args.output, '-o', png_file], check=True)
            print(f"✅ PNG图像已生成: {png_file}")
        except Exception as e:
            print(f"❌ 生成PNG图像失败: {e}")
            print("   请确保已安装Graphviz并将其添加到PATH环境变量中")
    
    print(f"\n要查看生成的图形，请使用Graphviz工具:")
    print(f"  dot -Tpng {args.output} -o ast.png")
    print(f"  start ast.png")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())