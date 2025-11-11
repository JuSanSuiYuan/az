#!/usr/bin/env python3
"""
AZ Build Tool
编译AZ代码为可执行文件
"""

import sys
import os
import subprocess
import tempfile
from pathlib import Path

# 添加编译器路径
sys.path.insert(0, str(Path(__file__).parent.parent / "bootstrap"))
sys.path.insert(0, str(Path(__file__).parent.parent / "compiler"))

from az_compiler import Lexer, Parser
from codegen_c import generate_c_code

class AZBuilder:
    """AZ构建器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.runtime_dir = self.project_root / "runtime"
        self.temp_dir = None
        
    def build(self, source_file: str, output_file: str = None, verbose: bool = False):
        """构建AZ程序"""
        print(f"🔨 Building {source_file}...")
        
        # 1. 读取源文件
        print("[1/5] Reading source file...")
        with open(source_file, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # 2. 编译到AST
        print("[2/5] Compiling to AST...")
        ast = self.compile_to_ast(source_file, source_code)
        if ast is None:
            print("❌ Compilation failed")
            return False
        
        # 3. 生成C代码
        print("[3/5] Generating C code...")
        c_code = generate_c_code(ast)
        
        if verbose:
            print("\n=== Generated C Code ===")
            print(c_code)
            print("========================\n")
        
        # 4. 编译C代码
        print("[4/5] Compiling C code...")
        if not self.compile_c_code(c_code, output_file or "a.out"):
            print("❌ C compilation failed")
            return False
        
        # 5. 完成
        print("[5/5] Done!")
        print(f"✅ Successfully built: {output_file or 'a.out'}")
        return True
    
    def compile_to_ast(self, filename: str, source_code: str):
        """编译到AST"""
        try:
            # 词法分析
            lexer = Lexer(source_code, filename)
            tokens_result = lexer.tokenize()
            
            if not tokens_result.is_ok:
                print(f"Lexer error: {tokens_result.error.message}")
                return None
            
            # 语法分析
            parser = Parser(tokens_result.value)
            ast_result = parser.parse()
            
            if not ast_result.is_ok:
                print(f"Parser error: {ast_result.error.message}")
                return None
            
            # 转换AST为字典格式
            return self.ast_to_dict(ast_result.value)
            
        except Exception as e:
            print(f"Compilation error: {e}")
            return None
    
    def ast_to_dict(self, ast):
        """将AST对象转换为字典"""
        # 简化版本：假设AST已经是字典格式
        if hasattr(ast, '__dict__'):
            return ast.__dict__
        return ast
    
    def compile_c_code(self, c_code: str, output_file: str) -> bool:
        """编译C代码"""
        try:
            # 创建临时目录
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # 写入C代码
                c_file = temp_path / "main.c"
                with open(c_file, 'w', encoding='utf-8') as f:
                    f.write(c_code)
                
                # 复制运行时文件
                runtime_h = self.runtime_dir / "az_runtime.h"
                runtime_c = self.runtime_dir / "az_runtime.c"
                
                if not runtime_h.exists() or not runtime_c.exists():
                    print(f"❌ Runtime files not found in {self.runtime_dir}")
                    return False
                
                # 检测C编译器
                compiler = self.detect_c_compiler()
                if not compiler:
                    print("❌ No C compiler found (gcc, clang, or cl)")
                    return False
                
                # 编译命令
                if compiler == "cl":  # MSVC
                    cmd = [
                        compiler,
                        "/nologo",
                        f"/I{self.runtime_dir}",
                        str(c_file),
                        str(runtime_c),
                        f"/Fe{output_file}",
                        "/link", "/SUBSYSTEM:CONSOLE"
                    ]
                else:  # GCC or Clang
                    cmd = [
                        compiler,
                        f"-I{self.runtime_dir}",
                        str(c_file),
                        str(runtime_c),
                        "-o", output_file,
                        "-lm"  # 链接数学库
                    ]
                
                # 执行编译
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    print("C compiler output:")
                    print(result.stdout)
                    print(result.stderr)
                    return False
                
                return True
                
        except Exception as e:
            print(f"C compilation error: {e}")
            return False
    
    def detect_c_compiler(self) -> str:
        """检测可用的C编译器"""
        compilers = ["gcc", "clang", "cl"]
        
        for compiler in compilers:
            try:
                result = subprocess.run(
                    [compiler, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return compiler
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        return None


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AZ Build Tool - Compile AZ code to executable"
    )
    parser.add_argument("source", help="Source file (.az)")
    parser.add_argument("-o", "--output", help="Output file name")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--version", action="version", version="az-build 0.1.0")
    
    args = parser.parse_args()
    
    # 检查源文件
    if not os.path.exists(args.source):
        print(f"❌ Source file not found: {args.source}")
        return 1
    
    # 构建
    builder = AZBuilder()
    success = builder.build(args.source, args.output, args.verbose)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
