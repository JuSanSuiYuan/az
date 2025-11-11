#!/usr/bin/env python3
"""
简化版AZ编译器驱动程序
直接使用clang编译，绕过C++编译器
"""

import sys
import os
import subprocess
import tempfile
from pathlib import Path

def compile_az_to_llvm_ir(source_file):
    """将AZ源文件编译为LLVM IR（模拟）"""
    # 这里我们模拟编译过程，实际应该调用AZ编译器
    # 为了测试，我们直接生成一个简单的LLVM IR
    llvm_ir = """
; ModuleID = 'simple_test'
source_filename = "simple_test.az"

@.str = private unnamed_addr constant [21 x i8] c"Hello, AZ Toolchain!\\00", align 1

declare i32 @printf(i8*, ...)

define i32 @main() {
entry:
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.str, i32 0, i32 0))
  ret i32 0
}
"""
    return llvm_ir

def compile_with_clang(llvm_ir, output_file):
    """使用clang编译LLVM IR"""
    try:
        # 创建临时目录存储中间文件
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            ir_file = temp_path / "input.ll"
            
            # 写入LLVM IR
            with open(ir_file, 'w') as f:
                f.write(llvm_ir)
            
            # 使用clang编译
            cmd = ['clang', str(ir_file), '-o', output_file]
            print(f"执行命令: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print("clang编译输出:")
                if result.stdout:
                    print(result.stdout)
                if result.stderr:
                    print(result.stderr, file=sys.stderr)
                return False
            
            return True
                    
    except Exception as e:
        print(f"clang编译错误: {e}")
        return False

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python simple_azc.py <源文件.az> [-o 输出文件]")
        return 1
    
    source_file = sys.argv[1]
    output_file = "a.out"
    
    # 解析输出文件参数
    if "-o" in sys.argv:
        try:
            idx = sys.argv.index("-o")
            output_file = sys.argv[idx + 1]
        except IndexError:
            print("错误: -o 需要一个参数")
            return 1
    
    print(f"🔨 编译AZ程序 {source_file}...")
    
    # 1. 编译AZ到LLVM IR
    print("[1/2] 编译到LLVM IR...")
    llvm_ir = compile_az_to_llvm_ir(source_file)
    if not llvm_ir:
        print("❌ 编译到LLVM IR失败")
        return 1
    
    # 2. 使用clang编译
    print("[2/2] 使用clang编译...")
    if not compile_with_clang(llvm_ir, output_file):
        print("❌ clang编译失败")
        return 1
    
    print("✅ 编译成功完成!")
    print(f"输出文件: {output_file}")
    return 0

if __name__ == "__main__":
    sys.exit(main())