#!/usr/bin/env python3
"""
AZ语言编译器命令行工具
快速编译AZ代码为可执行文件
"""

import sys
import os
import subprocess
import argparse
import time

def compile_az(source_file, output_file=None, optimize=False, keep_c=False, verbose=False):
    """编译AZ源文件为可执行文件"""
    
    start_time = time.time()
    
    # 检查源文件是否存在
    if not os.path.exists(source_file):
        print(f"❌ 错误: 找不到文件 '{source_file}'")
        return False
    
    # 确定输出文件名
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(source_file))[0]
        output_file = base_name
    
    # Windows需要.exe扩展名
    if os.name == 'nt' and not output_file.endswith('.exe'):
        output_file += '.exe'
    
    # 生成C代码文件名
    c_file = os.path.splitext(source_file)[0] + '.c'
    
    try:
        # 步骤1: 编译AZ代码到C代码
        print(f"[1/3] 编译 {source_file} -> {c_file}")
        
        python_cmd = sys.executable  # 使用当前Python解释器
        cmd = [
            python_cmd, 'bootstrap/az_compiler.py',
            source_file,
            '--emit-c', '-o', c_file
        ]
        
        if verbose:
            print(f"  命令: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=not verbose)
        if result.returncode != 0:
            print("❌ AZ编译失败")
            if not verbose and result.stderr:
                print(result.stderr.decode())
            return False
        
        # 步骤2: 使用Clang编译C代码
        print(f"[2/3] 编译 {c_file} -> {output_file}")
        
        clang_cmd = ['clang', c_file, 'runtime/azstd.c', '-o', output_file]
        
        # 添加优化选项
        if optimize:
            clang_cmd.insert(1, '-O3')
            clang_cmd.insert(1, '-flto')  # 链接时优化
        
        # 添加数学库
        if os.name != 'nt':  # Linux/macOS需要链接数学库
            clang_cmd.append('-lm')
        
        if verbose:
            print(f"  命令: {' '.join(clang_cmd)}")
        
        result = subprocess.run(clang_cmd, capture_output=not verbose)
        if result.returncode != 0:
            print("❌ C代码编译失败")
            if not verbose and result.stderr:
                print(result.stderr.decode())
            return False
        
        # 步骤3: 清理临时文件
        if not keep_c:
            print(f"[3/3] 清理临时文件")
            try:
                os.remove(c_file)
            except:
                pass
        else:
            print(f"[3/3] 保留C代码: {c_file}")
        
        # 编译成功
        elapsed = time.time() - start_time
        print(f"\n✅ 编译成功!")
        print(f"   输出: {output_file}")
        print(f"   耗时: {elapsed:.2f}秒")
        
        return True
        
    except Exception as e:
        print(f"❌ 编译过程出错: {e}")
        return False

def run_program(program_path):
    """运行编译后的程序"""
    # 确保路径正确
    if not os.path.exists(program_path):
        print(f"❌ 错误: 找不到程序 '{program_path}'")
        return 1
    
    print(f"\n{'='*50}")
    print(f"运行: {program_path}")
    print(f"{'='*50}\n")
    
    # 使用绝对路径
    abs_path = os.path.abspath(program_path)
    
    try:
        result = subprocess.run([abs_path])
        print(f"\n{'='*50}")
        print(f"程序退出码: {result.returncode}")
        print(f"{'='*50}")
        return result.returncode
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        return 1

def main():
    parser = argparse.ArgumentParser(
        description='AZ语言编译器 - 将AZ代码编译为可执行文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  %(prog)s hello.az                    # 编译hello.az
  %(prog)s hello.az -o hello           # 指定输出文件名
  %(prog)s hello.az -O                 # 优化编译
  %(prog)s hello.az --run              # 编译并运行
  %(prog)s hello.az -O --run           # 优化编译并运行
  %(prog)s hello.az --keep-c           # 保留生成的C代码
        '''
    )
    
    parser.add_argument('source', help='AZ源文件 (.az)')
    parser.add_argument('-o', '--output', help='输出文件名')
    parser.add_argument('-O', '--optimize', action='store_true', 
                       help='启用优化 (-O3 -flto)')
    parser.add_argument('--run', action='store_true', 
                       help='编译后立即运行')
    parser.add_argument('--keep-c', action='store_true', 
                       help='保留生成的C代码')
    parser.add_argument('-v', '--verbose', action='store_true', 
                       help='显示详细信息')
    
    args = parser.parse_args()
    
    # 显示版本信息
    if args.verbose:
        print("AZ编译器 v0.5.0-dev")
        print("基于LLVM/Clang技术栈")
        print()
    
    # 编译
    success = compile_az(
        args.source, 
        args.output, 
        args.optimize, 
        args.keep_c,
        args.verbose
    )
    
    if not success:
        sys.exit(1)
    
    # 运行
    if args.run:
        output = args.output or os.path.splitext(os.path.basename(args.source))[0]
        # Windows需要.exe扩展名
        if os.name == 'nt':
            if not output.endswith('.exe'):
                output += '.exe'
        exit_code = run_program(output)
        sys.exit(exit_code)

if __name__ == '__main__':
    main()
