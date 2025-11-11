#!/usr/bin/env python3
"""
工具链测试脚本
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, cwd=None):
    """运行命令并返回结果"""
    print(f"执行: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)

def test_compilation():
    """测试编译功能"""
    print("=== 测试编译功能 ===")
    
    # 测试基本编译
    ret, stdout, stderr = run_command([
        "python", "tools/azc", "test/toolchain/hello.az", "-o", "test/toolchain/hello"
    ], cwd=Path(__file__).parent.parent.parent)
    
    if ret != 0:
        print(f"❌ 编译失败: {stderr}")
        return False
    else:
        print("✅ 基本编译成功")
    
    # 测试生成LLVM IR
    ret, stdout, stderr = run_command([
        "python", "tools/azc", "test/toolchain/hello.az", "--emit-llvm", "-o", "test/toolchain/hello.ll"
    ], cwd=Path(__file__).parent.parent.parent)
    
    if ret != 0:
        print(f"❌ 生成LLVM IR失败: {stderr}")
        return False
    else:
        print("✅ 生成LLVM IR成功")
    
    # 测试生成汇编代码
    ret, stdout, stderr = run_command([
        "python", "tools/azc", "test/toolchain/hello.az", "--emit-asm", "-o", "test/toolchain/hello.s"
    ], cwd=Path(__file__).parent.parent.parent)
    
    if ret != 0:
        print(f"❌ 生成汇编代码失败: {stderr}")
        return False
    else:
        print("✅ 生成汇编代码成功")
    
    return True

def test_jit():
    """测试JIT功能"""
    print("=== 测试JIT功能 ===")
    
    ret, stdout, stderr = run_command([
        "python", "tools/azc", "--jit", "test/toolchain/hello.az"
    ], cwd=Path(__file__).parent.parent.parent)
    
    if ret != 0:
        print(f"❌ JIT运行失败: {stderr}")
        return False
    else:
        print("✅ JIT运行成功")
    
    return True

def test_args():
    """测试参数传递"""
    print("=== 测试参数传递 ===")
    
    ret, stdout, stderr = run_command([
        "python", "tools/azc", "--jit", "test/toolchain/args.az", "--", "arg1", "arg2", "arg3"
    ], cwd=Path(__file__).parent.parent.parent)
    
    if ret != 0:
        print(f"❌ 参数传递测试失败: {stderr}")
        return False
    else:
        print("✅ 参数传递测试成功")
    
    return True

def test_stdlib():
    """测试标准库"""
    print("=== 测试标准库 ===")
    
    ret, stdout, stderr = run_command([
        "python", "tools/azc", "test/toolchain/stdlib_test.az", "-o", "test/toolchain/stdlib_test"
    ], cwd=Path(__file__).parent.parent.parent)
    
    if ret != 0:
        print(f"❌ 标准库测试编译失败: {stderr}")
        return False
    else:
        print("✅ 标准库测试编译成功")
    
    return True

def test_optimization():
    """测试优化选项"""
    print("=== 测试优化选项 ===")
    
    # 测试不同优化级别
    opt_levels = ["-O0", "-O1", "-O2", "-O3", "-Os", "-Oz"]
    
    for opt in opt_levels:
        ret, stdout, stderr = run_command([
            "python", "tools/azc", opt, "test/toolchain/hello.az", "-o", f"test/toolchain/hello_{opt}"
        ], cwd=Path(__file__).parent.parent.parent)
        
        if ret != 0:
            print(f"❌ {opt} 优化测试失败: {stderr}")
            return False
        else:
            print(f"✅ {opt} 优化测试成功")
    
    return True

def main():
    """主函数"""
    print("开始工具链测试...")
    
    # 确保测试目录存在
    test_dir = Path(__file__).parent
    os.makedirs(test_dir, exist_ok=True)
    
    # 运行所有测试
    tests = [
        test_compilation,
        test_jit,
        test_args,
        test_stdlib,
        test_optimization
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ 测试 {test.__name__} 发生异常: {e}")
            failed += 1
    
    print(f"\n=== 测试结果 ===")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    print(f"总计: {passed + failed}")
    
    if failed == 0:
        print("🎉 所有测试通过!")
        return 0
    else:
        print("💥 部分测试失败!")
        return 1

if __name__ == "__main__":
    sys.exit(main())