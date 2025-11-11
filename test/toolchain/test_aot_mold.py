#!/usr/bin/env python3
"""
AOT和mold链接器测试脚本
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

def test_mold_linker():
    """测试mold链接器"""
    print("=== 测试mold链接器 ===")
    
    # 检查mold是否可用
    ret, stdout, stderr = run_command(["mold", "--version"])
    if ret != 0:
        print("⚠️  mold链接器不可用，跳过测试")
        return True
    
    print(f"✅ mold版本: {stdout.splitlines()[0] if stdout else 'Unknown'}")
    
    # 使用mold链接器编译测试程序
    ret, stdout, stderr = run_command([
        "python", "tools/azc", "--linker", "mold", "test/toolchain/hello.az", "-o", "test/toolchain/hello_mold"
    ], cwd=Path(__file__).parent.parent.parent)
    
    if ret != 0:
        print(f"❌ 使用mold链接器编译失败: {stderr}")
        return False
    else:
        print("✅ 使用mold链接器编译成功")
    
    return True

def test_aot_compilation():
    """测试AOT编译"""
    print("=== 测试AOT编译 ===")
    
    # AOT编译测试程序
    ret, stdout, stderr = run_command([
        "python", "tools/azc", "--aot", "test/toolchain/aot_test.az", "-o", "test/toolchain/aot_test"
    ], cwd=Path(__file__).parent.parent.parent)
    
    if ret != 0:
        print(f"❌ AOT编译失败: {stderr}")
        return False
    else:
        print("✅ AOT编译成功")
    
    # 运行AOT编译的程序
    ret, stdout, stderr = run_command([
        "./test/toolchain/aot_test"
    ], cwd=Path(__file__).parent.parent.parent)
    
    if ret != 0:
        print(f"❌ AOT程序运行失败: {stderr}")
        return False
    else:
        print("✅ AOT程序运行成功")
        print(f"程序输出:\n{stdout}")
    
    return True

def test_linker_detection():
    """测试链接器检测功能"""
    print("=== 测试链接器检测 ===")
    
    # 测试自动检测链接器
    ret, stdout, stderr = run_command([
        "python", "tools/azc", "test/toolchain/hello.az", "-o", "test/toolchain/hello_auto"
    ], cwd=Path(__file__).parent.parent.parent)
    
    if ret != 0:
        print(f"❌ 自动链接器检测编译失败: {stderr}")
        return False
    else:
        print("✅ 自动链接器检测编译成功")
    
    return True

def main():
    """主函数"""
    print("开始AOT和mold链接器测试...")
    
    # 确保测试目录存在
    test_dir = Path(__file__).parent
    os.makedirs(test_dir, exist_ok=True)
    
    # 运行所有测试
    tests = [
        test_linker_detection,
        test_mold_linker,
        test_aot_compilation
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