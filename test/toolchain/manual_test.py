#!/usr/bin/env python3
"""
手动测试AZ工具链的mold链接器和AOT功能
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

def test_with_clang():
    """使用clang测试编译"""
    print("=== 使用clang测试编译 ===")
    
    # 编译简单测试程序
    ret, stdout, stderr = run_command([
        "python", "tools/azc", "--linker", "clang", "test/toolchain/simple_test.az", "-o", "test/toolchain/simple_test_clang"
    ])
    
    if ret != 0:
        print(f"❌ 使用clang链接器编译失败: {stderr}")
        return False
    else:
        print("✅ 使用clang链接器编译成功")
    
    return True

def test_aot_compilation():
    """测试AOT编译"""
    print("=== 测试AOT编译 ===")
    
    # AOT编译测试程序
    ret, stdout, stderr = run_command([
        "python", "tools/azc", "--aot", "test/toolchain/aot_test.az", "-o", "test/toolchain/aot_test_compiled"
    ])
    
    if ret != 0:
        print(f"❌ AOT编译失败: {stderr}")
        return False
    else:
        print("✅ AOT编译成功")
    
    return True

def main():
    """主函数"""
    print("开始手动测试AZ工具链...")
    
    # 运行测试
    tests = [
        test_with_clang,
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
