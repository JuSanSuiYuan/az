#!/usr/bin/env python3
"""
测试所有新实现的功能：模式匹配、for循环和数组切片
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

def test_all_features():
    """测试所有新功能"""
    print("=== 测试所有新功能 ===")
    
    # 使用azc编译测试文件
    ret, stdout, stderr = run_command([
        "python", "tools/azc", "test/frontend/complete_test.az", "--emit-llvm"
    ])
    
    if ret != 0:
        print(f"❌ 编译失败: {stderr}")
        return False
    else:
        print("✅ 所有功能解析成功")
        return True

def main():
    """主函数"""
    print("开始测试所有新实现的功能...")
    
    # 运行测试
    if test_all_features():
        print("\n🎉 所有功能测试通过!")
        return 0
    else:
        print("\n💥 功能测试失败!")
        return 1

if __name__ == "__main__":
    sys.exit(main())