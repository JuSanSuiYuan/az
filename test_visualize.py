#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试AZ AST可视化功能
"""

import os
import sys
import subprocess

def test_ast_visualization():
    """测试AST可视化功能"""
    print("🔍 测试AZ AST可视化功能")
    
    # 检查必要的文件是否存在
    files_to_check = [
        "tools/ast_visualizer.py",
        "examples/visualize_test.az",
        "docs/AST_VISUALIZATION.md"
    ]
    
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            print(f"❌ 错误: 找不到文件 {file_path}")
            return False
        print(f"✅ 找到文件: {file_path}")
    
    # 测试Python可视化工具
    print("\n🧪 测试Python可视化工具...")
    try:
        result = subprocess.run([
            sys.executable, 
            "tools/ast_visualizer.py", 
            "examples/visualize_test.az",
            "-o", "test_ast.dot"
        ], capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print("✅ Python可视化工具运行成功")
            print(result.stdout)
            
            # 检查输出文件是否存在
            if os.path.exists("test_ast.dot"):
                print("✅ 生成了DOT文件: test_ast.dot")
                
                # 显示文件大小
                size = os.path.getsize("test_ast.dot")
                print(f"📄 DOT文件大小: {size} 字节")
                
                # 显示文件前几行
                with open("test_ast.dot", "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    print("📋 DOT文件前5行:")
                    for i, line in enumerate(lines[:5]):
                        print(f"  {i+1}: {line.rstrip()}")
            else:
                print("❌ 未生成DOT文件")
                return False
        else:
            print("❌ Python可视化工具运行失败")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ 运行Python可视化工具时出错: {e}")
        return False
    
    return True

def main():
    """主函数"""
    print("🚀 AZ AST可视化功能测试")
    print("=" * 50)
    
    if test_ast_visualization():
        print("\n🎉 所有测试通过!")
        print("\n📝 下一步:")
        print("  1. 安装Graphviz (https://graphviz.org/download/)")
        print("  2. 运行: dot -Tpng test_ast.dot -o ast.png")
        print("  3. 查看生成的AST图: ast.png")
        return 0
    else:
        print("\n💥 测试失败!")
        return 1

if __name__ == "__main__":
    sys.exit(main())