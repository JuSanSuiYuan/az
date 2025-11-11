#!/usr/bin/env python3
"""测试C代码生成器是否正确导入"""

import sys
sys.path.insert(0, 'bootstrap')

# 尝试导入
try:
    from az_compiler import CCodeGenerator
    print("✅ CCodeGenerator导入成功")
    
    # 尝试创建实例
    codegen = CCodeGenerator()
    print("✅ CCodeGenerator实例化成功")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
