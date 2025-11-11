# AZ语言模块系统实现指南

## 概述

AZ语言采用**无头文件的模块化设计**，类似C3，但更加现代化。

## 核心特性

### 1. 无头文件
- ✅ 不需要.h文件
- ✅ 不需要接口文件
- ✅ 模块即文件
- ✅ 自动导出公开符号

### 2. 模块声明

```az
// 文件: src/math/vector.az
module math.vector;  // 声明模块名

import std.io;       // 导入标准库
import std.math;     // 导入数学库

// 公开结构体
pub struct Vec3 {
    pub x: float,
    pub y: float,
    pub z: float
}

// 公开函数
pub fn dot(a: Vec3, b: Vec3) float {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// 私有函数（默认）
fn internal_helper() void {
    // 只能在本模块内使用
}
```

### 3. 导入模块

```az
// 文件: src/main.az
module main;

// 导入整个模块
import math.vector;

fn main() int {
    // 使用模块中的公开符号
    let v1 = math.vector.Vec3 { x: 1.0, y: 2.0, z: 3.0 };
    let v2 = math.vector.Vec3 { x: 4.0, y: 5.0, z: 6.0 };
    let result = math.vector.dot(v1, v2);
    
    println("点积: " + result);
    return 0;
}
```

### 4. 选择性导入

```az
// 只导入需要的符号
import math.vector.Vec3;
import math.vector.dot;

fn main() int {
    let v1 = Vec3 { x: 1.0, y: 2.0, z: 3.0 };
    let v2 = Vec3 { x: 4.0, y: 5.0, z: 6.0 };
    let result = dot(v1, v2);
    return 0;
}
```

### 5. 别名导入

```az
import math.vector as vec;

fn main() int {
    let v = vec.Vec3 { x: 1.0, y: 2.0, z: 3.0 };
    return 0;
}
```

## 编译流程

### 单模块编译

```bash
# 编译单个文件
python az.py src/main.az -o main

# 生成过程:
# 1. 词法分析 main.az
# 2. 语法分析 main.az
# 3. 生成 C 代码
# 4. Clang 编译
# 5. 生成可执行文件
```

### 多模块编译

```bash
# 编译整个项目
python az.py src/main.az --with-modules -o myapp

# 生成过程:
# 1. 扫描所有 import 语句
# 2. 解析依赖关系
# 3. 按依赖顺序编译模块
# 4. 链接所有目标文件
# 5. 生成可执行文件
```

## 目录结构

```
myproject/
├── src/
│   ├── main.az              # 主程序
│   ├── math/
│   │   ├── vector.az        # math.vector模块
│   │   ├── matrix.az        # math.matrix模块
│   │   └── quaternion.az    # math.quaternion模块
│   ├── graphics/
│   │   ├── renderer.az      # graphics.renderer模块
│   │   └── shader.az        # graphics.shader模块
│   └── utils/
│       ├── string.az        # utils.string模块
│       └── file.az          # utils.file模块
├── runtime/
│   └── azstd.c              # 运行时标准库
├── package.az               # 包配置
└── README.md
```

## 实现状态

### 已实现 ✅

1. **基础语法**
   - ✅ 变量声明（let/var）
   - ✅ 函数定义
   - ✅ 基本运算
   - ✅ 控制流（if/while）
   - ✅ 函数调用
   - ✅ 递归

2. **C代码生成**
   - ✅ 表达式生成
   - ✅ 语句生成
   - ✅ 函数生成
   - ✅ 控制流生成

3. **工具链**
   - ✅ Bootstrap编译器（Python）
   - ✅ C代码生成器
   - ✅ Clang集成
   - ✅ az.py命令行工具

### 待实现 📋

1. **模块系统**
   - 📋 module声明解析
   - 📋 import语句解析
   - 📋 pub可见性控制
   - 📋 模块依赖解析
   - 📋 多文件编译

2. **数据结构**
   - 📋 struct结构体
   - 📋 enum枚举
   - 📋 数组类型
   - 📋 字符串类型

3. **高级特性**
   - 📋 for循环
   - 📋 match语句执行
   - 📋 泛型（后续）
   - 📋 所有权（后续）

## 快速实现计划

### 第1步：添加模块解析（今天）

```python
# 在Parser中添加
def parse_module_decl(self) -> Result:
    """解析module声明"""
    result = self.consume(TokenType.MODULE, "期望'module'")
    if not result.is_ok:
        return result
    
    # 解析模块路径: math.vector
    path_parts = []
    result = self.consume(TokenType.IDENTIFIER, "期望模块名")
    if not result.is_ok:
        return result
    path_parts.append(result.value.lexeme)
    
    while self.match(TokenType.DOT):
        result = self.consume(TokenType.IDENTIFIER, "期望模块名")
        if not result.is_ok:
            return result
        path_parts.append(result.value.lexeme)
    
    result = self.consume(TokenType.SEMICOLON, "期望';'")
    if not result.is_ok:
        return result
    
    return Result.Ok(Stmt(
        kind=StmtKind.MODULE_DECL,
        module_path='.'.join(path_parts)
    ))
```

### 第2步：添加pub可见性（今天）

```python
# 在Parser中修改
def parse_function(self) -> Result:
    """解析函数声明"""
    # 检查pub关键字
    is_public = self.match(TokenType.PUB)
    
    result = self.consume(TokenType.FN, "期望'fn'")
    if not result.is_ok:
        return result
    
    # ... 其余解析代码
    
    return Result.Ok(Stmt(
        kind=StmtKind.FUNC_DECL,
        name=name,
        is_public=is_public,  # 添加可见性标记
        # ...
    ))
```

### 第3步：添加struct支持（明天）

```python
def parse_struct(self) -> Result:
    """解析struct声明"""
    is_public = self.match(TokenType.PUB)
    
    result = self.consume(TokenType.STRUCT, "期望'struct'")
    if not result.is_ok:
        return result
    
    result = self.consume(TokenType.IDENTIFIER, "期望结构体名")
    if not result.is_ok:
        return result
    name = result.value.lexeme
    
    result = self.consume(TokenType.LEFT_BRACE, "期望'{'")
    if not result.is_ok:
        return result
    
    fields = []
    while not self.check(TokenType.RIGHT_BRACE):
        # 解析字段
        field_public = self.match(TokenType.PUB)
        
        result = self.consume(TokenType.IDENTIFIER, "期望字段名")
        if not result.is_ok:
            return result
        field_name = result.value.lexeme
        
        result = self.consume(TokenType.COLON, "期望':'")
        if not result.is_ok:
            return result
        
        result = self.consume(TokenType.IDENTIFIER, "期望类型名")
        if not result.is_ok:
            return result
        field_type = result.value.lexeme
        
        fields.append({
            'name': field_name,
            'type': field_type,
            'is_public': field_public
        })
        
        if not self.match(TokenType.COMMA):
            break
    
    result = self.consume(TokenType.RIGHT_BRACE, "期望'}'")
    if not result.is_ok:
        return result
    
    return Result.Ok(Stmt(
        kind=StmtKind.STRUCT_DECL,
        name=name,
        is_public=is_public,
        fields=fields
    ))
```

### 第4步：添加for循环（明天）

```python
def parse_for(self) -> Result:
    """解析for循环"""
    result = self.consume(TokenType.FOR, "期望'for'")
    if not result.is_ok:
        return result
    
    result = self.consume(TokenType.LEFT_PAREN, "期望'('")
    if not result.is_ok:
        return result
    
    # 初始化
    init = None
    if not self.check(TokenType.SEMICOLON):
        result = self.parse_var_declaration()
        if not result.is_ok:
            return result
        init = result.value
    else:
        self.advance()  # consume ';'
    
    # 条件
    condition = None
    if not self.check(TokenType.SEMICOLON):
        result = self.parse_expression()
        if not result.is_ok:
            return result
        condition = result.value
    
    result = self.consume(TokenType.SEMICOLON, "期望';'")
    if not result.is_ok:
        return result
    
    # 更新
    update = None
    if not self.check(TokenType.RIGHT_PAREN):
        result = self.parse_expression()
        if not result.is_ok:
            return result
        update = result.value
    
    result = self.consume(TokenType.RIGHT_PAREN, "期望')'")
    if not result.is_ok:
        return result
    
    # 循环体
    result = self.parse_statement()
    if not result.is_ok:
        return result
    body = result.value
    
    return Result.Ok(Stmt(
        kind=StmtKind.FOR,
        init=init,
        condition=condition,
        update=update,
        body=body
    ))
```

## 使用示例

### 示例1：简单模块

```az
// math.az
module math;

pub fn add(a: int, b: int) int {
    return a + b;
}

pub fn multiply(a: int, b: int) int {
    return a * b;
}
```

```az
// main.az
module main;

import math;

fn main() int {
    let result = math.add(3, 5);
    println("3 + 5 = " + result);
    return 0;
}
```

### 示例2：结构体模块

```az
// vector.az
module vector;

pub struct Vec3 {
    pub x: float,
    pub y: float,
    pub z: float
}

pub fn new(x: float, y: float, z: float) Vec3 {
    return Vec3 { x: x, y: y, z: z };
}

pub fn length(v: Vec3) float {
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}
```

```az
// main.az
module main;

import vector;

fn main() int {
    let v = vector.new(1.0, 2.0, 3.0);
    let len = vector.length(v);
    println("向量长度: " + len);
    return 0;
}
```

### 示例3：数组和for循环

```az
// array_utils.az
module array_utils;

pub fn sum(arr: []int, len: int) int {
    var total = 0;
    for (var i = 0; i < len; i = i + 1) {
        total = total + arr[i];
    }
    return total;
}

pub fn average(arr: []int, len: int) float {
    let total = sum(arr, len);
    return total / len;
}
```

```az
// main.az
module main;

import array_utils;

fn main() int {
    let numbers = [1, 2, 3, 4, 5];
    let total = array_utils.sum(numbers, 5);
    let avg = array_utils.average(numbers, 5);
    
    println("总和: " + total);
    println("平均值: " + avg);
    return 0;
}
```

## 编译命令

```bash
# 编译单文件
python az.py main.az

# 编译多模块项目
python az.py main.az --modules math.az vector.az array_utils.az

# 优化编译
python az.py main.az -O

# 编译并运行
python az.py main.az --run
```

## 总结

AZ语言的模块系统：
- ✅ **无头文件** - 简化开发
- ✅ **模块化设计** - 类似C3
- ✅ **pub/priv可见性** - 清晰的接口
- ✅ **编译时解析** - 快速编译
- ✅ **现代化语法** - 易于使用

**下一步**：立即实现模块解析和pub可见性！
