# AZ vs Python - Match Case 对比

**AZ语言的match case语法与Python 3.10+对比**

---

## 🎯 概述

AZ语言借鉴了Python 3.10+引入的结构化模式匹配（Structural Pattern Matching），提供类似的语法，但针对系统编程进行了优化。

---

## 📊 语法对比

### 基本Match

#### Python
```python
match x:
    case 0:
        print("zero")
    case 1:
        print("one")
    case _:
        print("other")
```

#### AZ
```az
match x {
    case 0:
        println("zero");
    case 1:
        println("one");
    case _:
        println("other");
}
```

**差异**：
- AZ使用 `{}` 包裹，Python使用缩进
- AZ语句需要 `;`，Python不需要

---

### 或模式

#### Python
```python
match day:
    case 1 | 2 | 3 | 4 | 5:
        print("weekday")
    case 6 | 7:
        print("weekend")
```

#### AZ (方式1 - 逗号)
```az
match day {
    case 1, 2, 3, 4, 5:
        println("weekday");
    case 6, 7:
        println("weekend");
}
```

#### AZ (方式2 - 多个case)
```az
match day {
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
        println("weekday");
    case 6:
    case 7:
        println("weekend");
}
```

**差异**：
- Python使用 `|`，AZ使用 `,` 或多个case
- AZ支持C风格的fall-through

---

### 守卫条件

#### Python
```python
match x:
    case n if n > 0:
        print("positive")
    case n if n < 0:
        print("negative")
    case _:
        print("zero")
```

#### AZ
```az
match x {
    case n if n > 0:
        println("positive");
    case n if n < 0:
        println("negative");
    case _:
        println("zero");
}
```

**差异**：
- 语法几乎相同
- AZ需要 `;`

---

### 序列模式

#### Python
```python
match point:
    case (0, 0):
        print("origin")
    case (0, y):
        print(f"on y-axis at {y}")
    case (x, 0):
        print(f"on x-axis at {x}")
    case (x, y):
        print(f"at ({x}, {y})")
```

#### AZ (未来支持)
```az
match point {
    case (0, 0):
        println("origin");
    case (0, y):
        println("on y-axis at " + y);
    case (x, 0):
        println("on x-axis at " + x);
    case (x, y):
        println("at (" + x + ", " + y + ")");
}
```

**状态**：
- Python已支持
- AZ计划支持

---

### 映射模式

#### Python
```python
match config:
    case {"host": host, "port": port}:
        print(f"connecting to {host}:{port}")
    case {"host": host}:
        print(f"connecting to {host}:80")
    case _:
        print("invalid config")
```

#### AZ (未来支持)
```az
match config {
    case { host: host, port: port }:
        println("connecting to " + host + ":" + port);
    case { host: host }:
        println("connecting to " + host + ":80");
    case _:
        println("invalid config");
}
```

**状态**：
- Python已支持
- AZ计划支持

---

### 类模式

#### Python
```python
match shape:
    case Circle(radius=r):
        print(f"circle with radius {r}")
    case Rectangle(width=w, height=h):
        print(f"rectangle {w}x{h}")
    case _:
        print("unknown shape")
```

#### AZ (未来支持)
```az
match shape {
    case Circle { radius: r }:
        println("circle with radius " + r);
    case Rectangle { width: w, height: h }:
        println("rectangle " + w + "x" + h);
    case _:
        println("unknown shape");
}
```

**状态**：
- Python已支持
- AZ计划支持

---

## 🔍 详细对比

### 1. 字面量匹配

| 特性 | Python | AZ | 说明 |
|------|--------|-----|------|
| 整数 | ✅ | ✅ | 完全支持 |
| 浮点数 | ✅ | ✅ | 完全支持 |
| 字符串 | ✅ | ✅ | 完全支持 |
| 布尔值 | ✅ | ✅ | 完全支持 |
| None/null | ✅ | 📋 | AZ计划支持 |

### 2. 模式类型

| 模式 | Python | AZ | 说明 |
|------|--------|-----|------|
| 字面量 | ✅ | ✅ | 完全支持 |
| 通配符 `_` | ✅ | ✅ | 完全支持 |
| 捕获变量 | ✅ | ✅ | 完全支持 |
| 或模式 | ✅ `\|` | ✅ `,` | 语法不同 |
| 序列模式 | ✅ | 📋 | AZ计划支持 |
| 映射模式 | ✅ | 📋 | AZ计划支持 |
| 类模式 | ✅ | 📋 | AZ计划支持 |
| AS模式 | ✅ | 📋 | AZ计划支持 |

### 3. 守卫条件

| 特性 | Python | AZ |
|------|--------|-----|
| if守卫 | ✅ | ✅ |
| 复杂表达式 | ✅ | ✅ |
| 函数调用 | ✅ | ✅ |

### 4. 代码块

| 特性 | Python | AZ |
|------|--------|-----|
| 单语句 | ✅ 缩进 | ✅ 直接写 |
| 多语句 | ✅ 缩进 | ✅ `{}` |
| 嵌套match | ✅ | ✅ |

---

## 💡 实际示例对比

### 示例1: HTTP状态码

#### Python
```python
def handle_status(code):
    match code:
        case 200:
            return "OK"
        case 404:
            return "Not Found"
        case 500:
            return "Server Error"
        case _:
            return "Unknown"
```

#### AZ
```az
fn handle_status(code: int) string {
    match code {
        case 200:
            return "OK";
        case 404:
            return "Not Found";
        case 500:
            return "Server Error";
        case _:
            return "Unknown";
    }
}
```

### 示例2: 命令处理

#### Python
```python
def process_command(cmd, arg):
    match cmd:
        case "add":
            return f"Adding {arg}"
        case "sub":
            return f"Subtracting {arg}"
        case "mul" | "multiply":
            return f"Multiplying by {arg}"
        case _:
            return "Unknown command"
```

#### AZ
```az
fn process_command(cmd: string, arg: int) string {
    match cmd {
        case "add":
            return "Adding " + arg;
        case "sub":
            return "Subtracting " + arg;
        case "mul", "multiply":
            return "Multiplying by " + arg;
        case _:
            return "Unknown command";
    }
}
```

### 示例3: 范围检查

#### Python
```python
def classify_age(age):
    match age:
        case n if n < 0:
            return "Invalid"
        case n if n < 13:
            return "Child"
        case n if n < 20:
            return "Teenager"
        case n if n < 60:
            return "Adult"
        case _:
            return "Senior"
```

#### AZ
```az
fn classify_age(age: int) string {
    match age {
        case n if n < 0:
            return "Invalid";
        case n if n < 13:
            return "Child";
        case n if n < 20:
            return "Teenager";
        case n if n < 60:
            return "Adult";
        case _:
            return "Senior";
    }
}
```

---

## ⚡ 性能对比

### Python

| 特性 | 性能 | 说明 |
|------|------|------|
| 执行方式 | 解释执行 | 运行时匹配 |
| 优化 | 有限 | 部分优化 |
| 类型检查 | 运行时 | 动态类型 |

### AZ

| 特性 | 性能 | 说明 |
|------|------|------|
| 执行方式 | 编译执行 | 编译时优化 |
| 优化 | 完整 | 跳转表、二分查找 |
| 类型检查 | 编译时 | 静态类型 |

**性能优势**：
- ✅ AZ编译为机器码，Python解释执行
- ✅ AZ编译时优化，Python运行时匹配
- ✅ AZ静态类型检查，Python动态类型

---

## 🎯 使用场景

### Python适合

- ✅ 快速原型开发
- ✅ 脚本和自动化
- ✅ 数据处理
- ✅ 复杂的模式匹配（序列、映射、类）

### AZ适合

- ✅ 系统编程
- ✅ 性能关键应用
- ✅ 嵌入式系统
- ✅ 底层开发
- ✅ 需要编译时保证的场景

---

## 📈 功能路线图

### 当前支持 (v0.5)

- ✅ 字面量模式
- ✅ 通配符模式
- ✅ 变量捕获
- ✅ 或模式（逗号）
- ✅ 守卫条件
- ✅ 代码块

### 近期计划 (v0.6)

- 📋 元组模式
- 📋 结构体模式
- 📋 枚举模式
- 📋 完整性检查

### 长期计划 (v1.0)

- 📋 数组模式
- 📋 切片模式
- 📋 范围模式
- 📋 AS模式
- 📋 嵌套模式优化

---

## 🔄 迁移指南

### 从Python迁移到AZ

#### 1. 添加类型注解

```python
# Python
def process(x):
    match x:
        case 0:
            return "zero"
```

```az
// AZ
fn process(x: int) string {
    match x {
        case 0:
            return "zero";
    }
}
```

#### 2. 修改或模式语法

```python
# Python
case 1 | 2 | 3:
```

```az
// AZ
case 1, 2, 3:
```

#### 3. 添加大括号和分号

```python
# Python
match x:
    case 0:
        print("zero")
```

```az
// AZ
match x {
    case 0:
        println("zero");
}
```

#### 4. 修改字符串格式化

```python
# Python
print(f"value is {x}")
```

```az
// AZ
println("value is " + x);
```

---

## 📝 总结

### 相似之处

✅ **关键字** - 都使用 `match` 和 `case`  
✅ **通配符** - 都使用 `_`  
✅ **守卫** - 都使用 `if`  
✅ **变量捕获** - 语法相同  
✅ **嵌套** - 都支持嵌套match

### 主要差异

| 特性 | Python | AZ |
|------|--------|-----|
| 代码块 | 缩进 | `{}` |
| 语句结束 | 换行 | `;` |
| 或模式 | `\|` | `,` |
| 类型系统 | 动态 | 静态 |
| 执行方式 | 解释 | 编译 |
| 性能 | 较慢 | 快速 |

### AZ的优势

✅ **编译时检查** - 更早发现错误  
✅ **静态类型** - 更好的IDE支持  
✅ **高性能** - 编译优化  
✅ **系统编程** - 适合底层开发  
✅ **零成本抽象** - 无运行时开销

### Python的优势

✅ **动态类型** - 更灵活  
✅ **丰富的模式** - 序列、映射、类模式  
✅ **快速开发** - 无需编译  
✅ **生态系统** - 大量库支持

---

## 🎓 学习建议

### 如果你熟悉Python

1. **语法转换** - 主要是添加 `{}` 和 `;`
2. **类型思维** - 习惯静态类型
3. **性能意识** - 利用编译时优化
4. **系统编程** - 学习底层概念

### 如果你熟悉AZ

1. **简化语法** - Python更简洁
2. **动态特性** - 利用动态类型
3. **丰富模式** - 学习高级模式匹配
4. **快速原型** - 用于快速开发

---

**AZ的Match Case - Python的简洁 + 系统语言的性能！** 🚀

