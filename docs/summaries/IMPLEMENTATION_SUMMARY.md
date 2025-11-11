# AZ编译器实现总结

## 概述

本项目实现了AZ编程语言的完整编译器，采用**C3风格的错误处理**（Result类型），提供了从源代码到执行的完整流程。

## 核心特性

### 1. C3风格的错误处理

这是本编译器最重要的特性，完全采用C3语言的错误处理方式：

**特点：**
- 使用Result类型而不是异常
- 明确的错误处理流程
- 编译时可检查错误处理
- 零运行时开销
- 适合系统编程

**实现：**
```python
class Result:
    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error
        self.is_ok = error is None
    
    @staticmethod
    def Ok(value):
        return Result(value=value)
    
    @staticmethod
    def Err(error):
        return Result(error=error)
```

**使用示例：**
```python
def compile_file(filename: str) -> Result:
    # 词法分析
    lexer = Lexer(source, filename)
    tokens_result = lexer.tokenize()
    if not tokens_result.is_ok:
        return tokens_result  # 传播错误
    
    # 语法分析
    parser = Parser(tokens_result.value, filename)
    ast_result = parser.parse()
    if not ast_result.is_ok:
        return ast_result  # 传播错误
    
    return Result.Ok(None)
```

### 2. 完整的编译流程

```
源代码 (.az)
    ↓
[词法分析] Lexer
    ↓ Result<[]Token>
[语法分析] Parser
    ↓ Result<Program>
[语义分析] Semantic Analyzer
    ↓ Result<void>
[代码生成] Interpreter/CodeGen
    ↓ Result<void>
执行结果
```

每个阶段都返回Result类型，确保错误能够被正确处理。

## 实现的组件

### 1. 词法分析器 (Lexer)

**文件：** `bootstrap/az_compiler.py` (Lexer类)

**功能：**
- 将源代码转换为token流
- 识别关键字、标识符、字面量、运算符
- 处理注释和空白字符
- 错误检测和报告

**支持的Token类型：**
- 关键字：fn, return, if, else, for, while, let, var, const, import, struct, enum, match, comptime
- 标识符和字面量：IDENTIFIER, INT_LITERAL, FLOAT_LITERAL, STRING_LITERAL
- 运算符：+, -, *, /, %, =, ==, !=, <, <=, >, >=, &&, ||, !
- 分隔符：(), {}, [], ,, ;, :, ., ->, |

**错误处理：**
```python
def scan_token(self) -> Result:
    # ... 扫描逻辑 ...
    if unknown_char:
        return Result.Err(CompileError(
            ErrorKind.LEXER_ERROR,
            f"未知字符: '{c}'",
            self.line,
            self.column,
            self.filename
        ))
```

### 2. 语法分析器 (Parser)

**文件：** `bootstrap/az_compiler.py` (Parser类)

**功能：**
- 将token流转换为抽象语法树(AST)
- 递归下降解析
- 运算符优先级处理
- 语法错误检测

**支持的语法结构：**
- 函数声明：`fn name(params) return_type { body }`
- 变量声明：`let/var name: type = value;`
- 控制流：if/else, while
- 表达式：二元运算、一元运算、函数调用
- 代码块：`{ statements }`

**运算符优先级：**
1. 逻辑或 (||)
2. 逻辑与 (&&)
3. 相等性 (==, !=)
4. 比较 (<, <=, >, >=)
5. 加减 (+, -)
6. 乘除模 (*, /, %)
7. 一元运算 (!, -)
8. 后缀运算 (函数调用, 成员访问)
9. 基本表达式 (字面量, 标识符, 括号)

**错误处理：**
```python
def parse_statement(self) -> Result:
    # ... 解析逻辑 ...
    if error:
        return Result.Err(CompileError(
            ErrorKind.PARSER_ERROR,
            "期望表达式",
            token.line,
            token.column,
            self.filename
        ))
```

### 3. 抽象语法树 (AST)

**文件：** `bootstrap/az_compiler.py` (AST类定义)

**节点类型：**

**表达式节点 (Expr):**
- INT_LITERAL: 整数字面量
- FLOAT_LITERAL: 浮点数字面量
- STRING_LITERAL: 字符串字面量
- IDENTIFIER: 标识符
- BINARY: 二元运算
- UNARY: 一元运算
- CALL: 函数调用
- MEMBER: 成员访问

**语句节点 (Stmt):**
- EXPRESSION: 表达式语句
- VAR_DECL: 变量声明
- FUNC_DECL: 函数声明
- RETURN: return语句
- IF: if语句
- WHILE: while语句
- BLOCK: 代码块
- IMPORT: import语句

### 4. 解释器 (Interpreter)

**文件：** `bootstrap/az_compiler.py` (Interpreter类)

**功能：**
- 遍历AST并执行
- 管理变量作用域
- 处理函数调用和返回值
- 运行时错误检测

**运行时值类型：**
- INT: 整数
- FLOAT: 浮点数
- STRING: 字符串
- BOOL: 布尔值
- VOID: 空值

**环境管理：**
```python
class Environment:
    def __init__(self, parent=None):
        self.variables = {}
        self.functions = {}
        self.parent = parent
```

**错误处理：**
```python
def evaluate_expression(self, expr: Expr) -> Result:
    # ... 求值逻辑 ...
    if error:
        return Result.Err(CompileError(
            ErrorKind.RUNTIME_ERROR,
            "除数不能为零",
            0, 0, ""
        ))
```

### 5. 错误报告

**文件：** `bootstrap/az_compiler.py` (CompileError类)

**错误类型：**
- LEXER_ERROR: 词法错误（未知字符、未终止的字符串等）
- PARSER_ERROR: 语法错误（缺少分号、括号不匹配等）
- SEMANTIC_ERROR: 语义错误（未定义的变量、函数等）
- TYPE_ERROR: 类型错误（类型不匹配、不支持的运算等）
- RUNTIME_ERROR: 运行时错误（除零、栈溢出等）

**错误报告格式：**
```
[错误] 词法错误 在 test.az:5:10
  未知字符: '@'
```

## 支持的语言特性

### 1. 数据类型
- int: 整数
- float: 浮点数
- string: 字符串
- bool: 布尔值
- void: 空类型

### 2. 变量声明
```az
let x = 10;              // 不可变
var y = 20;              // 可变
let name: string = "AZ"; // 带类型注解
```

### 3. 函数
```az
fn add(a: int, b: int) int {
    return a + b;
}

fn greet(name: string) void {
    println("Hello, " + name);
}
```

### 4. 控制流
```az
// if-else
if (x > 0) {
    println("正数");
} else {
    println("负数");
}

// while循环
var i = 0;
while (i < 10) {
    println(i);
    i = i + 1;
}
```

### 5. 运算符
- 算术：+, -, *, /, %
- 比较：==, !=, <, <=, >, >=
- 逻辑：&&, ||, !
- 赋值：=

### 6. 内置函数
- println(...): 打印并换行
- print(...): 打印不换行

## 示例程序

项目包含5个示例程序：

1. **hello.az** - Hello World
2. **variables.az** - 变量和运算
3. **functions.az** - 函数定义和调用
4. **control_flow.az** - 控制流（if, while）
5. **fibonacci.az** - 递归函数（斐波那契数列）

## 测试

### 运行单个示例
```bash
python bootstrap/az_compiler.py examples/hello.az
```

### 运行所有测试
```bash
# Windows
test_compiler.bat

# Linux/Mac
./test_compiler.sh
```

## 文件结构

```
AZ/
├── bootstrap/
│   ├── az_compiler.py      # 主编译器（1000+行）
│   └── README.md           # Bootstrap文档
├── compiler/               # AZ语言编写的编译器源码
│   ├── ast.az             # AST定义
│   ├── error.az           # 错误处理
│   ├── token.az           # Token定义
│   ├── lexer.az           # 词法分析器
│   ├── parser.az          # 语法分析器
│   ├── semantic.az        # 语义分析器
│   ├── codegen.az         # 代码生成器
│   └── main.az            # 主程序
├── examples/              # 示例程序
│   ├── hello.az
│   ├── variables.az
│   ├── functions.az
│   ├── control_flow.az
│   └── fibonacci.az
├── docs/                  # 文档
│   ├── README.md
│   ├── AZGC.md
│   ├── os-development.md
│   └── ownership-and-gc.md
├── README_COMPILER.md     # 编译器文档
├── QUICKSTART.md          # 快速入门
├── IMPLEMENTATION_SUMMARY.md  # 本文件
├── test_compiler.bat      # Windows测试脚本
└── test_compiler.sh       # Linux/Mac测试脚本
```

## 代码统计

- **Bootstrap编译器**: ~1000行Python代码
- **AZ编译器源码**: ~1500行AZ代码
- **示例程序**: ~200行AZ代码
- **文档**: ~3000行Markdown

## 技术亮点

### 1. C3风格错误处理
完全采用Result类型，避免异常机制，提供明确的错误处理流程。

### 2. 递归下降解析
使用递归下降解析器，代码清晰易懂，易于扩展。

### 3. 运算符优先级
正确实现了运算符优先级和结合性。

### 4. 作用域管理
使用环境链表管理变量作用域，支持嵌套作用域。

### 5. 递归函数
支持递归函数调用，如斐波那契数列。

### 6. 详细错误报告
提供详细的错误信息，包括文件名、行号、列号和错误描述。

## 性能特点

- **词法分析**: O(n)，n为源代码长度
- **语法分析**: O(n)，n为token数量
- **解释执行**: 直接遍历AST，无需生成中间代码

## 未来改进

### 短期（v0.2.0）
- [ ] 完整的类型系统
- [ ] 结构体和枚举
- [ ] 模式匹配
- [ ] for循环
- [ ] 数组和切片

### 中期（v0.3.0）
- [ ] 编译时执行（comptime）
- [ ] 泛型
- [ ] 接口/trait
- [ ] 更多标准库

### 长期（v1.0.0）
- [ ] LLVM代码生成
- [ ] 所有权系统
- [ ] AZGC垃圾回收器
- [ ] 包管理器（chim）
- [ ] 完整的标准库

## 与其他语言的对比

| 特性 | AZ | C3 | Zig | Rust |
|------|----|----|-----|------|
| 错误处理 | Result | Result | Error Union | Result |
| 内存管理 | 计划中 | 手动+可选GC | 手动 | 所有权 |
| 编译时执行 | 计划中 | 支持 | 支持 | 部分支持 |
| 类型系统 | 静态 | 静态 | 静态 | 静态 |
| 学习曲线 | 低 | 低 | 中 | 高 |

## 贡献指南

欢迎贡献！可以：
1. 报告bug
2. 提出新特性建议
3. 提交代码改进
4. 完善文档
5. 添加示例程序

## 许可证

本项目采用木兰宽松许可证2.0（Mulan Permissive License，Version 2）。

## 致谢

感谢以下项目的启发：
- C3语言 - 错误处理方式
- Zig语言 - 编译时执行理念
- Python - Bootstrap实现语言
- LLVM - 未来的代码生成后端

## 总结

本项目成功实现了一个完整的AZ编程语言编译器，核心特性是采用C3风格的Result类型错误处理。编译器包含词法分析、语法分析、语义分析和解释执行四个主要阶段，支持函数、变量、控制流等基本语言特性。项目提供了详细的文档和示例程序，可以立即使用。

这是一个坚实的基础，为未来添加更多高级特性（如所有权系统、LLVM代码生成、垃圾回收器等）做好了准备。
