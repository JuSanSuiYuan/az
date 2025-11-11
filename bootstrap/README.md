# AZ编译器 Bootstrap版本

这是AZ编程语言的Bootstrap编译器实现，使用Python编写。

## 特性

- **C3风格的错误处理**：采用Result类型进行错误处理，而不是异常机制
- **完整的编译流程**：词法分析 → 语法分析 → 语义分析 → 代码执行
- **解释执行**：当前版本通过解释器直接执行AST
- **支持的语法**：
  - 函数声明和调用
  - 变量声明（let/var）
  - 基本数据类型（int, float, string, bool）
  - 运算符（算术、比较、逻辑）
  - 控制流（if/else, while）
  - 递归函数

## 使用方法

### 运行示例程序

```bash
# Hello World
python bootstrap/az_compiler.py examples/hello.az

# 变量和运算
python bootstrap/az_compiler.py examples/variables.az

# 函数
python bootstrap/az_compiler.py examples/functions.az

# 控制流
python bootstrap/az_compiler.py examples/control_flow.az

# 斐波那契数列
python bootstrap/az_compiler.py examples/fibonacci.az
```

### 编写自己的程序

创建一个 `.az` 文件，例如 `test.az`：

```az
import std.io;

fn main() int {
    println("Hello from AZ!");
    return 0;
}
```

然后运行：

```bash
python bootstrap/az_compiler.py test.az
```

## 编译器架构

### 1. 词法分析器 (Lexer)
- 将源代码转换为token流
- 识别关键字、标识符、字面量、运算符等
- 处理注释和空白字符

### 2. 语法分析器 (Parser)
- 将token流转换为抽象语法树(AST)
- 使用递归下降解析
- 支持运算符优先级

### 3. 解释器 (Interpreter)
- 遍历AST并执行
- 管理变量作用域
- 处理函数调用和返回值

### 4. 错误处理
- 采用C3风格的Result类型
- 详细的错误信息，包含位置和描述
- 不使用异常机制

## C3风格的错误处理

AZ编译器采用C3语言的错误处理方式，使用Result类型而不是异常：

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

使用示例：

```python
result = compile_file(filename)
if result.is_ok:
    print("编译成功")
else:
    result.error.report()
```

这种方式的优点：
- 明确的错误处理流程
- 编译时可以检查是否处理了错误
- 避免异常带来的性能开销
- 更符合系统编程语言的风格

## 支持的语法

### 变量声明
```az
let x = 10;        // 不可变变量
var y = 20;        // 可变变量
let name: string = "AZ";  // 带类型注解
```

### 函数声明
```az
fn add(a: int, b: int) int {
    return a + b;
}
```

### 控制流
```az
if (x > 5) {
    println("x大于5");
} else {
    println("x小于等于5");
}

while (i < 10) {
    println(i);
    i = i + 1;
}
```

### 运算符
- 算术：`+`, `-`, `*`, `/`, `%`
- 比较：`==`, `!=`, `<`, `<=`, `>`, `>=`
- 逻辑：`&&`, `||`, `!`

### 内置函数
- `println(...)` - 打印并换行
- `print(...)` - 打印不换行

## 未来计划

- [ ] 完整的类型系统
- [ ] 结构体和枚举
- [ ] 模式匹配
- [ ] 编译时执行（comptime）
- [ ] LLVM代码生成
- [ ] 标准库
- [ ] 包管理器集成

## 许可证

本项目采用木兰宽松许可证2.0（Mulan Permissive License，Version 2）。
