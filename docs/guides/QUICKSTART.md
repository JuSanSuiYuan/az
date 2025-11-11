# AZ编程语言快速入门

## 安装

### 前提条件
- Python 3.7 或更高版本

### 验证安装
```bash
python --version
# 或
python3 --version
```

## 第一个程序

### 1. 创建文件 `hello.az`

```az
import std.io;

fn main() int {
    println("Hello, AZ!");
    return 0;
}
```

### 2. 运行程序

**Windows:**
```cmd
python bootstrap\az_compiler.py hello.az
```

**Linux/Mac:**
```bash
python3 bootstrap/az_compiler.py hello.az
```

### 3. 输出
```
AZ编译器 v0.1.0
采用C3风格的错误处理

正在编译: hello.az
[1/4] 词法分析...
  生成了 11 个token
[2/4] 语法分析...
  生成了 2 个顶层语句
[3/4] 语义分析...
  语义检查通过
[4/4] 执行程序...
---输出---
Hello, AZ!
----------

编译成功！
```

## 基本语法

### 变量

```az
// 不可变变量（推荐）
let x = 10;
let name = "AZ";

// 可变变量
var count = 0;
count = count + 1;

// 带类型注解
let age: int = 25;
let pi: float = 3.14;
```

### 函数

```az
// 无返回值函数
fn greet(name: string) void {
    println("Hello, " + name);
}

// 有返回值函数
fn add(a: int, b: int) int {
    return a + b;
}

// 调用函数
fn main() int {
    greet("World");
    let sum = add(5, 3);
    println(sum);
    return 0;
}
```

### 控制流

```az
// if-else
fn check_number(x: int) void {
    if (x > 0) {
        println("正数");
    } else if (x < 0) {
        println("负数");
    } else {
        println("零");
    }
}

// while循环
fn count_to_ten() void {
    var i = 1;
    while (i <= 10) {
        println(i);
        i = i + 1;
    }
}
```

### 运算符

```az
fn operators_demo() void {
    // 算术运算
    let a = 10 + 5;   // 15
    let b = 10 - 5;   // 5
    let c = 10 * 5;   // 50
    let d = 10 / 5;   // 2
    let e = 10 % 3;   // 1
    
    // 比较运算
    let eq = (5 == 5);    // true
    let ne = (5 != 3);    // true
    let lt = (3 < 5);     // true
    let gt = (5 > 3);     // true
    
    // 逻辑运算
    let and = (true && false);  // false
    let or = (true || false);   // true
    let not = !true;            // false
}
```

## 示例程序

### 计算阶乘

```az
import std.io;

fn factorial(n: int) int {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}

fn main() int {
    let n = 5;
    let result = factorial(n);
    
    print("factorial(");
    print(n);
    print(") = ");
    println(result);
    
    return 0;
}
```

### 判断素数

```az
import std.io;

fn is_prime(n: int) int {
    if (n <= 1) {
        return 0;
    }
    
    var i = 2;
    while (i * i <= n) {
        if (n % i == 0) {
            return 0;
        }
        i = i + 1;
    }
    
    return 1;
}

fn main() int {
    var num = 2;
    println("前10个素数:");
    
    var count = 0;
    while (count < 10) {
        if (is_prime(num) == 1) {
            println(num);
            count = count + 1;
        }
        num = num + 1;
    }
    
    return 0;
}
```

### 最大公约数

```az
import std.io;

fn gcd(a: int, b: int) int {
    while (b != 0) {
        let temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

fn main() int {
    let a = 48;
    let b = 18;
    let result = gcd(a, b);
    
    print("gcd(");
    print(a);
    print(", ");
    print(b);
    print(") = ");
    println(result);
    
    return 0;
}
```

## 运行示例程序

项目包含了多个示例程序，可以直接运行：

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

或者运行测试脚本：

**Windows:**
```cmd
test_compiler.bat
```

**Linux/Mac:**
```bash
chmod +x test_compiler.sh
./test_compiler.sh
```

## C3风格的错误处理

AZ采用C3语言的错误处理方式，使用Result类型：

```az
// 函数返回Result类型
fn divide(a: int, b: int) Result<int> {
    if (b == 0) {
        return Result.Err(make_error(
            ErrorKind.RuntimeError,
            "除数不能为零",
            0, 0, ""
        ));
    }
    return Result.Ok(a / b);
}

// 检查Result
fn main() int {
    let result = divide(10, 2);
    if (result is Ok) {
        println(result.value);
    } else {
        report_error(result.error);
    }
    return 0;
}
```

## 常见错误

### 1. 忘记分号
```az
let x = 10  // 错误：期望 ';'
```

**正确写法：**
```az
let x = 10;
```

### 2. 未定义的变量
```az
fn main() int {
    println(x);  // 错误：未定义的变量: x
    return 0;
}
```

**正确写法：**
```az
fn main() int {
    let x = 10;
    println(x);
    return 0;
}
```

### 3. 类型不匹配
```az
let x: int = "hello";  // 错误：类型不匹配
```

**正确写法：**
```az
let x: string = "hello";
```

### 4. 缺少main函数
```az
fn hello() void {
    println("Hello");
}
// 错误：未找到main函数
```

**正确写法：**
```az
fn hello() void {
    println("Hello");
}

fn main() int {
    hello();
    return 0;
}
```

## 下一步

- 阅读 [README_COMPILER.md](../../README_COMPILER.md) 了解编译器架构
- 查看 [examples/](../../examples/) 目录中的更多示例
- 阅读 [docs/](../../docs/) 目录中的语言设计文档

## 获取帮助

如果遇到问题：
1. 检查错误信息，它会告诉你问题所在
2. 查看示例程序，学习正确的语法
3. 阅读文档，了解语言特性

## 许可证

本项目采用木兰宽松许可证2.0（Mulan Permissive License，Version 2）。
