# AZ语言快速使用指南

**5分钟上手AZ语言！**

---

## 🚀 快速开始

### 1. 编写第一个程序

创建 `hello.az`:

```az
fn main() int {
    println("Hello, AZ!");
    return 0;
}
```

### 2. 编译并运行

```bash
python az.py hello.az --run
```

输出：
```
[1/3] 编译 hello.az -> hello.c
[2/3] 编译 hello.c -> hello.exe
[3/3] 清理临时文件

✅ 编译成功!
   输出: hello.exe
   耗时: 0.36秒

==================================================
运行: hello.exe
==================================================

Hello, AZ!

==================================================
程序退出码: 0
==================================================
```

**就这么简单！** ✅

---

## 📚 基础语法

### 变量

```az
fn main() int {
    let x = 10;           // 不可变
    var y = 20;           // 可变
    let name = "AZ";      // 字符串
    
    y = y + 1;            // 可以修改
    // x = x + 1;         // 错误！x不可变
    
    return 0;
}
```

### 函数

```az
fn add(a: int, b: int) int {
    return a + b;
}

fn greet(name: string) void {
    println("Hello, " + name);
}

fn main() int {
    let sum = add(3, 5);
    greet("World");
    return 0;
}
```

### 控制流

```az
fn main() int {
    // if语句
    let x = 10;
    if (x > 5) {
        println("x大于5");
    } else {
        println("x小于等于5");
    }
    
    // while循环
    var i = 0;
    while (i < 5) {
        println("i = " + i);
        i = i + 1;
    }
    
    return 0;
}
```

### 递归

```az
fn factorial(n: int) int {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}

fn main() int {
    let result = factorial(5);
    println("5! = " + result);
    return 0;
}
```

---

## 🛠️ 使用标准库

### 文件操作

```az
fn main() int {
    // 读取文件
    let content = az_read_file("input.txt");
    
    // 处理内容
    let upper = az_string_to_upper(content);
    
    // 写入文件
    az_write_file("output.txt", upper);
    
    println("文件处理完成！");
    return 0;
}
```

### 字符串操作

```az
fn main() int {
    let str1 = "Hello";
    let str2 = "World";
    
    // 连接字符串
    let combined = az_string_concat(str1, str2);
    
    // 获取长度
    let len = az_string_length(combined);
    
    // 转大写
    let upper = az_string_to_upper(combined);
    
    println(upper);
    return 0;
}
```

---

## 💻 命令行选项

### 基本编译

```bash
python az.py program.az
```

### 指定输出文件

```bash
python az.py program.az -o myprogram
```

### 优化编译

```bash
python az.py program.az -O
```

### 编译并运行

```bash
python az.py program.az --run
```

### 保留C代码

```bash
python az.py program.az --keep-c
```

### 详细输出

```bash
python az.py program.az -v
```

### 组合使用

```bash
python az.py program.az -O --run -v
```

---

## 📝 完整示例

### 示例1: 文件处理工具

```az
// file_processor.az
fn main() int {
    println("文件处理工具");
    
    // 读取文件
    let content = az_read_file("data.txt");
    if (content == null) {
        println("错误：无法读取文件");
        return 1;
    }
    
    // 转换为大写
    let processed = az_string_to_upper(content);
    
    // 写入新文件
    let result = az_write_file("output.txt", processed);
    if (result != 0) {
        println("错误：无法写入文件");
        return 1;
    }
    
    println("处理完成！");
    return 0;
}
```

编译运行：
```bash
python az.py file_processor.az --run
```

### 示例2: 数学计算

```az
// calculator.az
fn fibonacci(n: int) int {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

fn main() int {
    println("斐波那契数列前10项:");
    
    var i = 0;
    while (i < 10) {
        let fib = fibonacci(i);
        println("fib(" + i + ") = " + fib);
        i = i + 1;
    }
    
    return 0;
}
```

### 示例3: 字符串处理

```az
// string_tool.az
fn process_string(input: string) string {
    // 转大写
    let upper = az_string_to_upper(input);
    
    // 添加前缀
    let result = az_string_concat("处理结果: ", upper);
    
    return result;
}

fn main() int {
    let input = "hello world";
    let output = process_string(input);
    
    println(output);
    return 0;
}
```

---

## 🎯 最佳实践

### 1. 使用明确的类型

```az
// ✅ 好
fn add(a: int, b: int) int {
    return a + b;
}

// ❌ 不好（类型推导还不完善）
fn add(a, b) {
    return a + b;
}
```

### 2. 检查错误

```az
// ✅ 好
let content = az_read_file("data.txt");
if (content == null) {
    println("错误：无法读取文件");
    return 1;
}

// ❌ 不好（不检查错误）
let content = az_read_file("data.txt");
// 直接使用content，可能为null
```

### 3. 使用有意义的变量名

```az
// ✅ 好
let user_name = "Alice";
let total_count = 100;

// ❌ 不好
let x = "Alice";
let n = 100;
```

### 4. 适当的注释

```az
// ✅ 好
// 计算阶乘
fn factorial(n: int) int {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}

// ❌ 不好（没有注释）
fn f(n: int) int {
    if (n <= 1) {
        return 1;
    }
    return n * f(n - 1);
}
```

---

## 🐛 常见问题

### Q: 编译失败怎么办？

**A**: 检查错误信息，常见问题：
- 语法错误（缺少分号、括号不匹配）
- 类型错误（类型不匹配）
- 未定义的变量或函数

### Q: 运行时崩溃怎么办？

**A**: 可能的原因：
- 空指针访问
- 数组越界
- 除零错误

使用 `--keep-c` 选项保留C代码，查看生成的代码。

### Q: 性能不够好怎么办？

**A**: 使用优化选项：
```bash
python az.py program.az -O
```

### Q: 需要更多功能怎么办？

**A**: 
- 查看标准库文档
- 直接使用C函数（在运行时库中添加）
- 等待后续更新

---

## 📚 更多资源

- **完整文档**: [README.md](README.md)
- **语言对比**: [AZ vs C3](AZ_VS_C3.md), [AZ vs Zig](AZ_VS_ZIG.md)
- **技术栈**: [TECH_STACK.md](TECH_STACK.md)
- **当前状态**: [CURRENT_STATUS.md](CURRENT_STATUS.md)
- **生产就绪**: [PRODUCTION_READY_STATUS.md](PRODUCTION_READY_STATUS.md)

---

## 🎉 开始使用

```bash
# 1. 创建程序
echo 'fn main() int { println("Hello, AZ!"); return 0; }' > hello.az

# 2. 编译运行
python az.py hello.az --run

# 3. 享受AZ语言！
```

**就这么简单！开始你的AZ之旅吧！** 🚀
