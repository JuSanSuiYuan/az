# AZ标准库文档

AZ语言标准库提供了丰富的功能模块，帮助你快速开发应用程序。

## 📚 模块列表

### 核心模块

| 模块 | 说明 | 状态 |
|------|------|------|
| `std.io` | 输入输出 | ✅ 可用 |
| `std.string` | 字符串操作 | ✅ 可用 |
| `std.math` | 数学函数 | ✅ 可用 |
| `std.fs` | 文件系统 | ✅ 可用 |
| `std.collections` | 集合类型 | ⚠️ 部分可用 |
| `std.mem` | 内存管理 | ✅ 可用 |
| `std.os` | 操作系统接口 | ⚠️ 部分可用 |
| `std.time` | 时间和日期 | 📋 计划中 |
| `std.convert` | 类型转换 | ✅ 可用 |
| `std.error` | 错误处理 | 📋 计划中 |

## 🚀 快速开始

### 导入模块

```az
import std.io;
import std.string;
import std.fs;

fn main() int {
    println("Hello, AZ!");
    return 0;
}
```

## 📖 模块详解

### std.io - 输入输出

基础的输入输出功能。

```az
import std.io;

fn main() int {
    // 输出
    println("Hello, World!");
    print("不换行输出");
    
    // 输入
    let line = az_read_line();
    println("你输入了: " + line);
    
    return 0;
}
```

**可用函数**：
- `println(s: string)` - 打印并换行
- `print(s: string)` - 打印不换行
- `az_read_line() string` - 读取一行输入

### std.string - 字符串操作

强大的字符串处理功能。

```az
import std.string;

fn main() int {
    let s1 = "Hello";
    let s2 = "World";
    
    // 连接
    let combined = az_string_concat(s1, s2);
    
    // 长度
    let len = az_string_length(combined);
    
    // 大小写转换
    let upper = az_string_to_upper(combined);
    let lower = az_string_to_lower(combined);
    
    // 查找
    let pos = az_string_find(combined, "World");
    
    // 替换
    let replaced = az_string_replace(combined, "World", "AZ");
    
    // 去除空白
    let trimmed = az_string_trim("  hello  ");
    
    return 0;
}
```

**可用函数**：
- `az_string_concat(a, b)` - 连接字符串
- `az_string_length(s)` - 获取长度
- `az_string_substring(s, start, end)` - 获取子串
- `az_string_equals(a, b)` - 比较字符串
- `az_string_to_upper(s)` - 转大写
- `az_string_to_lower(s)` - 转小写
- `az_string_find(s, sub)` - 查找子串
- `az_string_contains(s, sub)` - 是否包含
- `az_string_starts_with(s, prefix)` - 是否开头匹配
- `az_string_ends_with(s, suffix)` - 是否结尾匹配
- `az_string_trim(s)` - 去除空白
- `az_string_replace(s, old, new)` - 替换
- `az_string_repeat(s, count)` - 重复
- `az_string_reverse(s)` - 反转

### std.math - 数学函数

完整的数学运算支持。

```az
import std.math;

fn main() int {
    // 基础运算
    let sqrt_val = az_sqrt(16.0);
    let pow_val = az_pow(2.0, 3.0);
    let abs_val = az_abs(-5.5);
    
    // 三角函数
    let sin_val = az_sin(3.14159 / 2.0);
    let cos_val = az_cos(0.0);
    
    // 对数和指数
    let exp_val = az_exp(1.0);
    let log_val = az_log(2.718);
    
    // 取整
    let floor_val = az_floor(3.7);
    let ceil_val = az_ceil(3.2);
    let round_val = az_round(3.5);
    
    // 最大最小值
    let max_val = az_max(10, 20);
    let min_val = az_min(10, 20);
    
    // 限制范围
    let clamped = az_clamp(15, 0, 10);
    
    return 0;
}
```

**数学常量**：
- `PI` = 3.14159265358979323846
- `E` = 2.71828182845904523536
- `TAU` = 6.28318530717958647692
- `PHI` = 1.61803398874989484820

### std.fs - 文件系统

文件和目录操作。

```az
import std.fs;

fn main() int {
    // 读取文件
    let content = az_read_file("input.txt");
    if (content != null) {
        println("文件内容: " + content);
    }
    
    // 写入文件
    let result = az_write_file("output.txt", "Hello, File!");
    if (result == 0) {
        println("写入成功");
    }
    
    // 追加到文件
    az_append_file("output.txt", "\n新的一行");
    
    // 检查文件是否存在
    if (az_file_exists("test.txt")) {
        println("文件存在");
    }
    
    // 获取文件大小
    let size = az_file_size("test.txt");
    
    // 检查是否是文件/目录
    if (az_is_file("test.txt")) {
        println("这是一个文件");
    }
    if (az_is_dir("mydir")) {
        println("这是一个目录");
    }
    
    // 创建目录
    az_create_dir("newdir");
    
    // 删除文件
    az_remove_file("temp.txt");
    
    // 重命名文件
    az_rename_file("old.txt", "new.txt");
    
    return 0;
}
```

### std.collections - 集合类型

动态数组和其他集合。

```az
import std.collections;

fn main() int {
    // 创建动态数组
    let vec = az_vec_new();
    
    // 添加元素
    az_vec_push(vec, "Hello");
    az_vec_push(vec, "World");
    
    // 获取元素
    let item = az_vec_get(vec, 0);
    
    // 获取长度
    let len = az_vec_len(vec);
    
    // 插入元素
    az_vec_insert(vec, 1, "Beautiful");
    
    // 删除元素
    let removed = az_vec_remove(vec, 1);
    
    // 清空
    az_vec_clear(vec);
    
    // 释放
    az_vec_free(vec);
    
    return 0;
}
```

### std.mem - 内存管理

底层内存操作。

```az
import std.mem;

fn main() int {
    // 分配内存
    let ptr = az_malloc(1024);
    
    // 使用内存
    // ...
    
    // 释放内存
    az_free(ptr);
    
    // 重新分配
    let new_ptr = az_realloc(ptr, 2048);
    
    return 0;
}
```

### std.os - 操作系统接口

与操作系统交互。

```az
import std.os;

fn main() int {
    // 获取环境变量
    let path = az_getenv("PATH");
    if (path != null) {
        println("PATH: " + path);
    }
    
    // 设置环境变量
    az_setenv("MY_VAR", "my_value");
    
    // 执行系统命令
    let result = az_system("ls -la");
    
    // 获取进程ID
    let pid = az_getpid();
    
    // 睡眠
    az_sleep_millis(1000);  // 睡眠1秒
    
    return 0;
}
```

### std.convert - 类型转换

各种类型之间的转换。

```az
import std.convert;

fn main() int {
    // 整数转字符串
    let str = az_int_to_string(42);
    
    // 字符串转整数
    let num = az_string_to_int("123");
    
    // 浮点数转字符串
    let float_str = az_float_to_string(3.14);
    
    // 字符串转浮点数
    let float_num = az_string_to_float("2.718");
    
    return 0;
}
```

### std.time - 时间和日期

时间相关操作（计划中）。

```az
import std.time;

fn main() int {
    // 获取当前时间
    let now = az_time_now();
    
    // 睡眠
    az_sleep_millis(1000);
    
    return 0;
}
```

## 🎯 完整示例

### 文件处理工具

```az
import std.io;
import std.fs;
import std.string;

fn process_file(input_path: string, output_path: string) int {
    // 检查文件是否存在
    if (!az_file_exists(input_path)) {
        println("错误: 文件不存在 - " + input_path);
        return 1;
    }
    
    // 读取文件
    let content = az_read_file(input_path);
    if (content == null) {
        println("错误: 无法读取文件");
        return 1;
    }
    
    // 处理内容（转大写）
    let processed = az_string_to_upper(content);
    
    // 写入新文件
    let result = az_write_file(output_path, processed);
    if (result != 0) {
        println("错误: 无法写入文件");
        return 1;
    }
    
    println("处理完成!");
    return 0;
}

fn main() int {
    return process_file("input.txt", "output.txt");
}
```

### 字符串处理

```az
import std.io;
import std.string;

fn main() int {
    let text = "  Hello, World!  ";
    
    // 去除空白
    let trimmed = az_string_trim(text);
    println("去除空白: " + trimmed);
    
    // 转大写
    let upper = az_string_to_upper(trimmed);
    println("大写: " + upper);
    
    // 替换
    let replaced = az_string_replace(upper, "WORLD", "AZ");
    println("替换: " + replaced);
    
    // 重复
    let repeated = az_string_repeat("*", 10);
    println("重复: " + repeated);
    
    // 反转
    let reversed = az_string_reverse(trimmed);
    println("反转: " + reversed);
    
    return 0;
}
```

### 数学计算

```az
import std.io;
import std.math;
import std.convert;

fn main() int {
    let x = 16.0;
    
    // 平方根
    let sqrt_x = az_sqrt(x);
    println("sqrt(16) = " + az_float_to_string(sqrt_x));
    
    // 幂运算
    let pow_x = az_pow(2.0, 8.0);
    println("2^8 = " + az_float_to_string(pow_x));
    
    // 三角函数
    let pi = 3.14159265359;
    let sin_val = az_sin(pi / 2.0);
    println("sin(π/2) = " + az_float_to_string(sin_val));
    
    // 对数
    let e = 2.71828182846;
    let log_e = az_log(e);
    println("ln(e) = " + az_float_to_string(log_e));
    
    return 0;
}
```

## 📝 注意事项

1. **内存管理**: 使用`az_malloc`分配的内存需要手动`az_free`
2. **字符串**: 大部分字符串函数返回新分配的字符串，需要管理内存
3. **错误处理**: 检查返回值，很多函数在失败时返回`null`或`-1`
4. **平台差异**: 某些功能在不同平台上可能有差异

## 🔧 编译选项

使用标准库时，确保链接运行时库：

```bash
python az.py your_program.az
```

运行时库会自动链接。

## 📚 更多资源

- [AZ语言文档](../README.md)
- [快速开始指南](../QUICK_USE_GUIDE.md)
- [示例程序](../examples/)

## 🤝 贡献

欢迎贡献新的标准库模块！请查看[贡献指南](../CONTRIBUTING.md)。
