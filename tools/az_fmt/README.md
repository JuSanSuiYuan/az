# AZ fmt - AZ语言代码格式化工具

**类似于rustfmt的AZ代码格式化工具**

---

## 📖 简介

AZ fmt是AZ语言的官方代码格式化工具，自动格式化AZ代码，确保代码风格一致。

### 特性

- ✅ 自动格式化AZ代码
- ✅ 可配置的格式化规则
- ✅ 支持检查模式（不修改文件）
- ✅ 支持批量格式化
- ✅ 保留注释
- ✅ 智能缩进
- ✅ 对齐结构体字段
- ✅ 统一空格和换行

---

## 🚀 快速开始

### 安装

```bash
# 添加到PATH（可选）
export PATH=$PATH:/path/to/az/tools/az_fmt
```

### 基本用法

```bash
# 格式化单个文件
python tools/az_fmt/azfmt.py file.az  # 使用az fmt格式化单个文件

# 格式化多个文件
python tools/az_fmt/azfmt.py file1.az file2.az  # 使用az fmt格式化多个文件

# 检查格式（不修改文件）
python tools/az_fmt/azfmt.py --check file.az  # 使用az fmt检查格式

# 使用配置文件
python tools/az_fmt/azfmt.py --config azfmt.toml file.az  # 使用az fmt配置文件

# 自定义缩进
python tools/az_fmt/azfmt.py --indent 2 file.az  # 使用az fmt自定义缩进

# 自定义行宽
python tools/az_fmt/azfmt.py --max-width 120 file.az  # 使用az fmt自定义行宽
```

---

## 📋 格式化规则

### 1. 缩进

**默认**: 4个空格

```az
// 格式化前
fn main() int {
return 0;
}

// 格式化后
fn main() int {
    return 0;
}
```

### 2. 空格

**大括号前**: 添加空格

```az
// 格式化前
fn main()int{
    return 0;
}

// 格式化后
fn main() int {
    return 0;
}
```

**逗号后**: 添加空格

```az
// 格式化前
fn add(a:int,b:int) int {
    return a+b;
}

// 格式化后
fn add(a: int, b: int) int {
    return a + b;
}
```

**运算符周围**: 添加空格

```az
// 格式化前
let x=10+20*30;

// 格式化后
let x = 10 + 20 * 30;
```

### 3. 结构体字段对齐

**默认**: 对齐字段

```az
// 格式化前
struct Point {
    x: int,
    y: int,
    name: string
}

// 格式化后
struct Point {
    x:    int,
    y:    int,
    name: string,
}
```

### 4. 导入语句

**格式化**: 统一格式，添加空行

```az
// 格式化前
import std.io;import std.string;

// 格式化后
import std.io;

import std.string;
```

### 5. 函数定义

**格式化**: 统一格式，添加空行

```az
// 格式化前
pub fn add(a:int,b:int)int{return a+b;}

// 格式化后
pub fn add(a: int, b: int) int {
    return a + b;
}
```

### 6. 枚举定义

**格式化**: 统一格式，每个变体一行

```az
// 格式化前
enum Result<T,E>{Ok(T),Err(E)}

// 格式化后
enum Result<T, E> {
    Ok(T),
    Err(E),
}
```

### 7. Match表达式

**格式化**: 统一格式，对齐箭头

```az
// 格式化前
match x{1=>println("one"),2=>println("two"),_=>println("other")}

// 格式化后
match x {
    case 1 => println("one"),
    case 2 => println("two"),
    case _ => println("other"),
}
```

---

## ⚙️ 配置

### 配置文件

创建 `azfmt.toml` 文件：

```toml
# 缩进设置
indent_size = 4
use_spaces = true

# 行宽设置
max_line_length = 100

# 空格设置
space_before_brace = true
space_after_comma = true
space_around_operators = true

# 换行设置
newline_before_brace = false

# 对齐设置
align_struct_fields = true
align_function_params = false

# 其他设置
trailing_comma = true
```

### 配置选项

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `indent_size` | int | 4 | 缩进大小 |
| `use_spaces` | bool | true | 使用空格而非制表符 |
| `max_line_length` | int | 100 | 最大行宽 |
| `space_before_brace` | bool | true | 大括号前添加空格 |
| `space_after_comma` | bool | true | 逗号后添加空格 |
| `space_around_operators` | bool | true | 运算符周围添加空格 |
| `newline_before_brace` | bool | false | 大括号前换行 |
| `align_struct_fields` | bool | true | 对齐结构体字段 |
| `align_function_params` | bool | false | 对齐函数参数 |
| `trailing_comma` | bool | true | 添加尾随逗号 |

---

## 📚 示例

### 示例1: 格式化Hello World

**格式化前** (`hello.az`):

```az
import std.io;fn main()int{println("Hello, AZ!");return 0;}
```

**格式化后**:

```az
import std.io;

fn main() int {
    println("Hello, AZ!");
    return 0;
}
```

**命令**:

```bash
python tools/az_fmt/azfmt.py hello.az
```

### 示例2: 格式化结构体

**格式化前** (`point.az`):

```az
struct Point{x:int,y:int,name:string}
```

**格式化后**:

```az
struct Point {
    x:    int,
    y:    int,
    name: string,
}
```

### 示例3: 格式化函数

**格式化前** (`math.az`):

```az
pub fn add(a:int,b:int)int{return a+b;}
pub fn subtract(a:int,b:int)int{return a-b;}
```

**格式化后**:

```az
pub fn add(a: int, b: int) int {
    return a + b;
}

pub fn subtract(a: int, b: int) int {
    return a - b;
}
```

### 示例4: 格式化Match表达式

**格式化前** (`match.az`):

```az
fn classify(x:int)string{match x{1=>return"one",2=>return"two",_=>return"other"}}
```

**格式化后**:

```az
fn classify(x: int) string {
    match x {
        case 1 => return "one",
        case 2 => return "two",
        case _ => return "other",
    }
}
```

---

## 🔧 集成

### VS Code集成

在 `.vscode/settings.json` 中添加：

```json
{
    "[az]": {
        "editor.formatOnSave": true,
        "editor.defaultFormatter": "az_fmt"
    }
}
```

### Git Hook集成

在 `.git/hooks/pre-commit` 中添加：

```bash
#!/bin/bash
# 格式化所有AZ文件
python tools/az_fmt/azfmt.py --check $(git diff --cached --name-only --diff-filter=ACM | grep '\.az$')
```

### CI/CD集成

在 `.github/workflows/format.yml` 中添加：

```yaml
name: Format Check

on: [push, pull_request]

jobs:
  format:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Check formatting
        run: |
          python tools/az_fmt/azfmt.py --check $(find . -name '*.az')
```

---

## 🎯 与rustfmt对比

| 特性 | rustfmt | az_fmt |
|------|---------|-------|
| 自动格式化 | ✅ | ✅ |
| 可配置 | ✅ | ✅ |
| 检查模式 | ✅ | ✅ |
| 保留注释 | ✅ | ✅ |
| 对齐字段 | ✅ | ✅ |
| IDE集成 | ✅ | 🚧 计划中 |
| 增量格式化 | ✅ | ❌ |
| 宏格式化 | ✅ | 🚧 计划中 |

---

## 📖 命令行选项

```
用法: azfmt.py [-h] [--check] [--config CONFIG] [--indent INDENT]
               [--max-width MAX_WIDTH] [--version]
               files [files ...]

AZ fmt - AZ语言代码格式化工具

位置参数:
  files                 要格式化的文件

可选参数:
  -h, --help            显示帮助信息
  --check               检查格式但不修改文件
  --config CONFIG       配置文件路径
  --indent INDENT       缩进大小（默认4）
  --max-width MAX_WIDTH 最大行宽（默认100）
  --version             显示版本信息

示例:
  python tools/az_fmt/azfmt.py file.az                    格式化单个文件
  python tools/az_fmt/azfmt.py file1.az file2.az          格式化多个文件
  python tools/az_fmt/azfmt.py --check file.az            检查格式但不修改
  python tools/az_fmt/azfmt.py --config azfmt.toml file.az 使用配置文件
```

---

## 🐛 已知限制

1. **宏格式化** - 暂不支持宏的格式化
2. **增量格式化** - 暂不支持只格式化修改的部分
3. **复杂表达式** - 对于非常复杂的表达式可能格式化不完美
4. **注释位置** - 某些情况下注释位置可能不理想

---

## 🔮 未来计划

- [ ] 增量格式化
- [ ] 宏格式化
- [ ] 更智能的换行
- [ ] 更好的注释处理
- [ ] IDE插件
- [ ] 性能优化
- [ ] 更多配置选项

---

## 🤝 贡献

欢迎贡献代码！请参考 [CONTRIBUTING.md](../../CONTRIBUTING.md)

---

## 📝 许可证

本项目采用木兰宽松许可证2.0（Mulan Permissive License，Version 2）。

---

## 📞 联系方式

- **GitHub**: https://github.com/JuSanSuiYuan/az
- **Issues**: https://github.com/JuSanSuiYuan/az/issues

---

<div align="center">

**AZ fmt - 让AZ代码更美观**

Made with ❤️ by [JuSanSuiYuan](https://github.com/JuSanSuiYuan)

</div>
