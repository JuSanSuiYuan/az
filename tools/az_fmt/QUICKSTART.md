# AZ fmt 快速开始

**5分钟学会使用AZ fmt**

---

## 🚀 第一步：安装

AZ fmt是Python脚本，无需安装，直接使用：

```bash
cd tools/az_fmt
```

---

## 📝 第二步：创建测试文件

创建 `hello.az`:

```az
import std.io;fn main()int{println("Hello, AZ!");return 0;}
```

---

## ✨ 第三步：格式化

```bash
# Windows
python azfmt.py hello.az

# Linux/macOS
python3 azfmt.py hello.az
```

**结果**:

```az
import std.io;

fn main() int {
    println("Hello, AZ!");
    return 0;
}
```

---

## 🎯 常用命令

### 1. 格式化单个文件

```bash
python azfmt.py file.az
```

### 2. 格式化多个文件

```bash
python azfmt.py file1.az file2.az file3.az
```

### 3. 检查格式（不修改）

```bash
python azfmt.py --check file.az
```

### 4. 自定义缩进

```bash
# 使用2个空格缩进
python azfmt.py --indent 2 file.az

# 使用8个空格缩进
python azfmt.py --indent 8 file.az
```

### 5. 自定义行宽

```bash
# 最大行宽120
python azfmt.py --max-width 120 file.az
```

### 6. 使用配置文件

```bash
python azfmt.py --config azfmt.toml file.az
```

---

## 📋 格式化示例

### 示例1: 函数

**格式化前**:
```az
pub fn add(a:int,b:int)int{return a+b;}
```

**格式化后**:
```az
pub fn add(a: int, b: int) int {
    return a + b;
}
```

### 示例2: 结构体

**格式化前**:
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

### 示例3: 枚举

**格式化前**:
```az
enum Result<T,E>{Ok(T),Err(E)}
```

**格式化后**:
```az
enum Result<T, E> {
    Ok(T),
    Err(E),
}
```

### 示例4: If语句

**格式化前**:
```az
if(x>10){println("big");}else{println("small");}
```

**格式化后**:
```az
if (x > 10) {
    println("big");
} else {
    println("small");
}
```

---

## ⚙️ 配置文件

创建 `azfmt.toml`:

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

# 对齐设置
align_struct_fields = true
```

使用配置文件：

```bash
python azfmt.py --config azfmt.toml file.az
```

---

## 🔧 集成到项目

### 方法1: 手动格式化

```bash
# 格式化所有AZ文件
find . -name "*.az" -exec python tools/az_fmt/azfmt.py {} \;
```

### 方法2: Git Hook

在 `.git/hooks/pre-commit` 中添加：

```bash
#!/bin/bash
python tools/az_fmt/azfmt.py --check $(git diff --cached --name-only | grep '\.az$')
```

### 方法3: VS Code

在 `.vscode/settings.json` 中添加：

```json
{
    "[az]": {
        "editor.formatOnSave": true
    }
}
```

---

## 📚 更多信息

- 完整文档: [README.md](README.md)
- 配置选项: [azfmt.toml](azfmt.toml)
- 示例代码: [test_unformatted.az](test_unformatted.az)

---

## 🎉 完成！

现在你已经学会了AZ fmt的基本使用！

**下一步**:
1. 格式化你的AZ代码
2. 配置自定义规则
3. 集成到开发流程

---

<div align="center">

**AZ fmt - 让代码更美观**

⭐ [Star on GitHub](https://github.com/JuSanSuiYuan/az)

</div>
