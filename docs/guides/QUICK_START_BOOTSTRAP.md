# AZ语言快速自举指南

## 前提条件

1. **安装Python 3.7+**
   ```bash
   # Windows (使用winget)
   winget install Python.Python.3.12
   
   # 验证安装
   python --version
   ```

2. **安装Clang/LLVM**
   ```bash
   # Windows - 从LLVM官网下载或使用Visual Studio的LLVM工具
   # https://releases.llvm.org/download.html
   
   # Linux
   sudo apt install clang llvm  # Ubuntu/Debian
   sudo dnf install clang llvm  # Fedora
   
   # macOS
   brew install llvm
   
   # 验证安装
   clang --version
   ```

## 快速开始

### 步骤1: 测试Bootstrap编译器

```bash
# 测试解释执行模式
python bootstrap/az_compiler.py examples/hello.az

# 应该输出:
# AZ编译器 v0.1.0
# 采用C3风格的错误处理
# 
# 正在编译: examples/hello.az
# [1/4] 词法分析...
#   生成了 11 个token
# [2/4] 语法分析...
#   生成了 2 个顶层语句
# [3/4] 语义分析...
#   语义检查通过
# [4/4] 执行程序...
# ---输出---
# Hello, AZ!
# ----------
# 
# 编译成功！
```

### 步骤2: 测试C代码生成

```bash
# 生成C代码
python bootstrap/az_compiler.py examples/test_codegen.az --emit-c -o output.c

# 查看生成的C代码
cat output.c  # Linux/macOS
type output.c  # Windows

# 使用Clang编译C代码
clang output.c -o output

# 运行
./output  # Linux/macOS
output.exe  # Windows
```

### 步骤3: 创建最小化编译器

创建 `compiler/minimal/` 目录结构：

```
compiler/minimal/
├── main.az          # 主程序
├── lexer.az         # 词法分析器
├── parser.az        # 语法分析器
├── codegen.az       # C代码生成器
└── utils.az         # 工具函数
```

#### main.az

```az
// AZ最小化编译器主程序

import std.io;

fn main() int {
    println("AZ Minimal Compiler v0.1");
    
    // TODO: 实现编译器逻辑
    
    return 0;
}
```

### 步骤4: 第一次自举

```bash
# 使用Python Bootstrap编译AZ编译器
python bootstrap/az_compiler.py compiler/minimal/main.az --emit-c -o gen/az1.c

# 使用Clang编译生成的C代码
clang gen/az1.c -o gen/az1

# 测试第一代编译器
./gen/az1 examples/hello.az
```

### 步骤5: 第二次自举

```bash
# 使用第一代编译器编译自己
./gen/az1 compiler/minimal/main.az --emit-c -o gen/az2.c

# 使用Clang编译
clang gen/az2.c -o gen/az2

# 测试第二代编译器
./gen/az2 examples/hello.az
```

### 步骤6: 验证自举

```bash
# 使用第二代编译器编译自己
./gen/az2 compiler/minimal/main.az --emit-c -o gen/az3.c

# 比较生成的代码
diff gen/az2.c gen/az3.c

# 如果没有差异，自举成功！
```

## 当前实现状态

### ✅ 已完成

1. **Bootstrap编译器** (Python)
   - ✅ 词法分析器
   - ✅ 语法分析器
   - ✅ 解释执行器
   - ✅ C代码生成器（新增）

2. **C代码生成功能**
   - ✅ 函数定义
   - ✅ 变量声明
   - ✅ 表达式（算术、逻辑、比较）
   - ✅ 控制流（if, while）
   - ✅ 函数调用
   - ✅ 内置函数（println, print）

### 📋 待实现

1. **最小化编译器**
   - 📋 词法分析器（AZ实现）
   - 📋 语法分析器（AZ实现）
   - 📋 C代码生成器（AZ实现）

2. **标准库**
   - 📋 文件I/O
   - 📋 字符串操作
   - 📋 内存管理

## 测试用例

### 测试1: 简单函数

```az
fn add(a: int, b: int) int {
    return a + b;
}

fn main() int {
    let result = add(3, 5);
    return 0;
}
```

生成的C代码：

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// 内置函数
void println(const char* str) {
    printf("%s\n", str);
}

void print(const char* str) {
    printf("%s", str);
}

int add(int a, int b);
int main(void);

int add(int a, int b) {
    return (a + b);
}

int main(void) {
    int result = add(3, 5);
    return 0;
}
```

### 测试2: 递归函数

```az
fn factorial(n: int) int {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}

fn main() int {
    let fact = factorial(5);
    return 0;
}
```

### 测试3: 循环

```az
fn sum_to_n(n: int) int {
    var sum = 0;
    var i = 1;
    while (i <= n) {
        sum = sum + i;
        i = i + 1;
    }
    return sum;
}

fn main() int {
    let result = sum_to_n(10);
    return 0;
}
```

## 调试技巧

### 1. 查看生成的C代码

```bash
python bootstrap/az_compiler.py your_file.az --emit-c -o output.c
cat output.c
```

### 2. 编译时显示详细信息

```bash
gcc -v output.c -o output
```

### 3. 使用GDB调试

```bash
gcc -g output.c -o output
gdb output
```

### 4. 检查语法错误

```bash
gcc -fsyntax-only output.c
```

## 常见问题

### Q1: Python命令不可用

**A**: 确保Python已正确安装并添加到PATH。重启终端或计算机。

```bash
# Windows - 添加到PATH
setx PATH "%PATH%;C:\Users\YourName\AppData\Local\Programs\Python\Python312"

# 或使用py命令
py bootstrap/az_compiler.py examples/hello.az
```

### Q2: 生成的C代码编译失败

**A**: 检查生成的C代码，可能是类型不匹配或语法错误。

```bash
# 使用Clang查看详细错误信息
clang -Wall -Wextra output.c -o output
```

### Q3: 字符串处理问题

**A**: 当前实现使用`const char*`，可能需要手动管理内存。

```c
// 字符串连接需要手动实现
char* concat(const char* a, const char* b) {
    char* result = malloc(strlen(a) + strlen(b) + 1);
    strcpy(result, a);
    strcat(result, b);
    return result;
}
```

### Q4: 内置函数不工作

**A**: 确保生成的C代码包含内置函数定义。

## 性能优化

### 1. 编译优化

```bash
# 使用Clang优化选项
clang -O2 output.c -o output
clang -O3 output.c -o output  # 更激进的优化
```

### 2. 链接时优化（LTO）

```bash
clang -flto output.c -o output
```

### 3. 生成LLVM IR查看

```bash
# 生成LLVM IR
clang -S -emit-llvm output.c -o output.ll
cat output.ll

# 生成汇编代码
clang -S output.c -o output.s
cat output.s
```

## 下一步计划

### 短期（1-2周）

1. ✅ 完成C代码生成器
2. 📋 创建最小化编译器
3. 📋 实现第一次自举
4. 📋 验证和测试

### 中期（1个月）

1. 📋 添加更多语言特性
   - 结构体
   - 数组
   - Match语句
   - 字符串操作

2. 📋 优化生成的代码
   - 常量折叠
   - 死代码消除
   - 简单的寄存器分配

### 长期（3-6个月）

1. 📋 实现LLVM后端
2. 📋 完整的标准库
3. 📋 包管理器
4. 📋 LSP服务器

## 贡献指南

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md)

### 如何贡献

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

### 代码风格

- Python代码遵循PEP 8
- AZ代码使用4空格缩进
- C代码使用K&R风格

## 资源链接

- **项目主页**: https://github.com/JuSanSuiYuan/az
- **文档**: [docs/](docs/)
- **示例**: [examples/](examples/)
- **测试**: [test/](test/)

## 联系方式

- **Issues**: https://github.com/JuSanSuiYuan/az/issues
- **Discussions**: https://github.com/JuSanSuiYuan/az/discussions

---

**开始你的AZ语言自举之旅吧！** 🚀

记住：
1. 先测试Bootstrap编译器
2. 验证C代码生成
3. 创建最小化编译器
4. 实现自举
5. 不断迭代改进

**祝你成功！** 🎉
