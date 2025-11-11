# AZ编程语言 - 当前状态和使用指南

**更新日期**: 2025年10月29日  
**版本**: v0.3.0-dev  
**状态**: 前端完成，MLIR框架完成

---

## 🎯 当前可以做什么

### ✅ 1. 使用Bootstrap编译器（立即可用）

**完全可用，无需任何依赖！**

```bash
# 运行示例程序
python bootstrap/az_compiler.py examples/hello.az
python bootstrap/az_compiler.py examples/fibonacci.az

# 运行所有测试
python bootstrap/az_compiler.py examples/hello.az
python bootstrap/az_compiler.py examples/variables.az
python bootstrap/az_compiler.py examples/functions.az
python bootstrap/az_compiler.py examples/control_flow.az
python bootstrap/az_compiler.py examples/fibonacci.az
```

**输出示例**:
```
AZ编译器 v0.1.0
采用C3风格的错误处理

正在编译: examples/hello.az
[1/4] 词法分析...
  生成了 11 个token
[2/4] 语法分析...
  生成了 2 个顶层语句
[3/4] 语义分析...
  语义检查通过
[4/4] 执行程序...
---输出---
Hello, AZ!
欢迎使用AZ编程语言
----------

编译成功！
```

### ✅ 2. 学习编译器原理

**完整的编译器实现，代码清晰易懂**

- 📖 阅读Bootstrap编译器源码（~1,000行Python）
- 📖 阅读C++编译器源码（~3,750行C++）
- 📖 学习词法分析、语法分析、语义分析
- 📖 理解C3风格的错误处理
- 📖 学习MLIR集成

### ✅ 3. 编写AZ程序

**可以编写和运行简单的AZ程序**

```az
// 示例：计算斐波那契数列
import std.io;

fn fibonacci(n: int) int {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

fn main() int {
    var i = 0;
    while (i < 10) {
        let fib = fibonacci(i);
        print("fib(");
        print(i);
        print(") = ");
        println(fib);
        i = i + 1;
    }
    return 0;
}
```

### ✅ 4. 测试语言特性

**验证语法和语义**

```az
// 类型推导
let x = 10;        // 推导为 int
let y = 3.14;      // 推导为 float
let sum = x + y;   // 推导为 float

// 函数
fn add(a: int, b: int) int {
    return a + b;
}

// 控制流
if (x > 5) {
    println("x大于5");
}

while (i < 10) {
    println(i);
    i = i + 1;
}
```

## ❌ 当前不能做什么

### 1. 生成可执行文件

```bash
# ❌ 这个还不能用
$ az build hello.az -o hello
$ ./hello
```

**原因**: LLVM后端尚未完全实现

**预计**: 1-2个月后可用

### 2. 使用标准库

```az
// ❌ 这些还不能用
import std.fs;          // 文件系统
import std.net;         // 网络
import std.collections; // 集合
```

**原因**: 标准库尚未实现

**预计**: 2-3个月后可用

### 3. 使用包管理器

```bash
# ❌ 这个还不能用
$ chim init my-project
$ chim add some-package
```

**原因**: chim包管理器尚未实现

**预计**: 3-4个月后可用

### 4. 自举编译

```bash
# ❌ 这个还不能用
$ az build compiler/main.az -o az_new
```

**原因**: 需要完整的代码生成和标准库

**预计**: 6-12个月后可用

## 📊 功能完成度

### 编译器组件

| 组件 | 完成度 | 可用性 | 说明 |
|------|--------|--------|------|
| **词法分析器** | 100% | ✅ 完全可用 | 支持多编码 |
| **语法分析器** | 100% | ✅ 完全可用 | 递归下降 |
| **语义分析器** | 90% | ✅ 基本可用 | 类型检查完整 |
| **MLIR生成** | 60% | 🚧 部分可用 | 框架完成 |
| **LLVM后端** | 0% | ❌ 不可用 | 待实现 |
| **链接器** | 0% | ❌ 不可用 | 待实现 |
| **调试器** | 0% | ❌ 不可用 | 待实现 |

### 语言特性

| 特性 | 完成度 | 可用性 | 说明 |
|------|--------|--------|------|
| **变量声明** | 100% | ✅ 完全可用 | let/var |
| **函数** | 100% | ✅ 完全可用 | 包括递归 |
| **基本类型** | 100% | ✅ 完全可用 | int/float/string/bool |
| **运算符** | 100% | ✅ 完全可用 | 算术/比较/逻辑 |
| **控制流** | 80% | ✅ 基本可用 | if/while |
| **类型推导** | 90% | ✅ 基本可用 | 自动推导 |
| **结构体** | 0% | ❌ 不可用 | 待实现 |
| **枚举** | 0% | ❌ 不可用 | 待实现 |
| **数组** | 0% | ❌ 不可用 | 待实现 |
| **模式匹配** | 0% | ❌ 不可用 | 待实现 |

### 工具链

| 工具 | 完成度 | 可用性 | 说明 |
|------|--------|--------|------|
| **az编译器** | 60% | 🚧 部分可用 | 前端完成 |
| **chim包管理器** | 10% | ❌ 不可用 | 仅设计 |
| **LSP服务器** | 0% | ❌ 不可用 | 待实现 |
| **代码格式化器** | 0% | ❌ 不可用 | 待实现 |
| **文档生成器** | 0% | ❌ 不可用 | 待实现 |

## 🚀 快速开始

### 方法1：使用Bootstrap编译器（推荐）

**无需任何依赖，立即可用！**

```bash
# 1. 克隆仓库
git clone https://github.com/JuSanSuiYuan/az.git
cd az

# 2. 运行示例
python bootstrap/az_compiler.py examples/hello.az

# 3. 编写自己的程序
echo 'import std.io;

fn main() int {
    println("Hello from AZ!");
    return 0;
}' > my_program.az

# 4. 运行
python bootstrap/az_compiler.py my_program.az
```

### 方法2：构建C++编译器（需要LLVM）

**需要LLVM 17+环境**

```bash
# 1. 安装依赖
# Ubuntu/Debian
sudo apt install llvm-17-dev libmlir-17-dev cmake ninja-build

# macOS
brew install llvm@17 cmake ninja

# 2. 构建
./build.sh  # Linux/macOS
# 或
build.bat   # Windows

# 3. 运行测试
ctest --test-dir build

# 4. 使用编译器
./build/tools/az examples/hello.az
```

## 📚 学习资源

### 文档导航

**新手入门**:
1. [README.md](README.md) - 项目概述
2. [QUICKSTART.md](QUICKSTART.md) - 5分钟快速入门
3. [examples/](examples/) - 示例程序

**开发者**:
1. [BUILD.md](BUILD.md) - 构建指南
2. [ARCHITECTURE.md](ARCHITECTURE.md) - 架构设计
3. [CONTRIBUTING.md](CONTRIBUTING.md) - 贡献指南

**深入理解**:
1. [README_COMPILER.md](README_COMPILER.md) - 编译器详解
2. [SEMANTIC_ANALYZER_COMPLETE.md](SEMANTIC_ANALYZER_COMPLETE.md) - 语义分析器
3. [MLIR_IMPLEMENTATION.md](MLIR_IMPLEMENTATION.md) - MLIR实现

**项目状态**:
1. [STATUS.md](STATUS.md) - 实现状态
2. [ROADMAP.md](ROADMAP.md) - 开发路线图
3. [FINAL_PROGRESS_REPORT.md](FINAL_PROGRESS_REPORT.md) - 进度报告

### 示例程序

```bash
examples/
├── hello.az          # Hello World
├── variables.az      # 变量和运算
├── functions.az      # 函数示例
├── control_flow.az   # 控制流
└── fibonacci.az      # 递归函数
```

## 🎯 使用场景

### ✅ 适合的场景

1. **学习编译器原理** ⭐⭐⭐⭐⭐
   - 完整的编译器实现
   - 代码清晰易懂
   - 详细的文档

2. **研究语言设计** ⭐⭐⭐⭐⭐
   - C3风格错误处理
   - MLIR集成
   - 现代语言特性

3. **原型开发** ⭐⭐⭐⭐☆
   - 快速验证想法
   - 测试语法设计
   - 编写示例代码

4. **教学演示** ⭐⭐⭐⭐⭐
   - 展示编译器工作原理
   - 演示类型系统
   - 讲解错误处理

### ❌ 不适合的场景

1. **生产环境** ❌
   - 无法生成可执行文件
   - 标准库不完整
   - 工具链不完善

2. **大型项目** ❌
   - 性能不足（解释执行）
   - 功能有限
   - 生态系统不成熟

3. **商业应用** ❌
   - 不稳定
   - 缺少支持
   - 功能不完整

## 🔧 常见问题

### Q1: 如何运行AZ程序？

**A**: 使用Bootstrap编译器：
```bash
python bootstrap/az_compiler.py your_program.az
```

### Q2: 为什么不能生成可执行文件？

**A**: LLVM后端尚未完全实现。预计1-2个月后可用。

### Q3: 如何贡献代码？

**A**: 查看[CONTRIBUTING.md](CONTRIBUTING.md)了解详情。

### Q4: 支持哪些平台？

**A**: 
- Bootstrap编译器：所有支持Python 3.7+的平台
- C++编译器：Linux, macOS, Windows（需要LLVM）

### Q5: 如何报告bug？

**A**: 在[GitHub Issues](https://github.com/JuSanSuiYuan/az/issues)提交。

### Q6: 项目的长期计划是什么？

**A**: 查看[ROADMAP.md](ROADMAP.md)了解详细路线图。

## 📈 发展时间线

### 已完成 ✅

- **2025年10月29日**: Bootstrap编译器完成
- **2025年10月29日**: C++前端完成
- **2025年10月29日**: 语义分析器完成
- **2025年10月29日**: MLIR生成器框架完成

### 进行中 🚧

- **2025年11月**: 完善MLIR生成
- **2025年11-12月**: LLVM后端实现

### 计划中 📋

- **2025年12月**: 第一个可执行文件
- **2026年1-2月**: 工具链完善
- **2026年3-6月**: 标准库实现
- **2026年下半年**: v1.0.0发布

## 🎊 总结

**AZ编程语言当前状态**:

✅ **可以做**:
- 运行简单程序（Bootstrap）
- 学习编译器原理
- 研究语言设计
- 编写和测试代码

❌ **不能做**:
- 生成可执行文件
- 使用完整标准库
- 开发大型项目
- 生产环境使用

**总体评价**: 
- **学习和研究**: ⭐⭐⭐⭐⭐ 优秀
- **原型开发**: ⭐⭐⭐⭐☆ 良好
- **实际应用**: ⭐⭐☆☆☆ 有限
- **生产使用**: ⭐☆☆☆☆ 不推荐

**推荐用途**: 学习、研究、原型开发

---

**GitHub**: https://github.com/JuSanSuiYuan/az  
**更新日期**: 2025年10月29日  
**版本**: v0.3.0-dev

⭐ 如果您觉得这个项目有价值，请给我们一个Star！
