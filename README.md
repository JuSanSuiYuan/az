# AZ编程语言

<div align="center">

**一种现代系统编程语言，结合C3和Zig的优点**

[![License](https://img.shields.io/badge/license-MulanPSL--2.0-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.5.0-green.svg)]()

[快速入门](docs/guides/QUICKSTART.md) | [当前状态](docs/status/CURRENT_STATUS.md) | [模块系统](docs/MODULE_SYSTEM.md) | [现代化特性](docs/MODERN_FEATURES.md)

**语言对比**: [AZ vs C3](docs/comparisons/AZ_VS_C3.md) | [AZ vs Zig](docs/comparisons/AZ_VS_ZIG.md)

</div>

## ✨ 核心特性

- 🎯 **C3风格错误处理** - 使用Result类型而不是异常
- 🚀 **现代语法** - 简洁清晰，易于学习
- 🔒 **类型安全** - 静态类型系统，编译时检查
- ⚡ **高性能** - 基于LLVM，编译为高效的机器码
- 🛠️ **系统编程** - 适合操作系统和底层开发
- 🌏 **中英文支持** - 支持中英文标识符和关键字
- 🔧 **LLVM技术栈** - 使用MLIR和LLVM IR
- 📦 **模块化设计** - 类似C3的模块系统，配合chim包管理器
- 🧠 **双内存管理模型** - 显式内存管理（Zig风格）与自动垃圾回收（ZGC风格）
- 🧩 **语法风格语义绑定** - 大括号语法表示显式内存管理，Python缩进语法表示GC管理
- 🎭 **Actor并发模型** - 基于Microsoft Orleans的Virtual Actor模型实现

## 🧠 双内存管理模型

AZ语言的一大创新是提供了两种内存管理方式，开发者可以根据需求选择：

### 显式内存管理（Zig风格）
使用**大括号语法**表示需要显式管理内存的代码：
```az
// 显式内存管理 - 类似Zig
pub struct ArrayList<T> {
    vec: Vec<T>
}

impl ArrayList<T> {
    /// 释放ArrayList的内存（显式内存管理）
    pub fn drop(self: *ArrayList<T>) void {
        self.vec.drop();
    }
}
```

### 自动垃圾回收（ZGC风格）
使用**Python缩进风格**表示由垃圾收集器自动管理内存的代码：
```az
// 自动垃圾回收 - 类似ZGC
pub struct ArrayList<T>:
    vec: Vec<T>

impl ArrayList<T>:
    /// 在GC管理模式下，无需手动释放内存
    # 内存由垃圾收集器自动管理
```

## 🎭 Actor并发模型

AZ语言实现了基于Microsoft Orleans的Virtual Actor模型，提供了简单而强大的并发编程模型：

```az
// 定义消息类型
struct IncrementMessage {}
struct GetCountMessage {}

impl Message for IncrementMessage {}
impl Message for GetCountMessage {}

// 定义Actor
struct CounterActor {
    id: ActorId,
    count: int
}

impl Actor for CounterActor {
    fn handle_message(self: *Self, message: *Message) Result<void, Error> {
        match message {
            case IncrementMessage => 
                self.count = self.count + 1;
            case GetCountMessage => 
                // 返回当前计数
        }
        return Result.Ok(void);
    }
    
    fn get_id(self: *Self) ActorId {
        return self.id;
    }
    
    fn on_activate(self: *Self) Result<void, Error> {
        self.count = 0;
        return Result.Ok(void);
    }
    
    fn on_deactivate(self: *Self) Result<void, Error> {
        return Result.Ok(void);
    }
}

// 创建Actor系统
let system = actor.create_system();

// 创建Actor引用
let actor_id = actor.new_actor_id("counter1");
let actor_ref = actor.actor_of<CounterActor>(&system, actor_id);

// 发送消息
let message = IncrementMessage{};
actor.send(&system, actor_ref, &message);
```

## 🌍 中英文混合编程

AZ语言的一大特色是支持中英文混合编程，开发者可以自由选择使用中文或英文标识符：

```az
// 英文命名
let userName = "张三";
fn calculateSum(a: int, b: int) int {
    return a + b;
}

// 中文命名
let 用户名称 = "张三";
fn 计算总和(甲: int, 乙: int) int {
    return 甲 + 乙;
}
```

## 🚀 快速开始

### Hello World

创建 `hello.az` 文件：

```az
import std.io;

fn main() int {
    println("Hello, AZ!");
    return 0;
}
```

运行程序：

```bash
# 编译并运行
az cl compile hello.az -o hello
./hello
```

## 📊 当前状态

### ✅ v0.1.0 - Bootstrap版本（已完成）

- [x] Bootstrap编译器（Python实现）
- [x] 词法分析器
- [x] 语法分析器
- [x] 基本语义分析
- [x] 解释执行器
- [x] C3风格的错误处理
- [x] 基本语法支持（变量、函数、控制流）
- [x] 示例程序和文档

### ✅ v0.2.5 - C++前端（已完成）

- [x] C++编译器框架
- [x] 完整的词法分析器（C++，支持多编码）
- [x] 完整的语法分析器（C++）
- [x] 完整的语义分析器（C++）
- [x] 类型系统
- [x] 符号表管理
- [x] 类型推导
- [x] AST定义
- [x] C3风格Result类型（C++）

### ✅ v0.3.0 - MLIR后端（已完成）

- [x] MLIR-AIR Dialect框架
- [x] MLIR生成器（MLIRGen）
- [x] MLIR降级到LLVM IR
- [x] LLVM后端实现
- [x] 优化器（Optimizer）
- [x] 代码生成器（CodeGenerator）
- [x] 链接器（Linker）
- [x] 调试信息生成（DebugInfo）
- [x] JIT编译器（JIT）
- [x] 编译缓存（Cache）
- [x] 结构体和枚举
- [x] 模式匹配（match语句）
- [x] for循环
- [x] 数组和切片

### ✅ v0.4.0 - 工具链完善（已完成）

- [x] 完整的编译器驱动程序
- [x] 多种输出格式支持
- [x] 丰富的优化选项
- [x] 目标平台支持
- [x] 高级链接选项
- [x] 目标管理器
- [x] 静态库和共享库支持

### ✅ v0.5.0 - 完整实现（已完成）

- [x] MLIR-AIR Dialect完整实现
- [x] LLVM IR完整生成
- [x] lld链接器完整集成
- [x] lldb调试器集成
- [x] 编译时执行（comptime）
- [x] 所有权系统
- [x] AZGC垃圾回收器
- [x] 标准库
- [x] 包管理器（az mod）

## 📦 核心模块

### 标准库模块

- `std.io` - 输入输出
- `std.string` - 字符串处理
- `std.math` - 数学运算
- `std.fs` - 文件系统操作
- `std.collections` - 集合类型
- `std.net` - 网络编程
- `std.async` - 异步编程
- `std.actor` - Actor模型

### az mod包管理器

az mod是AZ语言的官方包管理器，提供高效的依赖管理和workspace支持：

```bash
# 创建新项目
az mod init my-project

# 添加依赖
az mod add std@1.0.0

# 安装依赖
az mod install

# 构建项目
az mod build
```

## 📖 文档

- [快速入门](docs/guides/QUICKSTART.md) - 5分钟学会AZ
- [语言设计](docs/README.md) - 语言设计理念和特性
- [架构设计](docs/architecture/ARCHITECTURE.md) - 编译器架构详解
- [开发路线图](ROADMAP.md) - 未来计划
- [构建指南](BUILD.md) - 如何构建和运行az cl编译器
- [AST可视化](docs/AST_VISUALIZATION.md) - 如何可视化代码的抽象语法树

## 🤝 贡献

欢迎贡献代码、报告问题或提出建议！

1. Fork本项目: https://github.com/JuSanSuiYuan/az
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📝 许可证

本项目采用木兰宽松许可证2.0（Mulan Permissive License，Version 2）。详见 [LICENSE](LICENSE) 文件。

<div align="center">

**用AZ，写出更安全、更高效的系统代码**

Made with ❤️ by [JuSanSuiYuan](https://github.com/JuSanSuiYuan)

[⭐ Star](https://github.com/JuSanSuiYuan/az) | [🐛 Report Bug](https://github.com/JuSanSuiYuan/az/issues) | [💡 Request Feature](https://github.com/JuSanSuiYuan/az/issues)

</div>