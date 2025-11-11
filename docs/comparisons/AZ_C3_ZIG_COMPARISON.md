# AZ vs C3 vs Zig - 三语言全面对比

## 概述

AZ、C3和Zig都是现代系统编程语言，都旨在改进或替代C语言，但采用了不同的设计理念和技术路线。

## 一句话总结

- **C3**: 更好的C，保持简单
- **Zig**: 简单而强大，独立自主
- **AZ**: 现代化，基于LLVM/MLIR

## 核心理念对比

| 方面 | C3 | Zig | AZ |
|------|----|----|-----|
| **设计目标** | 改进C | 更好的C | 现代系统语言 |
| **错误处理** | Result类型 | try/catch风格 | Result类型 |
| **内存管理** | 手动 | 手动+Allocator | 手动+可选GC |
| **编译器** | 自研 | 自研 | LLVM/MLIR |
| **编译时执行** | 有限支持 | 强大的comptime | comptime |
| **模块系统** | 简单模块 | @import | module/import |
| **包管理** | 无 | 内置 | chim |
| **工具链** | 基础 | 完整 | 完整 |

## 详细对比

### 1. 错误处理 🎯

#### C3
```c3
fn int! divide(int a, int b) {
    if (b == 0) return IoError.DIVISION_BY_ZERO?;
    return a / b;
}

fn void main() {
    int result = divide(10, 2)!;
    // 或
    int! result = divide(10, 2);
    if (catch err = result) {
        // 处理错误
    }
}
```

**特点**:
- `!` 表示可能出错
- `?` 传播错误
- `catch` 捕获错误
- 简洁明了

#### Zig
```zig
fn divide(a: i32, b: i32) !i32 {
    if (b == 0) return error.DivisionByZero;
    return @divTrunc(a, b);
}

pub fn main() !void {
    const result = try divide(10, 2);
    // 或
    const result = divide(10, 2) catch |err| {
        // 处理错误
        return err;
    };
}
```

**特点**:
- `!` 表示错误联合类型
- `try` 传播错误
- `catch` 捕获错误
- 错误是值

#### AZ
```az
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

**特点**:
- `Result<T>` 类型
- 显式检查
- 类似Rust
- 借鉴C3

**对比总结**:
| 语言 | 风格 | 简洁度 | 明确度 |
|------|------|--------|--------|
| C3 | Result | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Zig | 错误联合 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| AZ | Result | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### 2. 内存管理 💾

#### C3
```c3
fn void example() {
    int* ptr = malloc(sizeof(int) * 10);
    defer free(ptr);
    
    // 使用ptr
}
```

**特点**:
- 手动管理
- defer自动清理
- 简单直接

#### Zig
```zig
pub fn example() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const buffer = try allocator.alloc(u8, 10);
    defer allocator.free(buffer);
    
    // 使用buffer
}
```

**特点**:
- Allocator模式
- 显式传递
- defer自动清理
- 无隐藏分配

#### AZ
```az
fn example() int {
    // 手动管理
    let ptr = malloc(100);
    defer free(ptr);
    
    // 或使用GC
    #[gc]
    let obj = new Object();
    
    return 0;
}
```

**特点**:
- 手动管理
- 可选GC
- defer清理
- 灵活选择

**对比总结**:
| 语言 | 方式 | 灵活度 | 安全性 |
|------|------|--------|--------|
| C3 | 手动 | ⭐⭐⭐ | ⭐⭐⭐ |
| Zig | Allocator | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| AZ | 手动+GC | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### 3. 编译时执行 ⚡

#### C3
```c3
$if (DEBUG):
    io::println("Debug mode");
$endif

macro square(x) {
    return x * x;
}
```

**特点**:
- 有限的编译时支持
- 宏系统
- 条件编译

#### Zig
```zig
fn fibonacci(n: u32) u32 {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

pub fn main() void {
    const fib10 = comptime fibonacci(10);
    std.debug.print("Fib(10) = {}\n", .{fib10});
}
```

**特点**:
- 强大的comptime
- 任何地方都可用
- 编译时多态

#### AZ
```az
fn fibonacci(n: int) int {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

fn main() int {
    comptime {
        let fib10 = fibonacci(10);
        println("Fib(10) = " + fib10);
    }
    return 0;
}
```

**特点**:
- comptime块
- 编译时执行
- 类似Zig

**对比总结**:
| 语言 | 能力 | 易用性 | 强大度 |
|------|------|--------|--------|
| C3 | 有限 | ⭐⭐⭐⭐ | ⭐⭐ |
| Zig | 强大 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| AZ | 中等 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### 4. 模块系统 📦

#### C3
```c3
module math::vector;

public struct Vec3 {
    float x, y, z;
}

public fn float dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
```

```c3
import math::vector;

fn void main() {
    Vec3 v = { 1.0, 2.0, 3.0 };
}
```

#### Zig
```zig
// math.zig
pub fn add(a: i32, b: i32) i32 {
    return a + b;
}

// main.zig
const math = @import("math.zig");

pub fn main() void {
    const result = math.add(1, 2);
}
```

#### AZ
```az
// math.az
module math;

pub fn add(a: int, b: int) int {
    return a + b;
}

// main.az
import math;

fn main() int {
    let result = math.add(1, 2);
    return 0;
}
```

**对比总结**:
| 语言 | 风格 | 简洁度 | 功能 |
|------|------|--------|------|
| C3 | module | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Zig | @import | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| AZ | module | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### 5. 包管理 📦

#### C3
- ❌ 无内置包管理
- 需要手动管理依赖

#### Zig
```zig
// build.zig.zon
.{
    .name = "myproject",
    .version = "0.1.0",
    .dependencies = .{
        .json = .{
            .url = "https://...",
            .hash = "...",
        },
    },
}
```

```bash
zig build
zig build run
```

#### AZ
```az
// package.az
package {
    name: "myproject",
    version: "0.1.0",
    dependencies: {
        "json": "1.0.0"
    }
}
```

```bash
az_mod build
az_mod run
```

**对比总结**:
| 语言 | 包管理 | 易用性 | 功能 |
|------|--------|--------|------|
| C3 | ❌ 无 | - | - |
| Zig | ✅ 内置 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| AZ | ✅ az_mod | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### 6. 编译器技术 🔧

#### C3
- **自研编译器**
- 直接生成LLVM IR
- 简单快速

**优点**:
- 编译快
- 独立
- 轻量

**缺点**:
- 优化有限
- 功能较少

#### Zig
- **自研编译器**
- 直接生成机器码
- 增量编译
- 缓存系统

**优点**:
- 编译非常快
- 完全控制
- 无外部依赖

**缺点**:
- 优化能力有限
- 维护成本高

#### AZ
- **基于LLVM/MLIR**
- MLIR多级IR
- LLVM优化
- LLD链接

**优点**:
- 强大优化
- 成熟生态
- 持续更新

**缺点**:
- 编译较慢
- 依赖LLVM
- 体积大

**对比总结**:
| 语言 | 编译器 | 编译速度 | 优化能力 | 独立性 |
|------|--------|----------|----------|--------|
| C3 | 自研 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Zig | 自研 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| AZ | LLVM | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |

### 7. 标准库 📚

#### C3
- 基础I/O
- 基础数据结构
- 简单实用

#### Zig
- 完整的std库
- 需要allocator
- 功能丰富

#### AZ
- 完整的std库
- 默认allocator
- 功能丰富

**对比总结**:
| 语言 | 完整度 | 易用性 | 文档 |
|------|--------|--------|------|
| C3 | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Zig | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| AZ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### 8. 跨平台支持 🌍

#### C3
- 基于LLVM
- 支持主流平台
- 依赖工具链

#### Zig
- 内置交叉编译
- 可作为C编译器
- 自带libc
- 支持所有平台

#### AZ
- 基于LLVM
- 支持主流平台
- 依赖工具链

**对比总结**:
| 语言 | 跨平台 | 交叉编译 | 易用性 |
|------|--------|----------|--------|
| C3 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Zig | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| AZ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

### 9. 语法对比

#### 变量声明

**C3**:
```c3
int x = 10;
int* ptr = &x;
```

**Zig**:
```zig
const x: i32 = 10;
var y: i32 = 20;
```

**AZ**:
```az
let x: int = 10;
var y: int = 20;
```

#### 函数定义

**C3**:
```c3
fn int add(int a, int b) {
    return a + b;
}
```

**Zig**:
```zig
fn add(a: i32, b: i32) i32 {
    return a + b;
}
```

**AZ**:
```az
fn add(a: int, b: int) int {
    return a + b;
}
```

#### 控制流

**C3**:
```c3
if (x > 0) {
    // ...
}

switch (x) {
    case 1: break;
    default: break;
}
```

**Zig**:
```zig
if (x > 0) {
    // ...
}

switch (x) {
    0 => {},
    1 => {},
    else => {},
}
```

**AZ**:
```az
if (x > 0) {
    // ...
}

match x {
    0 => {},
    1 => {},
    _ => {}
}
```

### 10. 特殊特性

#### C3独有
- 简单直接的设计
- 接近C的思维
- 轻量级

#### Zig独有
- 作为C编译器
- 强大的comptime
- 向量类型
- anytype

#### AZ独有
- MLIR支持
- 可选GC
- Result类型
- chim包管理器

## 综合对比表

| 特性 | C3 | Zig | AZ |
|------|----|----|-----|
| **学习曲线** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **编译速度** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **运行性能** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **优化能力** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **工具链** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **包管理** | ❌ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **标准库** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **跨平台** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **社区** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| **成熟度** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ |
| **文档** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

## 相似度矩阵

```
        C3    Zig   AZ
C3      100%  30%   60%
Zig     30%   100%  40%
AZ      60%   40%   100%
```

### C3 vs Zig: 30%
- 都是系统语言
- 都追求简单
- 但技术路线完全不同

### C3 vs AZ: 60%
- 错误处理相似
- 模块系统相似
- AZ借鉴了C3

### Zig vs AZ: 40%
- 都支持comptime
- 都有包管理
- 但错误处理不同

## 使用场景推荐

### C3适合

1. ✅ 小型项目
2. ✅ 快速原型
3. ✅ C代码迁移
4. ✅ 学习系统编程
5. ✅ 简单直接的需求

### Zig适合

1. ✅ 系统编程
2. ✅ 嵌入式开发
3. ✅ 游戏开发
4. ✅ 需要快速编译
5. ✅ 跨平台工具
6. ✅ 替代C/C++

### AZ适合

1. ✅ 大型项目
2. ✅ 编译器开发
3. ✅ 需要MLIR
4. ✅ 需要强大优化
5. ✅ 操作系统开发
6. ✅ 需要完整工具链

## 学习路径建议

### 如果你是C程序员
1. **先学C3** - 最接近C
2. **再学Zig** - 了解现代特性
3. **最后学AZ** - 掌握MLIR

### 如果你是Rust程序员
1. **先学AZ** - 错误处理相似
2. **再学Zig** - 了解不同理念
3. **最后学C3** - 理解简单设计

### 如果你是新手
1. **先学C3** - 最简单
2. **再学Zig** - 功能丰富
3. **最后学AZ** - 最强大

## 性能对比

### 编译时间（相对）
```
C3:  ████████████████░░░░ 80%
Zig: ████████████████████ 100%
AZ:  ████░░░░░░░░░░░░░░░░ 20%
```

### 运行时性能（相对）
```
C3:  ████████████████████ 100%
Zig: ████████████████████ 100%
AZ:  ████████████████████ 100%
```

### 优化能力（相对）
```
C3:  ████████████░░░░░░░░ 60%
Zig: ████████████░░░░░░░░ 60%
AZ:  ████████████████████ 100%
```

## 生态系统对比

### 包数量（估计）
```
C3:  ██░░░░░░░░░░░░░░░░░░ ~100
Zig: ████████████████░░░░ ~1000
AZ:  █░░░░░░░░░░░░░░░░░░░ ~10
```

### 社区活跃度
```
C3:  ████░░░░░░░░░░░░░░░░ 低
Zig: ████████████████████ 高
AZ:  ██░░░░░░░░░░░░░░░░░░ 很低
```

### 文档完整度
```
C3:  ████████████░░░░░░░░ 中等
Zig: ████████████████░░░░ 良好
AZ:  ████████████░░░░░░░░ 中等
```

## 总结

### 一句话总结

- **C3**: 简单实用的C改进版
- **Zig**: 独立强大的系统语言
- **AZ**: 基于LLVM的现代系统语言

### 核心差异

```
设计理念:
C3  → 简单改进C
Zig → 重新思考C
AZ  → 现代系统语言

技术路线:
C3  → 自研 + LLVM IR
Zig → 完全自研
AZ  → LLVM + MLIR

目标用户:
C3  → C程序员
Zig → 所有系统程序员
AZ  → 需要强大工具的开发者
```

### 选择建议

**选择C3**，如果你：
- 想要最简单的C替代
- 项目规模小
- 不需要复杂特性

**选择Zig**，如果你：
- 需要快速编译
- 想要强大的comptime
- 需要跨平台编译
- 社区支持重要

**选择AZ**，如果你：
- 需要MLIR支持
- 需要最强优化
- 项目规模大
- 需要完整工具链

### 未来展望

**C3**: 保持简单，稳步发展  
**Zig**: 快速迭代，走向1.0  
**AZ**: 完善功能，实现自举

## 最终评分

### 综合评分（满分10分）

| 语言 | 简单性 | 功能性 | 性能 | 工具链 | 生态 | 总分 |
|------|--------|--------|------|--------|------|------|
| C3 | 9 | 6 | 9 | 5 | 4 | 6.6 |
| Zig | 7 | 9 | 9 | 8 | 8 | 8.2 |
| AZ | 7 | 8 | 9 | 9 | 3 | 7.2 |

### 推荐指数

- **C3**: ⭐⭐⭐⭐ (适合简单项目)
- **Zig**: ⭐⭐⭐⭐⭐ (适合大多数项目)
- **AZ**: ⭐⭐⭐⭐ (适合特定需求)

---

**三种语言都很优秀，选择取决于你的具体需求！**

**C3 = 简单** 🎯  
**Zig = 强大** ⚡  
**AZ = 现代** 🚀
