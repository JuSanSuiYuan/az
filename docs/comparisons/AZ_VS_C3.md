# AZ vs C3 - 详细对比

## 概述

AZ语言深受C3启发，但在多个方面进行了改进和扩展。

## 核心理念对比

| 方面 | C3 | AZ | 说明 |
|------|----|----|------|
| **设计目标** | 改进C语言 | 现代系统编程语言 | AZ目标更广 |
| **错误处理** | Result类型 | Result类型 | ✅ 相同 |
| **模块系统** | 简单模块 | 完整模块+包管理 | AZ更完善 |
| **编译器基础** | 自研 | LLVM/MLIR | AZ基于成熟生态 |
| **工具链** | 基础工具 | 完整工具链 | AZ更丰富 |

## 详细对比

### 1. 错误处理 ✅ 相同

#### C3
```c3
fn int! divide(int a, int b) {
    if (b == 0) return IoError.DIVISION_BY_ZERO?;
    return a / b;
}
```

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
```

**结论**: 两者都使用Result类型，AZ的语法更明确。

### 2. 模块系统 ⭐ AZ更完善

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

#### AZ
```az
module math.vector;

pub struct Vec3 {
    x: float,
    y: float,
    z: float
}

pub fn dot(a: Vec3, b: Vec3) float {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
```

```az
import math.vector;

fn main() int {
    let v = math.vector.Vec3 { x: 1.0, y: 2.0, z: 3.0 };
    return 0;
}
```

**AZ的优势**:
- ✅ 完整的包管理器（chim）
- ✅ 更好的依赖管理
- ✅ 版本控制
- ✅ 发布和分发机制

### 3. 编译器技术栈 ⭐⭐ AZ显著优势

#### C3
- 自研编译器
- 直接生成LLVM IR
- 基础优化

#### AZ
- 基于LLVM/Clang
- 使用MLIR（多级IR）
- 完整的LLVM优化Pass
- LLD链接器
- LLDB调试器

**AZ的优势**:
- ✅ MLIR提供更好的抽象和优化
- ✅ 成熟的LLVM生态系统
- ✅ 跨平台支持更好
- ✅ 持续更新和维护

### 4. 类型系统 ⚖️ 各有特色

#### C3
```c3
// 基本类型
int, float, bool, void

// 数组
int[10] arr;
int[] dynamic_arr;

// 结构体
struct Point {
    int x, y;
}

// 枚举
enum Color {
    RED,
    GREEN,
    BLUE
}

// 联合体
union Value {
    int i;
    float f;
}
```

#### AZ
```az
// 基本类型
int, float, bool, void, string

// 数组
let arr: [int; 10];
let dynamic: []int;

// 结构体
struct Point {
    x: int,
    y: int
}

// 枚举
enum Color {
    Red,
    Green,
    Blue
}

// Result类型（内置）
Result<T, E>

// Option类型（内置）
Option<T>
```

**AZ的优势**:
- ✅ 内置Result和Option类型
- ✅ 更好的泛型支持（计划中）
- ✅ 字符串作为一等类型

### 5. 语法对比

#### 变量声明

**C3**:
```c3
int x = 10;
int* ptr = &x;
```

**AZ**:
```az
let x = 10;        // 不可变
var y = 20;        // 可变
let ptr = &x;      // 指针
```

#### 函数定义

**C3**:
```c3
fn int add(int a, int b) {
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
} else {
    // ...
}

while (condition) {
    // ...
}

for (int i = 0; i < 10; i++) {
    // ...
}

switch (x) {
    case 1:
        // ...
    case 2:
        // ...
    default:
        // ...
}
```

**AZ**:
```az
if (x > 0) {
    // ...
} else {
    // ...
}

while (condition) {
    // ...
}

for (i in 0..10) {
    // ...
}

match x {
    1 => println("one"),
    2 => println("two"),
    _ => println("other")
}
```

**AZ的优势**:
- ✅ match表达式更强大
- ✅ 模式匹配
- ✅ 守卫条件

### 6. 内存管理 ⚖️ 不同策略

#### C3
- 手动内存管理
- 可选的引用计数
- 栈分配优先

#### AZ
- 手动内存管理
- 可选的AZGC垃圾回收器
- 所有权系统（计划中）
- 栈分配优先

**AZ的优势**:
- ✅ 可选的GC，更灵活
- ✅ 计划支持所有权系统
- ✅ 更好的内存安全保证

### 7. 编译时特性 ⚖️ 各有特色

#### C3
```c3
// 编译时执行
$if (DEBUG):
    io::println("Debug mode");
$endif

// 宏
macro square(x) {
    return x * x;
}
```

#### AZ
```az
// 编译时执行
comptime {
    println("Compile time");
}

// 条件编译
#[cfg(debug)]
fn debug_only() void {
    println("Debug");
}

// 宏（计划中）
macro! square(x) {
    x * x
}
```

### 8. 工具链对比 ⭐⭐ AZ显著优势

#### C3
- c3c编译器
- 基础构建工具
- 简单的测试框架

#### AZ
- az编译器
- chim包管理器（类似cargo）
- 完整的测试框架
- LSP服务器
- 代码格式化器
- 静态分析工具
- 文档生成器
- LLDB调试器集成

**AZ的优势**:
- ✅ 完整的开发工具链
- ✅ 现代化的包管理
- ✅ IDE支持更好
- ✅ 调试体验更好

### 9. 标准库对比 ⭐ AZ更完整

#### C3
- 基础I/O
- 基础数据结构
- 基础字符串操作

#### AZ
- 完整的I/O系统
- 丰富的数据结构（Vec, Map, Set等）
- 完整的字符串操作
- 文件系统
- 网络（TCP/UDP）
- 线程和并发
- 时间和日期
- 加密和哈希
- JSON/XML解析
- 正则表达式

**AZ的优势**:
- ✅ 标准库更完整
- ✅ 开箱即用的功能更多
- ✅ 更好的文档

### 10. 性能对比 ⚖️ 理论上相近

#### C3
- 接近C的性能
- 零成本抽象
- 编译时优化

#### AZ
- 接近C的性能
- 零成本抽象
- LLVM优化Pass
- LTO支持
- PGO支持

**结论**: 两者性能都接近C，AZ可能因为LLVM的优化而略有优势。

## 总体评价

### C3的优势

1. ✅ **简单** - 更接近C，学习曲线平缓
2. ✅ **轻量** - 编译器小巧
3. ✅ **成熟** - 已经有一定的用户基础
4. ✅ **稳定** - 语言设计相对稳定

### AZ的优势

1. ✅ **现代化** - 基于LLVM/MLIR，技术栈先进
2. ✅ **完整** - 工具链和标准库更完整
3. ✅ **强大** - MLIR提供更好的优化和扩展性
4. ✅ **生态** - 基于LLVM生态，资源丰富
5. ✅ **包管理** - chim提供现代化的包管理体验
6. ✅ **跨平台** - LLVM的跨平台支持更好

## 选择建议

### 选择C3，如果你：
- 想要一个简单的C替代品
- 不需要复杂的工具链
- 喜欢轻量级的编译器
- 项目规模较小

### 选择AZ，如果你：
- 需要现代化的工具链
- 想要完整的包管理系统
- 需要MLIR的强大功能
- 项目规模较大
- 需要更完整的标准库
- 想要更好的IDE支持

## 相似度评分

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AZ与C3的相似度: ████████████░░░░░░░░ 60%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ 错误处理        ████████████████████ 100%
✅ 模块系统        ████████████████░░░░  80%
⚖️ 语法           ████████████████░░░░  80%
⚖️ 类型系统        ██████████████░░░░░░  70%
❌ 编译器技术栈     ████░░░░░░░░░░░░░░░░  20%
❌ 工具链          ████████░░░░░░░░░░░░  40%
⚖️ 标准库          ████████████░░░░░░░░  60%
```

## 结论

**AZ语言借鉴了C3的优秀设计理念（特别是错误处理和模块系统），但在以下方面进行了显著改进：**

1. **编译器技术栈** - 基于LLVM/MLIR
2. **工具链** - 完整的现代化工具
3. **包管理** - chim包管理器
4. **标准库** - 更完整的功能
5. **生态系统** - 基于LLVM生态

**AZ = C3的理念 + LLVM的技术 + Rust的工具链 + 自己的创新**

---

**两者都是优秀的系统编程语言，选择取决于你的具体需求！** 🚀
