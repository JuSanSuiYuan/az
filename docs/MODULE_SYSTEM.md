# AZ语言模块系统设计

## 概述

**是的，AZ语言采用类似C3的模块化设计！**

AZ语言的模块系统借鉴了C3的优秀设计，同时结合现代语言的最佳实践，提供简洁、高效、类型安全的模块化方案。

## 设计理念

### 核心原则

1. **简单明了** - 模块系统应该易于理解和使用
2. **编译时解析** - 所有模块依赖在编译时确定
3. **无循环依赖** - 禁止模块间的循环依赖
4. **显式导入** - 必须显式导入需要的符号
5. **命名空间隔离** - 避免命名冲突

### 与C3的对比

| 特性 | C3 | AZ | 说明 |
|------|----|----|------|
| **模块声明** | `module foo;` | `module foo;` | ✅ 相同 |
| **导入方式** | `import foo;` | `import foo;` | ✅ 相同 |
| **选择性导入** | `import foo::bar;` | `import foo.bar;` | ⚠️ 语法略有不同 |
| **别名** | `import foo as f;` | `import foo as f;` | ✅ 相同 |
| **可见性** | `public/private` | `pub/priv` | ⚠️ 关键字简化 |
| **子模块** | 支持 | 支持 | ✅ 相同 |
| **包管理** | 无内置 | Az mod包管理器 | ✅ AZ更完善 |

## 模块系统语法

### 1. 模块声明

每个AZ文件都是一个模块，可以显式声明模块名：

```az
// 文件: src/math/vector.az
module math.vector;

pub struct Vec3 {
    x: float,
    y: float,
    z: float
}

pub fn dot(a: Vec3, b: Vec3) float {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// 私有函数，模块外不可见
fn internal_helper() void {
    // ...
}
```

### 2. 导入模块

#### 基本导入

```az
// 导入整个模块
import math.vector;

fn main() int {
    let v = math.vector.Vec3 { x: 1.0, y: 2.0, z: 3.0 };
    return 0;
}
```

#### 选择性导入

```az
// 只导入特定符号
import math.vector.Vec3;
import math.vector.dot;

fn main() int {
    let v1 = Vec3 { x: 1.0, y: 2.0, z: 3.0 };
    let v2 = Vec3 { x: 4.0, y: 5.0, z: 6.0 };
    let result = dot(v1, v2);
    return 0;
}
```

#### 使用别名

```az
// 导入并使用别名
import math.vector as vec;

fn main() int {
    let v = vec.Vec3 { x: 1.0, y: 2.0, z: 3.0 };
    return 0;
}
```

#### 通配符导入（谨慎使用）

```az
// 导入模块中的所有公开符号
import math.vector.*;

fn main() int {
    let v = Vec3 { x: 1.0, y: 2.0, z: 3.0 };
    let result = dot(v, v);
    return 0;
}
```

### 3. 可见性控制

```az
module mylib;

// 公开函数 - 可以被其他模块使用
pub fn public_function() void {
    println("This is public");
}

// 私有函数 - 只能在本模块内使用
fn private_function() void {
    println("This is private");
}

// 公开结构体
pub struct PublicStruct {
    pub field1: int,      // 公开字段
    field2: int           // 私有字段（默认）
}

// 私有结构体
struct PrivateStruct {
    data: int
}
```

### 4. 子模块

```az
// 文件: src/graphics/mod.az
module graphics;

pub import graphics.renderer;
pub import graphics.shader;
pub import graphics.texture;

// 重新导出子模块的符号
pub use graphics.renderer.Renderer;
pub use graphics.shader.Shader;
```

```az
// 文件: src/graphics/renderer.az
module graphics.renderer;

pub struct Renderer {
    // ...
}

pub fn create_renderer() Renderer {
    // ...
}
```

## 模块组织

### 目录结构

```
myproject/
├── src/
│   ├── main.az              # 主程序
│   ├── lib.az               # 库入口
│   ├── math/
│   │   ├── mod.az           # 模块入口
│   │   ├── vector.az        # math.vector
│   │   ├── matrix.az        # math.matrix
│   │   └── quaternion.az    # math.quaternion
│   ├── graphics/
│   │   ├── mod.az
│   │   ├── renderer.az
│   │   └── shader.az
│   └── utils/
│       ├── mod.az
│       ├── string.az
│       └── file.az
├── tests/
│   └── test_math.az
├── package.az               # 包配置
└── README.md
```

### 模块路径解析

```az
// 绝对路径导入（从项目根目录）
import myproject.math.vector;

// 相对路径导入（从当前模块）
import .sibling_module;      // 同级模块
import ..parent_module;      // 父级模块
import ...grandparent;       // 祖父级模块
```

## 标准库模块

### 核心模块

```az
// 标准I/O
import std.io;
println("Hello");

// 文件系统
import std.fs;
let content = std.fs.read_file("data.txt");

// 字符串操作
import std.string;
let s = std.string.concat("Hello", " World");

// 集合
import std.collections;
let vec = std.collections.Vec<int>.new();

// 数学
import std.math;
let result = std.math.sqrt(16.0);

// 时间
import std.time;
let now = std.time.now();

// 网络
import std.net;
let socket = std.net.TcpSocket.connect("127.0.0.1:8080");

// 线程
import std.thread;
let handle = std.thread.spawn(|| {
    println("In thread");
});
```

### 标准库结构

```
std/
├── core/           # 核心功能（自动导入）
│   ├── types.az    # 基本类型
│   ├── result.az   # Result类型
│   └── option.az   # Option类型
├── io/             # 输入输出
│   ├── mod.az
│   ├── print.az
│   └── read.az
├── fs/             # 文件系统
│   ├── mod.az
│   ├── file.az
│   └── path.az
├── collections/    # 集合
│   ├── mod.az
│   ├── vec.az
│   ├── map.az
│   └── set.az
├── string/         # 字符串
│   ├── mod.az
│   └── ops.az
├── math/           # 数学
│   ├── mod.az
│   ├── basic.az
│   └── trig.az
├── time/           # 时间
│   ├── mod.az
│   └── duration.az
├── net/            # 网络
│   ├── mod.az
│   ├── tcp.az
│   └── udp.az
└── thread/         # 线程
    ├── mod.az
    └── spawn.az
```

## 包管理 - chim

### package.az 配置

```az
package {
    name: "myproject",
    version: "0.1.0",
    authors: ["Your Name <you@example.com>"],
    license: "MIT",
    
    dependencies: {
        "json": "1.0.0",
        "http": "2.3.1",
        "crypto": { 
            version: "1.5.0",
            features: ["sha256", "aes"]
        }
    },
    
    dev_dependencies: {
        "test_framework": "0.5.0"
    },
    
    build_dependencies: {
        "codegen": "1.0.0"
    }
}
```

### 使用外部包

```az
// 导入外部包
import json;
import http;
import crypto;

fn main() int {
    let data = json.parse("{\"name\": \"AZ\"}");
    let client = http.Client.new();
    let hash = crypto.sha256("data");
    return 0;
}
```

### az_mod命令

```bash
# 创建新项目
az_mod new myproject

# 添加依赖
az_mod add json@1.0.0

# 构建项目
az_mod build

# 运行项目
az_mod run

# 测试
az_mod test

# 发布包
az_mod publish
```

## 模块编译

### 编译单元

每个模块是一个独立的编译单元：

```
源文件 (.az) → 编译 → 目标文件 (.o) → 链接 → 可执行文件
```

### 增量编译

```bash
# 只重新编译修改过的模块
az build --incremental

# 并行编译多个模块
az build -j8
```

### 预编译模块

```bash
# 预编译标准库
az precompile std

# 使用预编译模块加速编译
az build --use-precompiled
```

## 模块特性

### 1. 条件编译

```az
module mylib;

#[cfg(target_os = "windows")]
pub fn platform_specific() void {
    println("Windows");
}

#[cfg(target_os = "linux")]
pub fn platform_specific() void {
    println("Linux");
}

#[cfg(feature = "advanced")]
pub fn advanced_feature() void {
    println("Advanced feature enabled");
}
```

### 2. 模块属性

```az
#[deprecated("Use new_function instead")]
pub fn old_function() void {
    // ...
}

#[inline]
pub fn fast_function() int {
    return 42;
}

#[no_mangle]
pub fn c_compatible_function() void {
    // 可以从C代码调用
}
```

### 3. 模块文档

```az
/// 数学向量模块
/// 
/// 提供2D和3D向量的基本操作
module math.vector;

/// 3D向量结构
/// 
/// # 示例
/// ```az
/// let v = Vec3 { x: 1.0, y: 2.0, z: 3.0 };
/// ```
pub struct Vec3 {
    /// X坐标
    x: float,
    /// Y坐标
    y: float,
    /// Z坐标
    z: float
}

/// 计算两个向量的点积
/// 
/// # 参数
/// - `a`: 第一个向量
/// - `b`: 第二个向量
/// 
/// # 返回值
/// 点积结果
pub fn dot(a: Vec3, b: Vec3) float {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
```

## 与C3的详细对比

### 相似之处

1. **模块即文件** - 每个文件是一个模块
2. **显式导入** - 必须显式导入依赖
3. **编译时解析** - 所有依赖在编译时确定
4. **无头文件** - 不需要.h文件
5. **可见性控制** - public/private机制

### 差异之处

| 特性 | C3 | AZ | 原因 |
|------|----|----|------|
| **包管理** | 无 | az_mod | AZ提供完整的包管理器 |
| **模块路径** | `::` | `.` | AZ使用更常见的点号 |
| **可见性关键字** | `public` | `pub` | AZ更简洁 |
| **标准库** | 较小 | 完整 | AZ提供更完整的标准库 |
| **预编译** | 支持 | 支持 | 两者都支持 |
| **增量编译** | 支持 | 支持 | 两者都支持 |

### AZ的改进

1. **完整的包管理器** - az_mod提供类似cargo的体验
2. **更好的工具链** - 集成的构建、测试、文档工具
3. **模块文档** - 内置文档生成系统
4. **条件编译** - 更灵活的特性控制
5. **MLIR集成** - 更好的优化和代码生成

## 最佳实践

### 1. 模块组织

```az
// ✅ 好的做法
module mylib.feature;

pub struct Feature {
    // 公开接口
}

fn internal_helper() void {
    // 私有实现
}

// ❌ 不好的做法
module mylib;  // 模块名太宽泛

pub fn do_everything() void {
    // 功能太多，应该拆分
}
```

### 2. 导入管理

```az
// ✅ 好的做法
import std.io;
import std.fs;
import mylib.feature;

// ❌ 不好的做法
import std.*;  // 避免通配符导入
```

### 3. 可见性控制

```az
// ✅ 好的做法
pub struct PublicAPI {
    pub field: int,  // 明确标记公开字段
    internal: int    // 私有字段
}

// ❌ 不好的做法
pub struct BadAPI {
    pub everything: int,  // 暴露太多内部细节
    pub internal_state: int
}
```

### 4. 模块依赖

```az
// ✅ 好的做法 - 清晰的依赖关系
module app;
import lib.feature1;
import lib.feature2;

// ❌ 不好的做法 - 循环依赖
module a;
import b;  // a依赖b

module b;
import a;  // b依赖a - 禁止！
```

## 编译器实现

### 模块解析流程

```
1. 扫描项目目录
   ↓
2. 解析package.az
   ↓
3. 构建模块依赖图
   ↓
4. 检测循环依赖
   ↓
5. 拓扑排序模块
   ↓
6. 按顺序编译模块
   ↓
7. 链接生成可执行文件
```

### 模块缓存

```bash
# 模块缓存目录
.az/
├── cache/
│   ├── std.io.o
│   ├── std.fs.o
│   └── mylib.feature.o
└── deps/
    ├── json-1.0.0/
    └── http-2.3.1/
```

## 示例项目

### 完整示例

```az
// src/main.az
module main;

import std.io;
import mylib.math.vector;
import mylib.graphics.renderer;

fn main() int {
    println("AZ Module System Demo");
    
    let v = vector.Vec3 { x: 1.0, y: 2.0, z: 3.0 };
    let renderer = renderer.create();
    
    renderer.draw(v);
    
    return 0;
}
```

```az
// src/mylib/math/vector.az
module mylib.math.vector;

pub struct Vec3 {
    pub x: float,
    pub y: float,
    pub z: float
}

pub fn length(v: Vec3) float {
    import std.math;
    return std.math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}
```

```az
// src/mylib/graphics/renderer.az
module mylib.graphics.renderer;

import mylib.math.vector.Vec3;

pub struct Renderer {
    // ...
}

pub fn create() Renderer {
    return Renderer { };
}

pub fn draw(self: *Renderer, v: Vec3) void {
    // 渲染向量
}
```

## 总结

### AZ的模块系统

✅ **是的，AZ采用类似C3的模块化设计！**

主要特点：
1. **简单明了** - 易于理解和使用
2. **编译时解析** - 快速、安全
3. **完整的包管理** - chim包管理器
4. **LLVM/MLIR集成** - 优秀的优化
5. **现代化工具链** - 完整的开发体验

### 与C3的关系

- **借鉴** C3的优秀设计理念
- **改进** 包管理和工具链
- **扩展** 更完整的标准库
- **集成** MLIR和LLVM

### 优势

1. **无头文件** - 简化开发
2. **快速编译** - 增量编译和并行编译
3. **类型安全** - 编译时检查
4. **易于维护** - 清晰的模块结构
5. **完整工具链** - 从开发到发布

---

**AZ语言 - 现代化的模块系统，借鉴C3，超越C3！** 🚀
