# AZ编译器架构设计

## 概述

AZ编译器采用现代编译器架构，基于**LLVM**和**MLIR-AIR**构建，使用**lld**作为链接器，**lldb**作为调试器，**chim**作为包管理器。

## 编译器架构图

```
源代码 (.az)
    ↓
┌─────────────────────────────────────────┐
│  前端 (Frontend)                         │
├─────────────────────────────────────────┤
│  1. 词法分析 (Lexer)                     │
│     - UTF-8/多编码支持                   │
│     - Token生成                          │
│  2. 语法分析 (Parser)                    │
│     - 递归下降解析                       │
│     - AST生成                            │
│  3. 语义分析 (Semantic Analysis)         │
│     - 类型检查                           │
│     - 符号表管理                         │
│     - 作用域分析                         │
└─────────────────────────────────────────┘
    ↓ AST
┌─────────────────────────────────────────┐
│  中间表示生成 (IR Generation)            │
├─────────────────────────────────────────┤
│  4. MLIR-AIR生成                         │
│     - 高级IR (AIR Dialect)               │
│     - 编译时执行 (comptime)              │
│     - 类型推导                           │
│  5. MLIR优化                             │
│     - Dialect转换                        │
│     - 高级优化Pass                       │
│  6. LLVM IR生成                          │
│     - 从MLIR降级到LLVM IR                │
│     - 标准化表示                         │
└─────────────────────────────────────────┘
    ↓ LLVM IR
┌─────────────────────────────────────────┐
│  后端 (Backend)                          │
├─────────────────────────────────────────┤
│  7. LLVM优化                             │
│     - 优化Pass管道                       │
│     - 目标无关优化                       │
│  8. 代码生成                             │
│     - 目标相关优化                       │
│     - 机器码生成                         │
└─────────────────────────────────────────┘
    ↓ 目标文件 (.o)
┌─────────────────────────────────────────┐
│  链接 (Linking)                          │
├─────────────────────────────────────────┤
│  9. lld链接器                            │
│     - 快速链接                           │
│     - 多平台支持                         │
└─────────────────────────────────────────┘
    ↓ 可执行文件
┌─────────────────────────────────────────┐
│  调试支持 (Debugging)                    │
├─────────────────────────────────────────┤
│  10. lldb调试器                          │
│      - DWARF调试信息                     │
│      - 源码级调试                        │
└─────────────────────────────────────────┘
```

## 核心组件

### 1. 前端 (Frontend)

#### 1.1 词法分析器 (Lexer)
**文件**: `src/frontend/lexer.cpp`

**功能**:
- 多编码支持 (UTF-8, GBK, GB2312, GB18030等)
- Token流生成
- 位置信息跟踪
- C3风格错误处理

**技术**:
- 使用ICU库处理多编码
- 零拷贝token生成
- SIMD加速字符处理

#### 1.2 语法分析器 (Parser)
**文件**: `src/frontend/parser.cpp`

**功能**:
- 递归下降解析
- AST构建
- 错误恢复
- 增量解析支持

**技术**:
- 手写递归下降解析器
- 内存池分配AST节点
- 并行解析支持

#### 1.3 语义分析器 (Semantic Analyzer)
**文件**: `src/frontend/sema.cpp`

**功能**:
- 类型检查和推导
- 符号表管理
- 作用域分析
- 生命周期检查（未来）

**技术**:
- 基于约束的类型推导
- 增量类型检查
- 并行语义分析

### 2. 中间表示 (IR)

#### 2.1 MLIR-AIR层
**文件**: `src/mlir/air_dialect.cpp`

**AIR Dialect定义**:
```mlir
// AZ语言的高级IR表示
module {
  // 函数定义
  air.func @main() -> i32 {
    %0 = air.constant 42 : i32
    air.return %0 : i32
  }
  
  // 编译时执行
  air.comptime {
    %result = air.call @compute_at_compile_time()
  }
  
  // 类型定义
  air.struct @MyStruct {
    air.field "x" : i32
    air.field "y" : f64
  }
}
```

**优化Pass**:
- 常量折叠
- 死代码消除
- 内联优化
- 编译时执行

#### 2.2 LLVM IR层
**文件**: `src/mlir/llvm_lowering.cpp`

**降级策略**:
- AIR Dialect → Standard Dialect
- Standard Dialect → LLVM Dialect
- LLVM Dialect → LLVM IR

### 3. 后端 (Backend)

#### 3.1 LLVM优化
**文件**: `src/backend/optimizer.cpp`

**优化Pass管道**:
```cpp
// 优化级别配置
enum OptLevel {
    O0,  // 无优化
    O1,  // 基本优化
    O2,  // 标准优化
    O3,  // 激进优化
    Os,  // 大小优化
    Oz   // 极致大小优化
};

// Pass管道
PassManager pm;
pm.add(createInstructionCombiningPass());
pm.add(createReassociatePass());
pm.add(createGVNPass());
pm.add(createCFGSimplificationPass());
pm.add(createTailCallEliminationPass());
```

#### 3.2 代码生成
**文件**: `src/backend/codegen.cpp`

**支持的目标**:
- x86_64 (Linux, Windows, macOS)
- ARM64 (Linux, macOS, iOS, Android)
- RISC-V (Linux)
- WebAssembly

### 4. 链接器集成 (lld)

#### 4.1 链接器配置
**文件**: `src/linker/lld_driver.cpp`

**功能**:
- 快速链接
- LTO (Link-Time Optimization)
- 增量链接
- 多平台支持

**使用示例**:
```cpp
// 调用lld链接器
std::vector<const char*> args = {
    "lld",
    "-o", "output",
    "main.o",
    "lib.o",
    "-L/usr/lib",
    "-lc"
};
lld::elf::link(args);
```

### 5. 调试器集成 (lldb)

#### 5.1 调试信息生成
**文件**: `src/debug/dwarf_gen.cpp`

**功能**:
- DWARF调试信息生成
- 源码映射
- 变量信息
- 类型信息

**调试信息格式**:
```cpp
// 生成DWARF调试信息
DIBuilder builder(module);
DIFile *file = builder.createFile("main.az", "/path/to/project");
DICompileUnit *cu = builder.createCompileUnit(
    dwarf::DW_LANG_C_plus_plus,
    file,
    "AZ Compiler",
    false,
    "",
    0
);
```

#### 5.2 lldb插件
**文件**: `tools/lldb/az_plugin.py`

**功能**:
- AZ类型格式化
- 自定义命令
- 表达式求值

### 6. 包管理器 (az_mod)

#### 6.1 架构设计
**文件**: `tools/az_mod/`

**核心功能**:
```
az_mod/
├── src/
│   ├── resolver.rs      # 依赖解析
│   ├── fetcher.rs       # Git直连获取
│   ├── linker.rs        # 硬链接管理
│   ├── workspace.rs     # Workspace支持
│   └── cache.rs         # 缓存管理
├── Cargo.toml
└── README.md
```

**特性**:
- pnpm风格的workspace
- 硬链接节省空间
- Git直连依赖
- 并行下载和构建

#### 6.2 package.az格式
```toml
name = "my-project"
version = "0.1.0"
description = "My AZ project"

[dependencies]
std = "1.0.0"
http = { git = "https://gitee.com/az_lang/http", tag = "v0.2.0" }

[dev-dependencies]
test-framework = "0.1.0"

[workspace]
members = ["packages/*", "apps/*"]

[build]
target = "x86_64-unknown-linux-gnu"
opt-level = 2
lto = true
```

## 编译流程

### 完整编译命令

```bash
# 编译单个文件
az build main.az -o main

# 编译项目
az build --release

# 查看MLIR IR
az build main.az --emit-mlir

# 查看LLVM IR
az build main.az --emit-llvm

# 查看汇编
az build main.az --emit-asm

# 调试构建
az build --debug

# 交叉编译
az build --target=aarch64-linux-gnu
```

### 编译器驱动
**文件**: `src/driver/driver.cpp`

```cpp
class CompilerDriver {
public:
    Result<void> compile(const CompileOptions& opts) {
        // 1. 词法分析
        auto tokens = lexer.tokenize(source);
        if (!tokens.is_ok()) return tokens.error();
        
        // 2. 语法分析
        auto ast = parser.parse(tokens.value());
        if (!ast.is_ok()) return ast.error();
        
        // 3. 语义分析
        auto sema_result = sema.analyze(ast.value());
        if (!sema_result.is_ok()) return sema_result.error();
        
        // 4. MLIR生成
        auto mlir = mlir_gen.generate(ast.value());
        if (!mlir.is_ok()) return mlir.error();
        
        // 5. MLIR优化
        mlir_optimizer.optimize(mlir.value());
        
        // 6. LLVM IR生成
        auto llvm_ir = llvm_gen.lower(mlir.value());
        
        // 7. LLVM优化
        llvm_optimizer.optimize(llvm_ir);
        
        // 8. 代码生成
        auto obj = codegen.generate(llvm_ir);
        
        // 9. 链接
        auto exe = linker.link(obj);
        
        return Result::Ok();
    }
};
```

## 工具链组件

### 1. az (编译器驱动)
```bash
az build [options] <file>
az run <file>
az test
az fmt
az doc
```

### 2. chim (包管理器)
```bash
chim init <name>
chim add <package>
chim install
chim update
chim publish
```

### 3. az-lsp (语言服务器)
```bash
az-lsp --stdio
```

### 4. az-fmt (代码格式化)
```bash
az fmt <file>
az fmt --check
```

## 性能目标

### 编译速度
- 小项目 (<1000行): <100ms
- 中项目 (1万行): <1s
- 大项目 (10万行): <10s

### 优化
- 增量编译
- 并行编译
- 缓存中间结果
- 按需编译

### 内存使用
- 小项目: <50MB
- 中项目: <500MB
- 大项目: <2GB

## 依赖项

### 必需依赖
- LLVM 17.0+
- MLIR (包含在LLVM中)
- lld (LLVM链接器)
- lldb (LLVM调试器)
- ICU (国际化组件)

### 可选依赖
- Clang (C互操作)
- Rust (chim包管理器)
- Python (构建脚本)

## 构建系统

### CMake配置
**文件**: `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.20)
project(AZ_Compiler)

# 查找LLVM
find_package(LLVM 17 REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

# 包含目录
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# 编译器标志
add_definitions(${LLVM_DEFINITIONS})
set(CMAKE_CXX_STANDARD 17)

# 源文件
add_executable(az
    src/main.cpp
    src/frontend/lexer.cpp
    src/frontend/parser.cpp
    src/frontend/sema.cpp
    src/mlir/air_dialect.cpp
    src/mlir/llvm_lowering.cpp
    src/backend/optimizer.cpp
    src/backend/codegen.cpp
    src/linker/lld_driver.cpp
    src/debug/dwarf_gen.cpp
)

# 链接LLVM库
llvm_map_components_to_libnames(llvm_libs
    core
    support
    irreader
    codegen
    target
    x86codegen
    aarch64codegen
)

target_link_libraries(az ${llvm_libs} MLIRIR MLIRParser)
```

## 下一步实现

### Phase 1: 基础架构 (当前)
- [x] Bootstrap编译器（Python）
- [ ] C++编译器框架
- [ ] LLVM集成
- [ ] MLIR-AIR Dialect定义

### Phase 2: 核心功能
- [ ] 完整的前端实现
- [ ] MLIR IR生成
- [ ] LLVM IR生成
- [ ] 基本代码生成

### Phase 3: 工具链
- [ ] lld集成
- [ ] lldb集成
- [ ] chim包管理器
- [ ] LSP服务器

### Phase 4: 高级特性
- [ ] 编译时执行
- [ ] 所有权系统
- [ ] AZGC集成
- [ ] 完整标准库

## 总结

AZ编译器采用现代编译器架构，基于LLVM和MLIR-AIR构建，提供：
- 高性能编译
- 多平台支持
- 强大的优化能力
- 完整的工具链
- C3风格的错误处理

这是一个长期项目，当前的Bootstrap编译器是第一步，完整的LLVM/MLIR实现将在后续版本中完成。
