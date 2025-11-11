# AZ编程语言技术栈

## 核心技术栈

AZ语言基于**LLVM/Clang**生态系统构建，拒绝使用GCC和MSVC。

### 编译器基础设施

```
┌─────────────────────────────────────────┐
│         AZ编程语言技术栈                 │
├─────────────────────────────────────────┤
│                                         │
│  前端: C++ (Clang编译)                  │
│  ├─ 词法分析器                          │
│  ├─ 语法分析器                          │
│  └─ 语义分析器                          │
│                                         │
│  中间表示: MLIR                         │
│  ├─ AZ Dialect (自定义方言)            │
│  ├─ Standard Dialects                  │
│  └─ 渐进式降级                          │
│                                         │
│  后端: LLVM                             │
│  ├─ LLVM IR生成                        │
│  ├─ 优化Pass                           │
│  └─ 代码生成                            │
│                                         │
│  链接器: LLD (LLVM Linker)             │
│                                         │
│  调试器: LLDB                           │
│                                         │
│  工具链: Clang/LLVM Tools              │
│                                         │
└─────────────────────────────────────────┘
```

## 为什么选择LLVM/Clang？

### 1. 现代化架构
- ✅ 模块化设计
- ✅ 清晰的API
- ✅ 优秀的文档
- ✅ 活跃的社区

### 2. MLIR支持
- ✅ 多级IR表示
- ✅ 方言系统
- ✅ 渐进式降级
- ✅ 易于扩展

### 3. 跨平台支持
- ✅ Windows
- ✅ Linux
- ✅ macOS
- ✅ BSD系统

### 4. 优秀的优化
- ✅ 强大的优化Pass
- ✅ LTO（链接时优化）
- ✅ PGO（配置文件引导优化）
- ✅ 向量化

### 5. 完整的工具链
- ✅ Clang - C/C++编译器
- ✅ LLD - 快速链接器
- ✅ LLDB - 调试器
- ✅ Clang-Format - 代码格式化
- ✅ Clang-Tidy - 静态分析

## 技术细节

### 编译流程

```
AZ源代码 (.az)
    ↓
[词法分析] → Tokens
    ↓
[语法分析] → AST
    ↓
[语义分析] → 类型检查的AST
    ↓
[MLIR生成] → AZ Dialect
    ↓
[MLIR降级] → Standard Dialects
    ↓
[LLVM IR生成] → LLVM IR (.ll)
    ↓
[LLVM优化] → 优化的LLVM IR
    ↓
[代码生成] → 目标代码 (.o)
    ↓
[LLD链接] → 可执行文件
```

### 使用的LLVM组件

#### 核心库
- **LLVM Core** - IR表示和基础设施
- **LLVM Support** - 工具类和数据结构
- **LLVM Analysis** - 分析Pass
- **LLVM Transforms** - 优化Pass
- **LLVM CodeGen** - 代码生成

#### MLIR
- **MLIR Core** - MLIR基础设施
- **MLIR Dialects** - 标准方言
- **MLIR Transforms** - 转换Pass
- **MLIR Translation** - LLVM IR转换

#### 工具
- **Clang** - C/C++前端（用于编译AZ编译器本身）
- **LLD** - 链接器
- **LLDB** - 调试器

### 编译命令

#### 编译AZ编译器（使用Clang）

```bash
# 配置CMake使用Clang
cmake -B build \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm

# 编译
cmake --build build -j$(nproc)
```

#### 使用AZ编译器

```bash
# 编译AZ代码到LLVM IR
az compile input.az -o output.ll --emit-llvm

# 编译到目标代码
az compile input.az -o output.o

# 编译到可执行文件
az compile input.az -o output

# 使用优化
az compile input.az -o output -O3

# 使用LTO
az compile input.az -o output -flto
```

## 版本要求

### 最低版本
- **LLVM**: 17.0+
- **Clang**: 17.0+
- **CMake**: 3.20+
- **C++**: C++17

### 推荐版本
- **LLVM**: 18.0+ 或 19.0+
- **Clang**: 18.0+ 或 19.0+
- **CMake**: 3.25+
- **C++**: C++20

### 当前测试版本
- **LLVM**: 20.1.8 ✅
- **Clang**: 20.1.8 ✅

## 安装LLVM/Clang

### Windows

#### 方法1: 官方安装包
```bash
# 下载LLVM安装包
# https://releases.llvm.org/download.html

# 安装后添加到PATH
setx PATH "%PATH%;C:\Program Files\LLVM\bin"
```

#### 方法2: Visual Studio
```bash
# 在Visual Studio Installer中选择
# "使用C++的桌面开发" → "LLVM (clang-cl)"
```

#### 方法3: Chocolatey
```bash
choco install llvm
```

### Linux

#### Ubuntu/Debian
```bash
# 添加LLVM仓库
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 18

# 安装
sudo apt install clang-18 llvm-18 lld-18 lldb-18
```

#### Fedora
```bash
sudo dnf install clang llvm lld lldb
```

#### Arch Linux
```bash
sudo pacman -S clang llvm lld lldb
```

### macOS

```bash
# 使用Homebrew
brew install llvm

# 添加到PATH
echo 'export PATH="/usr/local/opt/llvm/bin:$PATH"' >> ~/.zshrc
```

## 构建选项

### CMake配置选项

```bash
# 基本配置
cmake -B build \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Release

# 启用LTO
cmake -B build \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON

# 启用MLIR
cmake -B build \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DAZ_ENABLE_MLIR=ON \
  -DMLIR_DIR=/path/to/llvm/lib/cmake/mlir

# 启用测试
cmake -B build \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Debug \
  -DAZ_BUILD_TESTS=ON
```

## 优化选项

### 编译时优化

```bash
# O0 - 无优化（调试）
clang -O0 -g code.c -o code

# O1 - 基本优化
clang -O1 code.c -o code

# O2 - 推荐优化（平衡）
clang -O2 code.c -o code

# O3 - 激进优化（性能）
clang -O3 code.c -o code

# Os - 优化大小
clang -Os code.c -o code

# Oz - 最小化大小
clang -Oz code.c -o code
```

### 链接时优化（LTO）

```bash
# 启用LTO
clang -flto code.c -o code

# 使用ThinLTO（更快）
clang -flto=thin code.c -o code

# 完整LTO（更好的优化）
clang -flto=full code.c -o code
```

### 配置文件引导优化（PGO）

```bash
# 1. 生成插桩代码
clang -fprofile-generate code.c -o code

# 2. 运行程序收集数据
./code

# 3. 使用配置文件优化
clang -fprofile-use=default.profdata code.c -o code
```

## 调试工具

### LLDB调试

```bash
# 编译带调试信息
clang -g code.c -o code

# 启动LLDB
lldb code

# LLDB命令
(lldb) breakpoint set --name main
(lldb) run
(lldb) step
(lldb) print variable
(lldb) continue
```

### Clang静态分析

```bash
# 运行静态分析
clang --analyze code.c

# 使用Clang-Tidy
clang-tidy code.c -- -I/path/to/includes
```

### 生成LLVM IR

```bash
# 生成人类可读的LLVM IR
clang -S -emit-llvm code.c -o code.ll

# 生成二进制LLVM IR
clang -c -emit-llvm code.c -o code.bc

# 查看LLVM IR
llvm-dis code.bc -o code.ll
cat code.ll
```

## 性能分析

### 使用LLVM工具

```bash
# 查看优化报告
clang -O3 -Rpass=.* code.c -o code

# 查看向量化报告
clang -O3 -Rpass=loop-vectorize code.c -o code

# 查看内联报告
clang -O3 -Rpass=inline code.c -o code
```

### 使用perf（Linux）

```bash
# 编译带性能计数器
clang -O3 -g code.c -o code

# 运行性能分析
perf record ./code
perf report
```

## 与其他编译器的对比

| 特性 | LLVM/Clang | GCC | MSVC |
|------|-----------|-----|------|
| **模块化** | ✅ 优秀 | ⚠️ 一般 | ❌ 较差 |
| **MLIR支持** | ✅ 原生 | ❌ 无 | ❌ 无 |
| **跨平台** | ✅ 优秀 | ✅ 优秀 | ❌ 仅Windows |
| **API清晰度** | ✅ 优秀 | ⚠️ 一般 | ⚠️ 一般 |
| **文档** | ✅ 优秀 | ⚠️ 一般 | ✅ 良好 |
| **编译速度** | ✅ 快 | ⚠️ 中等 | ✅ 快 |
| **优化质量** | ✅ 优秀 | ✅ 优秀 | ✅ 良好 |
| **LTO** | ✅ 优秀 | ✅ 良好 | ✅ 良好 |
| **调试器** | ✅ LLDB | ✅ GDB | ✅ VS调试器 |
| **许可证** | ✅ Apache 2.0 | ✅ GPL | ❌ 专有 |

## AZ语言的选择

### ✅ 选择LLVM/Clang的原因

1. **MLIR原生支持** - 这是最重要的原因
2. **模块化架构** - 易于扩展和维护
3. **清晰的API** - 降低开发难度
4. **跨平台** - 一次编写，到处编译
5. **现代化** - 持续更新，支持最新特性
6. **工具链完整** - 编译、链接、调试一体化
7. **许可证友好** - Apache 2.0，商业友好

### ❌ 不选择GCC的原因

1. **无MLIR支持** - 这是致命缺陷
2. **API复杂** - 难以集成
3. **GPL许可证** - 对商业应用不友好
4. **架构老旧** - 难以扩展

### ❌ 不选择MSVC的原因

1. **无MLIR支持** - 致命缺陷
2. **仅Windows** - 不跨平台
3. **专有软件** - 不开源
4. **API封闭** - 难以集成

## 未来计划

### 短期（3-6个月）
- ✅ 完成MLIR生成器
- ✅ 实现LLVM后端
- ✅ 集成LLD链接器
- ✅ 基本的LLDB支持

### 中期（6-12个月）
- 📋 优化Pass开发
- 📋 自定义MLIR Dialect
- 📋 JIT编译支持
- 📋 增量编译

### 长期（1-2年）
- 📋 完整的调试信息
- 📋 配置文件引导优化
- 📋 自动向量化
- 📋 GPU代码生成

## 资源链接

### 官方文档
- **LLVM**: https://llvm.org/docs/
- **Clang**: https://clang.llvm.org/docs/
- **MLIR**: https://mlir.llvm.org/docs/
- **LLD**: https://lld.llvm.org/
- **LLDB**: https://lldb.llvm.org/

### 下载
- **LLVM Releases**: https://releases.llvm.org/
- **GitHub**: https://github.com/llvm/llvm-project

### 社区
- **LLVM Discourse**: https://discourse.llvm.org/
- **LLVM Discord**: https://discord.gg/xS7Z362
- **Stack Overflow**: https://stackoverflow.com/questions/tagged/llvm

## 总结

AZ语言坚定地选择**LLVM/Clang**作为唯一的技术栈：

- ✅ **编译器**: Clang/Clang++
- ✅ **中间表示**: MLIR → LLVM IR
- ✅ **链接器**: LLD
- ✅ **调试器**: LLDB
- ✅ **工具**: LLVM工具链

**拒绝使用GCC和MSVC！**

这个选择确保了：
- 🎯 技术栈统一
- 🎯 MLIR原生支持
- 🎯 跨平台能力
- 🎯 现代化架构
- 🎯 持续发展

---

**AZ语言 - 基于LLVM/Clang的现代系统编程语言** 🚀
