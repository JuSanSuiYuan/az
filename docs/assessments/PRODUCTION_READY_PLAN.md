# AZ语言生产就绪快速实现计划

**目标**: 在最短时间内让AZ可以用于生产环境  
**策略**: 务实、快速、可用  
**时间**: 2-4周

---

## 🎯 核心策略

### 方案选择：混合方案（最快）

```
Python Bootstrap → C代码 → Clang编译 → 可执行文件
```

**为什么不直接实现LLVM后端？**
- ❌ 时间太长（2-3个月）
- ❌ 复杂度高
- ❌ 风险大

**为什么选择C作为中间语言？**
- ✅ 已经实现了C代码生成
- ✅ Clang编译器成熟可靠
- ✅ 可以立即使用
- ✅ 性能接近原生

---

## 📅 实施计划（2周）

### 第1周：完善C代码生成

#### Day 1-2: 完善基础功能
- [x] 基本表达式 ✅
- [x] 函数定义 ✅
- [x] 控制流（if, while） ✅
- [ ] **for循环**
- [ ] **数组支持**
- [ ] **字符串操作**

#### Day 3-4: 实现标准库（C实现）
- [ ] **std.io** - 文件I/O
- [ ] **std.string** - 字符串操作
- [ ] **std.collections** - 基础数据结构
- [ ] **std.mem** - 内存管理

#### Day 5-7: 工具链完善
- [ ] **构建系统** - 自动化编译
- [ ] **包管理器基础** - 依赖管理
- [ ] **错误报告改进** - 更好的错误信息
- [ ] **测试框架** - 单元测试支持

### 第2周：生产就绪

#### Day 8-10: 实际项目测试
- [ ] **命令行工具** - 实现几个实用工具
- [ ] **Web服务器** - 简单的HTTP服务器
- [ ] **数据处理** - 文件处理工具
- [ ] **性能测试** - 基准测试

#### Day 11-12: 文档和示例
- [ ] **完整文档** - API文档
- [ ] **教程** - 从入门到实践
- [ ] **示例项目** - 真实项目示例
- [ ] **最佳实践** - 编码规范

#### Day 13-14: 发布准备
- [ ] **版本v0.5.0** - 生产预览版
- [ ] **安装程序** - 一键安装
- [ ] **CI/CD** - 自动化测试和发布
- [ ] **社区建设** - 文档网站、论坛

---

## 🚀 立即行动（今天开始）

### 步骤1: 完善C代码生成器

让我立即开始实现缺失的功能：

#### 1.1 添加for循环支持

```python
# 在 bootstrap/az_compiler.py 中添加
def gen_for(self, stmt: Stmt):
    """生成for循环"""
    # for (init; condition; update) { body }
    self.emit("for (")
    # 初始化
    if stmt.init:
        self.gen_stmt(stmt.init)
    self.emit("; ")
    # 条件
    if stmt.condition:
        condition = self.gen_expr(stmt.condition)
        self.emit(condition)
    self.emit("; ")
    # 更新
    if stmt.update:
        update = self.gen_expr(stmt.update)
        self.emit(update)
    self.emit(") {")
    self.indent_level += 1
    self.gen_stmt(stmt.body)
    self.indent_level -= 1
    self.emit("}")
```

#### 1.2 添加数组支持

```python
def gen_array_decl(self, stmt: Stmt):
    """生成数组声明"""
    type_str = self.map_type(stmt.element_type)
    self.emit(f"{type_str} {stmt.name}[{stmt.size}];")

def gen_array_access(self, expr: Expr):
    """生成数组访问"""
    array = self.gen_expr(expr.array)
    index = self.gen_expr(expr.index)
    return f"{array}[{index}]"
```

#### 1.3 添加字符串操作

```python
def gen_string_concat(self, expr: Expr):
    """生成字符串连接"""
    # 使用C的strcat
    left = self.gen_expr(expr.left)
    right = self.gen_expr(expr.right)
    return f"string_concat({left}, {right})"
```

### 步骤2: 实现最小标准库（C实现）

创建 `runtime/std.c`:

```c
// AZ语言运行时标准库

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ============================================================================
// std.io - 输入输出
// ============================================================================

void println(const char* str) {
    printf("%s\n", str);
}

void print(const char* str) {
    printf("%s", str);
}

char* read_line() {
    char* line = NULL;
    size_t len = 0;
    getline(&line, &len, stdin);
    return line;
}

// ============================================================================
// std.string - 字符串操作
// ============================================================================

char* string_concat(const char* a, const char* b) {
    size_t len = strlen(a) + strlen(b) + 1;
    char* result = malloc(len);
    strcpy(result, a);
    strcat(result, b);
    return result;
}

int string_length(const char* str) {
    return strlen(str);
}

char* string_substring(const char* str, int start, int end) {
    int len = end - start;
    char* result = malloc(len + 1);
    strncpy(result, str + start, len);
    result[len] = '\0';
    return result;
}

// ============================================================================
// std.fs - 文件系统
// ============================================================================

char* read_file(const char* path) {
    FILE* file = fopen(path, "r");
    if (!file) return NULL;
    
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    char* content = malloc(size + 1);
    fread(content, 1, size, file);
    content[size] = '\0';
    
    fclose(file);
    return content;
}

int write_file(const char* path, const char* content) {
    FILE* file = fopen(path, "w");
    if (!file) return -1;
    
    fputs(content, file);
    fclose(file);
    return 0;
}

// ============================================================================
// std.collections - 集合
// ============================================================================

typedef struct {
    void** data;
    int size;
    int capacity;
} Vec;

Vec* vec_new() {
    Vec* vec = malloc(sizeof(Vec));
    vec->data = malloc(sizeof(void*) * 10);
    vec->size = 0;
    vec->capacity = 10;
    return vec;
}

void vec_push(Vec* vec, void* item) {
    if (vec->size >= vec->capacity) {
        vec->capacity *= 2;
        vec->data = realloc(vec->data, sizeof(void*) * vec->capacity);
    }
    vec->data[vec->size++] = item;
}

void* vec_get(Vec* vec, int index) {
    if (index < 0 || index >= vec->size) return NULL;
    return vec->data[index];
}

int vec_len(Vec* vec) {
    return vec->size;
}

void vec_free(Vec* vec) {
    free(vec->data);
    free(vec);
}
```

### 步骤3: 创建构建系统

创建 `az` 命令行工具（Python脚本）:

```python
#!/usr/bin/env python3
"""
AZ语言编译器命令行工具
"""

import sys
import os
import subprocess
import argparse

def compile_az(source_file, output_file=None, optimize=False):
    """编译AZ源文件"""
    
    # 1. 生成C代码
    c_file = source_file.replace('.az', '.c')
    cmd = [
        'python', 'bootstrap/az_compiler.py',
        source_file,
        '--emit-c', '-o', c_file
    ]
    
    print(f"[1/3] 编译 {source_file} -> {c_file}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("❌ 编译失败")
        return False
    
    # 2. 使用Clang编译C代码
    if output_file is None:
        output_file = source_file.replace('.az', '')
    
    clang_cmd = ['clang', c_file, 'runtime/std.c', '-o', output_file]
    if optimize:
        clang_cmd.insert(1, '-O3')
    
    print(f"[2/3] 编译 {c_file} -> {output_file}")
    result = subprocess.run(clang_cmd)
    if result.returncode != 0:
        print("❌ 编译失败")
        return False
    
    # 3. 清理临时文件
    print(f"[3/3] 清理临时文件")
    os.remove(c_file)
    
    print(f"✅ 编译成功: {output_file}")
    return True

def main():
    parser = argparse.ArgumentParser(description='AZ语言编译器')
    parser.add_argument('source', help='源文件')
    parser.add_argument('-o', '--output', help='输出文件')
    parser.add_argument('-O', '--optimize', action='store_true', help='优化')
    parser.add_argument('--run', action='store_true', help='编译后运行')
    
    args = parser.parse_args()
    
    success = compile_az(args.source, args.output, args.optimize)
    
    if success and args.run:
        output = args.output or args.source.replace('.az', '')
        print(f"\n运行 {output}:")
        subprocess.run([f'./{output}'])

if __name__ == '__main__':
    main()
```

使用方法:
```bash
# 编译
python az examples/hello.az -o hello

# 编译并运行
python az examples/hello.az --run

# 优化编译
python az examples/hello.az -O -o hello
```

---

## 📦 最小可用产品（MVP）功能清单

### 核心功能（必须）

- [x] 变量声明（let/var）
- [x] 函数定义
- [x] 基本运算
- [x] 控制流（if/while）
- [ ] **for循环**
- [ ] **数组**
- [ ] **字符串操作**
- [x] 函数调用
- [x] 递归

### 标准库（必须）

- [ ] **std.io** - println, print, read_line, read_file, write_file
- [ ] **std.string** - concat, length, substring, split
- [ ] **std.collections** - Vec, Map
- [ ] **std.mem** - malloc, free

### 工具（必须）

- [ ] **az命令** - 编译工具
- [ ] **包管理** - 基础依赖管理
- [ ] **测试框架** - 单元测试

### 文档（必须）

- [ ] **快速入门**
- [ ] **API文档**
- [ ] **示例项目**

---

## 🎯 生产就绪标准

### 功能完整性

```
必须功能: ████████████████████ 100%
├── 核心语法      ████████████████████ 100%
├── 标准库        ████████████████████ 100%
├── 工具链        ████████████████████ 100%
└── 文档          ████████████████████ 100%
```

### 稳定性

- ✅ 所有测试通过
- ✅ 无已知严重bug
- ✅ 错误处理完善
- ✅ 内存安全

### 性能

- ✅ 编译速度可接受（<5秒/1000行）
- ✅ 运行性能接近C（90%+）
- ✅ 内存使用合理

### 可用性

- ✅ 安装简单（一键安装）
- ✅ 文档完整
- ✅ 示例丰富
- ✅ 错误信息清晰

---

## 📈 时间表

### Week 1: 核心功能

| Day | 任务 | 状态 |
|-----|------|------|
| 1 | for循环、数组 | 📋 |
| 2 | 字符串操作 | 📋 |
| 3 | std.io实现 | 📋 |
| 4 | std.string实现 | 📋 |
| 5 | std.collections实现 | 📋 |
| 6 | az命令工具 | 📋 |
| 7 | 测试和修复 | 📋 |

### Week 2: 生产就绪

| Day | 任务 | 状态 |
|-----|------|------|
| 8 | 实际项目测试 | 📋 |
| 9 | 性能优化 | 📋 |
| 10 | 文档完善 | 📋 |
| 11 | 示例项目 | 📋 |
| 12 | 安装程序 | 📋 |
| 13 | CI/CD | 📋 |
| 14 | 发布v0.5.0 | 📋 |

---

## 🚀 立即开始

### 今天要做的事

1. **完善C代码生成器**
   - 添加for循环支持
   - 添加数组支持
   - 添加字符串操作

2. **实现标准库**
   - 创建runtime/std.c
   - 实现基础函数

3. **创建az命令**
   - 编写Python脚本
   - 测试编译流程

### 明天要做的事

1. **测试和修复**
   - 运行所有示例
   - 修复发现的bug

2. **开始文档**
   - 快速入门指南
   - API文档

---

## 💡 关键决策

### 为什么不等LLVM后端？

**时间对比**:
- LLVM后端: 2-3个月
- C代码方案: 2周

**质量对比**:
- LLVM后端: 更好的优化，但复杂
- C代码方案: 足够好，简单可靠

**结论**: 先用C代码方案快速上线，后续再优化

### 为什么不实现完整标准库？

**策略**: 最小可用产品（MVP）

只实现最常用的功能:
- ✅ I/O操作
- ✅ 字符串处理
- ✅ 基础集合
- ❌ 网络（后续）
- ❌ 并发（后续）
- ❌ 加密（后续）

---

## 📊 成功指标

### 2周后应该达到

- ✅ 可以编译实际项目
- ✅ 性能接近C（90%+）
- ✅ 有完整的文档
- ✅ 有实际的示例项目
- ✅ 安装和使用简单

### 示例项目

1. **命令行工具**
   ```az
   // grep工具
   fn main() int {
       let pattern = args[1];
       let file = read_file(args[2]);
       // 搜索和输出
       return 0;
   }
   ```

2. **文件处理**
   ```az
   // CSV处理
   fn main() int {
       let data = read_file("data.csv");
       let lines = string_split(data, "\n");
       // 处理数据
       return 0;
   }
   ```

3. **简单服务器**
   ```az
   // HTTP服务器
   fn main() int {
       let server = create_server(8080);
       server.listen();
       return 0;
   }
   ```

---

## 🎉 总结

### 策略

**务实 > 完美**
- 先快速可用
- 后续优化改进

### 时间表

**2周生产就绪**
- Week 1: 核心功能
- Week 2: 完善和发布

### 目标

**v0.5.0 - 生产预览版**
- 可以用于实际项目
- 性能可接受
- 文档完整

---

**让我们开始吧！** 🚀

**第一步**: 完善C代码生成器（今天）  
**第二步**: 实现标准库（明天）  
**第三步**: 创建工具链（本周）  
**第四步**: 测试和发布（下周）

**目标**: 2周后，AZ可以用于生产！
