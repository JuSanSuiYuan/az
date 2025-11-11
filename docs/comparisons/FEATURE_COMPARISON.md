# AZ编程语言 - 功能对比

**日期**: 2025年10月29日  
**版本**: v0.3.0-dev

---

## 📊 与其他语言对比

### 系统编程语言对比

| 特性 | AZ | C | C++ | Rust | Zig | C3 |
|------|----|----|-----|------|-----|-----|
| **错误处理** | Result | errno | 异常 | Result | Error Union | Result |
| **内存管理** | 计划中 | 手动 | 手动/RAII | 所有权 | 手动 | 手动+可选GC |
| **类型系统** | 静态 | 静态 | 静态 | 静态 | 静态 | 静态 |
| **编译时执行** | 计划中 | 宏 | constexpr | const fn | comptime | comptime |
| **泛型** | 计划中 | ❌ | ✅ | ✅ | ✅ | ✅ |
| **模式匹配** | 计划中 | ❌ | ❌ | ✅ | ✅ | ✅ |
| **学习曲线** | 低 | 低 | 高 | 高 | 中 | 低 |
| **编译速度** | 快 | 快 | 慢 | 慢 | 快 | 快 |
| **运行性能** | 目标90% | 100% | 100% | 95% | 100% | 95% |
| **生态系统** | 新 | 成熟 | 成熟 | 成长中 | 新 | 新 |
| **成熟度** | 开发中 | 成熟 | 成熟 | 成熟 | 开发中 | 开发中 |

### 错误处理对比

#### AZ (C3风格)
```az
fn divide(a: int, b: int) Result<int> {
    if (b == 0) {
        return Result.Err(error("除数不能为零"));
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

#### C (errno)
```c
int divide(int a, int b, int* result) {
    if (b == 0) {
        errno = EINVAL;
        return -1;
    }
    *result = a / b;
    return 0;
}

int main() {
    int result;
    if (divide(10, 2, &result) == 0) {
        printf("%d\n", result);
    } else {
        perror("divide");
    }
    return 0;
}
```

#### C++ (异常)
```cpp
int divide(int a, int b) {
    if (b == 0) {
        throw std::invalid_argument("除数不能为零");
    }
    return a / b;
}

int main() {
    try {
        int result = divide(10, 2);
        std::cout << result << std::endl;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
    return 0;
}
```

#### Rust (Result)
```rust
fn divide(a: i32, b: i32) -> Result<i32, String> {
    if b == 0 {
        return Err("除数不能为零".to_string());
    }
    Ok(a / b)
}

fn main() {
    match divide(10, 2) {
        Ok(result) => println!("{}", result),
        Err(e) => eprintln!("{}", e),
    }
}
```

### 优势分析

#### AZ的优势

1. **C3风格错误处理** ⭐⭐⭐⭐⭐
   - 明确的错误处理
   - 零运行时开销
   - 比C的errno更安全
   - 比C++的异常更高效

2. **现代语法** ⭐⭐⭐⭐
   - 类似C，易于学习
   - 融合现代特性
   - 中英文支持

3. **MLIR架构** ⭐⭐⭐⭐⭐
   - 现代化的IR设计
   - 强大的优化能力
   - 可扩展性强

4. **完整文档** ⭐⭐⭐⭐⭐
   - 18,000+行文档
   - 详细的教程
   - 清晰的示例

#### AZ的劣势

1. **不成熟** ⭐☆☆☆☆
   - 仍在开发中
   - 功能不完整
   - 生态系统缺失

2. **无法生成可执行文件** ⭐☆☆☆☆
   - 仅解释执行
   - 性能较低
   - 不适合生产

3. **标准库缺失** ⭐☆☆☆☆
   - 功能有限
   - 需要自己实现

## 🎯 适用场景

### ✅ 推荐使用

1. **编译器学习** ⭐⭐⭐⭐⭐
   - 完整的实现
   - 清晰的代码
   - 详细的文档

2. **语言设计研究** ⭐⭐⭐⭐⭐
   - C3风格错误处理
   - MLIR集成
   - 类型系统设计

3. **教学演示** ⭐⭐⭐⭐⭐
   - 展示编译器原理
   - 演示类型检查
   - 讲解错误处理

4. **快速原型** ⭐⭐⭐⭐☆
   - 验证语法设计
   - 测试语言特性
   - 编写示例代码

### ❌ 不推荐使用

1. **生产环境** ❌
   - 功能不完整
   - 性能不足
   - 缺少支持

2. **大型项目** ❌
   - 工具链不完善
   - 标准库缺失
   - 生态系统不成熟

3. **性能关键应用** ❌
   - 仅解释执行
   - 无优化
   - 性能较低

## 🔮 未来展望

### 短期（1-3个月）

**目标**: 生成可执行文件

- [ ] 完善MLIR生成
- [ ] 实现LLVM后端
- [ ] lld链接器集成
- [ ] 第一个可执行文件

**预期成果**:
```bash
$ az build hello.az -o hello
$ ./hello
Hello, AZ!
```

### 中期（3-6个月）

**目标**: 基础工具链

- [ ] lldb调试器集成
- [ ] 基础标准库
- [ ] az_mod包管理器
- [ ] LSP服务器

**预期成果**:
```bash
$ az_mod init my-project
$ cd my-project
$ az build
$ ./my-project
```

### 长期（6-12个月）

**目标**: 生产就绪

- [ ] 完整标准库
- [ ] 所有权系统
- [ ] AZGC垃圾回收器
- [ ] 完整工具链
- [ ] v1.0.0发布

**预期成果**:
```bash
$ az build compiler/main.az -o az_new
$ ./az_new --version
AZ Compiler v1.0.0
```

## 📊 性能对比（预期）

### 编译速度

| 语言 | 小项目 | 中项目 | 大项目 |
|------|--------|--------|--------|
| **AZ** | <100ms | <1s | <10s |
| C | <50ms | <500ms | <5s |
| C++ | <200ms | <5s | <60s |
| Rust | <500ms | <10s | <120s |
| Zig | <100ms | <1s | <10s |

### 运行性能（预期）

| 语言 | 相对性能 | 说明 |
|------|---------|------|
| C | 100% | 基准 |
| C++ | 100% | 与C相当 |
| **AZ** | **90%** | **目标** |
| Rust | 95% | 接近C |
| Zig | 100% | 与C相当 |

### 内存使用（预期）

| 语言 | 编译器内存 | 运行时内存 |
|------|-----------|-----------|
| **AZ** | **<500MB** | **最小** |
| C | <100MB | 最小 |
| C++ | <500MB | 最小 |
| Rust | <2GB | 最小 |
| Zig | <500MB | 最小 |

## 🌟 独特优势

### 1. C3风格错误处理

**AZ是少数采用C3风格Result类型的系统编程语言**

优势：
- 明确的错误处理
- 零运行时开销
- 编译时检查
- 适合系统编程

### 2. MLIR架构

**AZ是少数基于MLIR的新语言**

优势：
- 现代化的IR设计
- 强大的优化能力
- 可扩展的Dialect系统
- 渐进式降级

### 3. 多编码支持

**真正的国际化**

优势：
- 支持多种编码
- 中英文关键字
- 适合中文开发者

### 4. 完整文档

**18,000+行文档**

优势：
- 降低学习门槛
- 促进社区发展
- 提高代码质量

## 📞 获取帮助

### 资源

- **GitHub**: https://github.com/JuSanSuiYuan/az
- **文档**: 查看项目根目录的Markdown文件
- **示例**: examples/目录
- **Issues**: https://github.com/JuSanSuiYuan/az/issues
- **Discussions**: https://github.com/JuSanSuiYuan/az/discussions

### 社区

- 报告bug: GitHub Issues
- 功能建议: GitHub Issues
- 一般讨论: GitHub Discussions
- 贡献代码: Pull Requests

## 📄 许可证

本项目采用木兰宽松许可证2.0（Mulan Permissive License，Version 2）。

---

**AZ编程语言 - 现代、安全、高效的系统编程语言**

**GitHub**: https://github.com/JuSanSuiYuan/az

⭐ 如果您喜欢这个项目，请给我们一个Star！
