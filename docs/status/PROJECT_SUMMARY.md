# AZ编程语言项目总结

**项目地址**: https://github.com/JuSanSuiYuan/az  
**创建日期**: 2025年10月29日  
**当前版本**: v0.2.0-dev  
**许可证**: 木兰宽松许可证2.0

## 项目概述

AZ是一种现代系统编程语言，结合了C3和Zig的优点，以C3为主要基础，Zig为辅助补充。项目的核心特色是**采用C3风格的Result类型进行错误处理**，基于**LLVM和MLIR-AIR**构建编译器，使用**lld**作为链接器，**lldb**作为调试器，**chim**作为包管理器。

## 核心特性

### 1. C3风格的错误处理 ⭐⭐⭐⭐⭐
- 使用Result类型而不是异常
- 明确的错误处理路径
- 零运行时开销
- 适合系统编程

### 2. 现代编译器架构
- 基于LLVM 17+和MLIR-AIR
- 多阶段编译流程
- 强大的优化能力
- 多平台支持

### 3. 完整的工具链
- **az**: 编译器驱动
- **lld**: 快速链接器
- **lldb**: 强大的调试器
- **chim**: pnpm风格的包管理器

### 4. 多编码支持
- UTF-8（默认）
- GBK, GB2312, GB18030
- UTF-16, UTF-32
- 中英文关键字

## 已完成的工作

### ✅ v0.1.0 - Bootstrap版本

1. **Python实现的完整编译器**
   - 词法分析器（~200行）
   - 语法分析器（~300行）
   - 语义分析器（~200行）
   - 解释执行器（~300行）
   - C3风格错误处理（~100行）

2. **5个示例程序**
   - hello.az - Hello World
   - variables.az - 变量和运算
   - functions.az - 函数示例
   - control_flow.az - 控制流
   - fibonacci.az - 递归函数

3. **完整文档**（~10,000行）
   - README.md - 项目主页
   - QUICKSTART.md - 快速入门
   - README_COMPILER.md - 编译器文档
   - ARCHITECTURE.md - 架构设计
   - ROADMAP.md - 开发路线图
   - 等等...

### ✅ v0.2.0-dev - C++实现（进行中）

1. **C++编译器框架**
   - 完整的词法分析器（~400行C++）
   - 完整的语法分析器（~600行C++）
   - AST定义（~200行C++）
   - Result类型实现（~150行C++）

2. **MLIR-AIR Dialect定义**
   - TableGen定义（~300行）
   - 操作定义（20+个操作）
   - 类型定义（5种类型）
   - Result类型支持

3. **构建系统**
   - CMake配置
   - 构建脚本（Linux/macOS/Windows）
   - 测试框架
   - CI/CD准备

4. **测试**
   - 词法分析器测试
   - 语法分析器测试
   - 集成测试

5. **工具设计**
   - chim包管理器设计文档
   - LSP服务器规划
   - 调试器集成方案

## 项目结构

```
az/
├── bootstrap/              # Bootstrap编译器（Python）
│   ├── az_compiler.py     # 主编译器（~1000行）
│   └── README.md
├── include/               # C++头文件
│   └── AZ/
│       ├── Frontend/      # 前端
│       ├── Support/       # 支持库
│       └── IR/           # MLIR定义
├── lib/                   # C++实现
│   ├── Frontend/         # 前端实现
│   └── Support/          # 支持库实现
├── tools/                 # 工具
│   ├── az/               # 编译器驱动
│   └── chim/             # 包管理器（设计）
├── compiler/              # AZ语言编写的编译器
│   ├── ast.az
│   ├── lexer.az
│   ├── parser.az
│   └── ...
├── examples/              # 示例程序
│   ├── hello.az
│   ├── variables.az
│   └── ...
├── test/                  # 测试
│   ├── lexer_test.cpp
│   └── parser_test.cpp
├── docs/                  # 文档
│   ├── README.md
│   ├── AZGC.md
│   └── ...
├── CMakeLists.txt        # CMake配置
├── README.md             # 项目主页
├── QUICKSTART.md         # 快速入门
├── BUILD.md              # 构建指南
├── ARCHITECTURE.md       # 架构设计
├── ROADMAP.md            # 开发路线图
├── STATUS.md             # 实现状态
├── CONTRIBUTING.md       # 贡献指南
└── LICENSE               # 许可证
```

## 代码统计

| 组件 | 语言 | 行数 | 文件数 |
|------|------|------|--------|
| Bootstrap编译器 | Python | ~1,000 | 1 |
| C++编译器 | C++ | ~2,000 | 15 |
| AZ编译器源码 | AZ | ~1,500 | 8 |
| MLIR定义 | TableGen | ~300 | 1 |
| 测试 | C++ | ~500 | 2 |
| 示例程序 | AZ | ~200 | 5 |
| 文档 | Markdown | ~15,000 | 20+ |
| 构建脚本 | Shell/Batch | ~200 | 3 |
| **总计** | | **~20,700** | **55+** |

## 技术栈

### 编译器
- **语言**: C++17
- **IR**: MLIR-AIR, LLVM IR
- **构建**: CMake, Ninja
- **测试**: CTest, GoogleTest

### 工具链
- **链接器**: lld
- **调试器**: lldb
- **包管理器**: chim (Rust)

### 依赖
- LLVM 17+
- MLIR (包含在LLVM中)
- ICU (国际化)

## 关键设计决策

### 1. 为什么选择C3风格的错误处理？

**优势**:
- 明确性：错误处理路径清晰可见
- 性能：避免异常的栈展开开销
- 可靠性：编译时强制检查
- 系统编程友好：无运行时依赖

**实现**:
```cpp
template<typename T>
class Result {
    bool is_ok_;
    union {
        T value_;
        CompileError error_;
    };
public:
    static Result Ok(T value);
    static Result Err(CompileError error);
    bool isOk() const;
    T& value();
    CompileError& error();
};
```

### 2. 为什么选择LLVM和MLIR？

**LLVM优势**:
- 成熟的优化框架
- 多平台支持
- 活跃的社区
- 工具链完整

**MLIR优势**:
- 高级IR抽象
- 可扩展的Dialect系统
- 渐进式降级
- 编译时执行支持

### 3. 为什么设计chim包管理器？

**灵感来源**: pnpm

**特性**:
- 硬链接节省空间
- Workspace支持
- Git直连
- 并行构建

## 性能目标

### 编译速度
| 项目规模 | 目标时间 |
|---------|---------|
| 小项目 (<1000行) | <100ms |
| 中项目 (1万行) | <1s |
| 大项目 (10万行) | <10s |

### 运行性能
- 目标：达到C/C++的90%+性能
- 方法：LLVM优化 + 零成本抽象

### 内存使用
| 项目规模 | 目标内存 |
|---------|---------|
| 小项目 | <50MB |
| 中项目 | <500MB |
| 大项目 | <2GB |

## 开发路线图

### 短期（3-6个月）
- [x] Bootstrap编译器
- [x] C++前端框架
- [ ] 完整的语义分析
- [ ] MLIR-AIR实现
- [ ] 基本代码生成

### 中期（6-12个月）
- [ ] LLVM后端完成
- [ ] lld集成
- [ ] lldb集成
- [ ] chim包管理器
- [ ] 基础标准库

### 长期（12-24个月）
- [ ] 编译时执行
- [ ] 所有权系统
- [ ] AZGC垃圾回收器
- [ ] 完整标准库
- [ ] v1.0.0发布

## 社区和贡献

### 当前贡献者
- 项目创建者和主要开发者

### 贡献机会
1. **C++前端开发** - 语义分析器
2. **MLIR实现** - Dialect和Pass
3. **测试** - 单元测试和集成测试
4. **文档** - 教程和API文档
5. **工具** - LSP、格式化器等

### 如何参与
1. Fork项目: https://github.com/JuSanSuiYuan/az
2. 阅读[CONTRIBUTING.md](CONTRIBUTING.md)
3. 选择感兴趣的任务
4. 提交Pull Request

## 里程碑

### ✅ M1: 概念验证（2025年10月29日）
- Bootstrap编译器完成
- 语言设计验证
- 基本功能实现

### 🚧 M2: C++前端（预计2026年2月）
- 完整的前端实现
- 类型系统
- 符号表管理

### 📋 M3: MLIR-AIR（预计2026年4月）
- Dialect实现
- 优化Pass
- 编译时执行基础

### 📋 M4: LLVM后端（预计2026年6月）
- 代码生成
- 多平台支持
- 优化集成

### 📋 M5: 工具链（预计2026年9月）
- lld集成
- lldb集成
- chim实现

### 📋 M6: v1.0.0（预计2027年初）
- 生产就绪
- 完整文档
- 稳定API

## 影响和意义

### 技术创新
1. **C3风格错误处理在系统语言中的应用**
2. **MLIR在新语言中的实践**
3. **现代包管理器设计**

### 教育价值
1. **编译器设计教学案例**
2. **LLVM/MLIR学习资源**
3. **系统编程语言设计参考**

### 实用价值
1. **系统编程的新选择**
2. **操作系统开发潜力**
3. **嵌入式系统应用**

## 挑战和风险

### 技术挑战
1. MLIR学习曲线陡峭
2. 性能优化难度大
3. 内存管理设计复杂

### 资源挑战
1. 开发人力有限
2. 时间投入巨大
3. 生态建设需要时间

### 应对策略
1. 渐进式开发
2. 社区协作
3. 参考成功案例

## 未来展望

### 短期目标
- 完成C++前端
- 实现MLIR-AIR
- 发布v0.4.0

### 中期目标
- 完整的编译器
- 基础工具链
- 小规模应用

### 长期愿景
- 生产级语言
- 活跃的社区
- 丰富的生态

## 致谢

感谢以下项目的启发：
- **C3语言** - 错误处理方式
- **Zig语言** - 编译时执行理念
- **LLVM项目** - 编译器基础设施
- **MLIR项目** - 多级IR框架
- **pnpm** - 包管理器设计

## 许可证

本项目采用木兰宽松许可证2.0（Mulan Permissive License，Version 2）。

## 联系方式

- **GitHub**: https://github.com/JuSanSuiYuan/az
- **Issues**: https://github.com/JuSanSuiYuan/az/issues
- **Discussions**: https://github.com/JuSanSuiYuan/az/discussions

---

**项目创建**: 2025年10月29日  
**最后更新**: 2025年10月29日  
**文档版本**: 1.0

---

*AZ编程语言 - 现代、安全、高效的系统编程语言*
