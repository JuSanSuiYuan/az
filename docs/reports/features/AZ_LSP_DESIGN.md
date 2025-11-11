# AZ lsp - AZ Language Server Protocol

**AZ语言的官方LSP服务器**

---

## 🎯 概述

AZ lsp是AZ编程语言的Language Server Protocol实现，为IDE和编辑器提供智能代码补全、诊断、跳转等功能。

### 核心特性

- ✅ **代码补全** - 智能的上下文感知补全
- ✅ **语法诊断** - 实时错误和警告
- ✅ **跳转定义** - 快速导航到定义
- ✅ **查找引用** - 查找符号的所有使用
- ✅ **悬停提示** - 显示类型和文档
- ✅ **代码格式化** - 自动格式化代码
- ✅ **重命名** - 安全的符号重命名
- ✅ **代码操作** - 快速修复和重构

---

## 🏗️ 架构设计

### 技术栈

```
AZ lsp
├── 语言: Rust
├── LSP库: tower-lsp
├── 解析器: AZ Parser (Rust port)
├── 类型检查: AZ Type Checker
└── 协议: LSP 3.17
```

### 组件结构

```
tools/az_lsp/
├── src/
│   ├── main.rs              # 入口点
│   ├── server.rs            # LSP服务器实现
│   ├── parser/              # 解析器
│   │   ├── mod.rs
│   │   ├── lexer.rs         # 词法分析
│   │   ├── parser.rs        # 语法分析
│   │   └── ast.rs           # AST定义
│   ├── analyzer/            # 分析器
│   │   ├── mod.rs
│   │   ├── semantic.rs      # 语义分析
│   │   ├── type_checker.rs  # 类型检查
│   │   └── symbol_table.rs  # 符号表
│   ├── features/            # LSP功能
│   │   ├── mod.rs
│   │   ├── completion.rs    # 代码补全
│   │   ├── diagnostics.rs   # 诊断
│   │   ├── goto.rs          # 跳转定义
│   │   ├── hover.rs         # 悬停提示
│   │   ├── references.rs    # 查找引用
│   │   ├── rename.rs        # 重命名
│   │   └── formatting.rs    # 格式化
│   └── utils/               # 工具函数
│       ├── mod.rs
│       └── position.rs      # 位置转换
├── tests/                   # 测试
├── Cargo.toml              # Rust配置
└── README.md
```

---

## 📚 功能详解

### 1. 代码补全 (Completion)

#### 触发场景

```az
// 场景1: 模块导入
import std.|  // 触发：显示std的所有子模块

// 场景2: 函数调用
println(|)    // 触发：显示参数提示

// 场景3: 成员访问
point.|       // 触发：显示Point的所有字段和方法

// 场景4: 关键字
f|            // 触发：显示fn, for, float等
```

#### 补全类型

```rust
pub enum CompletionItemKind {
    Keyword,        // 关键字: fn, let, var
    Function,       // 函数: add, println
    Variable,       // 变量: x, count
    Struct,         // 结构体: Point, Vec3
    Field,          // 字段: x, y, z
    Module,         // 模块: std, math
    Enum,           // 枚举: Color, Option
    EnumVariant,    // 枚举变体: Some, None
}
```

#### 实现示例

```rust
async fn completion(
    &self,
    params: CompletionParams,
) -> Result<Option<CompletionResponse>> {
    let uri = params.text_document_position.text_document.uri;
    let position = params.text_document_position.position;
    
    // 获取文档
    let document = self.documents.get(&uri)?;
    
    // 解析到当前位置
    let ast = self.parser.parse_to_position(&document, position)?;
    
    // 分析上下文
    let context = self.analyzer.analyze_context(&ast, position)?;
    
    // 生成补全项
    let items = match context.kind {
        ContextKind::Import => self.complete_imports(&context),
        ContextKind::Member => self.complete_members(&context),
        ContextKind::Type => self.complete_types(&context),
        _ => self.complete_general(&context),
    };
    
    Ok(Some(CompletionResponse::Array(items)))
}
```

### 2. 语法诊断 (Diagnostics)

#### 诊断类型

```rust
pub enum DiagnosticSeverity {
    Error,      // 错误：语法错误、类型错误
    Warning,    // 警告：未使用的变量、废弃的API
    Info,       // 信息：代码风格建议
    Hint,       // 提示：优化建议
}
```

#### 诊断示例

```az
// 错误：类型不匹配
let x: int = "hello";  // Error: Type mismatch

// 警告：未使用的变量
let unused = 10;       // Warning: Unused variable 'unused'

// 提示：可以使用let
var constant = 42;     // Hint: Consider using 'let' for immutable

// 信息：可以简化
if (x == true) { }     // Info: Can be simplified to 'if (x)'
```

#### 实现示例

```rust
async fn diagnostics(&self, uri: &Url) -> Result<Vec<Diagnostic>> {
    let document = self.documents.get(uri)?;
    let mut diagnostics = Vec::new();
    
    // 词法分析错误
    let tokens = match self.lexer.tokenize(&document.text) {
        Ok(tokens) => tokens,
        Err(errors) => {
            diagnostics.extend(errors.into_iter().map(|e| e.to_diagnostic()));
            return Ok(diagnostics);
        }
    };
    
    // 语法分析错误
    let ast = match self.parser.parse(tokens) {
        Ok(ast) => ast,
        Err(errors) => {
            diagnostics.extend(errors.into_iter().map(|e| e.to_diagnostic()));
            return Ok(diagnostics);
        }
    };
    
    // 语义分析错误
    let semantic_errors = self.analyzer.analyze(&ast);
    diagnostics.extend(semantic_errors.into_iter().map(|e| e.to_diagnostic()));
    
    // 类型检查错误
    let type_errors = self.type_checker.check(&ast);
    diagnostics.extend(type_errors.into_iter().map(|e| e.to_diagnostic()));
    
    Ok(diagnostics)
}
```

### 3. 跳转定义 (Go to Definition)

#### 支持的跳转

```az
// 跳转到函数定义
let result = add(3, 5);  // Ctrl+Click on 'add' -> 跳转到函数定义

// 跳转到变量定义
println(x);              // Ctrl+Click on 'x' -> 跳转到变量声明

// 跳转到类型定义
let p: Point = ...;      // Ctrl+Click on 'Point' -> 跳转到struct定义

// 跳转到模块
import math.vector;      // Ctrl+Click on 'vector' -> 跳转到模块文件
```

#### 实现示例

```rust
async fn goto_definition(
    &self,
    params: GotoDefinitionParams,
) -> Result<Option<GotoDefinitionResponse>> {
    let uri = params.text_document_position_params.text_document.uri;
    let position = params.text_document_position_params.position;
    
    // 获取符号
    let symbol = self.get_symbol_at_position(&uri, position)?;
    
    // 查找定义
    let definition = self.symbol_table.find_definition(&symbol)?;
    
    // 返回位置
    Ok(Some(GotoDefinitionResponse::Scalar(Location {
        uri: definition.uri,
        range: definition.range,
    })))
}
```

### 4. 悬停提示 (Hover)

#### 显示内容

```az
// 悬停在函数上
fn add(a: int, b: int) int { ... }
// 显示：
// fn add(a: int, b: int) -> int
// 将两个整数相加

// 悬停在变量上
let x = 10;
// 显示：
// let x: int = 10

// 悬停在类型上
struct Point { x: int, y: int }
// 显示：
// struct Point {
//     pub x: int,
//     pub y: int
// }
```

#### 实现示例

```rust
async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
    let uri = params.text_document_position_params.text_document.uri;
    let position = params.text_document_position_params.position;
    
    // 获取符号
    let symbol = self.get_symbol_at_position(&uri, position)?;
    
    // 获取类型信息
    let type_info = self.type_checker.get_type(&symbol)?;
    
    // 获取文档
    let doc = self.get_documentation(&symbol);
    
    // 构建悬停内容
    let contents = format!(
        "```az\n{}\n```\n\n{}",
        type_info.signature(),
        doc.unwrap_or_default()
    );
    
    Ok(Some(Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value: contents,
        }),
        range: Some(symbol.range),
    }))
}
```

### 5. 查找引用 (Find References)

#### 查找范围

```az
// 定义
fn add(a: int, b: int) int { ... }

// 引用1
let x = add(1, 2);

// 引用2
let y = add(3, 4);

// 引用3
println(add(5, 6));
```

#### 实现示例

```rust
async fn references(
    &self,
    params: ReferenceParams,
) -> Result<Option<Vec<Location>>> {
    let uri = params.text_document_position.text_document.uri;
    let position = params.text_document_position.position;
    
    // 获取符号
    let symbol = self.get_symbol_at_position(&uri, position)?;
    
    // 查找所有引用
    let references = self.symbol_table.find_references(&symbol)?;
    
    // 转换为Location
    let locations = references
        .into_iter()
        .map(|r| Location {
            uri: r.uri,
            range: r.range,
        })
        .collect();
    
    Ok(Some(locations))
}
```

### 6. 重命名 (Rename)

#### 重命名范围

```az
// 重命名函数
fn old_name() { }  // 重命名为 new_name
old_name();        // 自动更新为 new_name()

// 重命名变量
let old_var = 10;  // 重命名为 new_var
println(old_var);  // 自动更新为 new_var
```

#### 实现示例

```rust
async fn rename(
    &self,
    params: RenameParams,
) -> Result<Option<WorkspaceEdit>> {
    let uri = params.text_document_position.text_document.uri;
    let position = params.text_document_position.position;
    let new_name = params.new_name;
    
    // 获取符号
    let symbol = self.get_symbol_at_position(&uri, position)?;
    
    // 查找所有引用
    let references = self.symbol_table.find_references(&symbol)?;
    
    // 构建编辑
    let mut changes = HashMap::new();
    for reference in references {
        let edits = changes.entry(reference.uri).or_insert_with(Vec::new);
        edits.push(TextEdit {
            range: reference.range,
            new_text: new_name.clone(),
        });
    }
    
    Ok(Some(WorkspaceEdit {
        changes: Some(changes),
        ..Default::default()
    }))
}
```

### 7. 代码格式化 (Formatting)

#### 格式化规则

```az
// 格式化前
fn add(a:int,b:int)int{return a+b;}

// 格式化后
fn add(a: int, b: int) int {
    return a + b;
}
```

#### 实现示例

```rust
async fn formatting(
    &self,
    params: DocumentFormattingParams,
) -> Result<Option<Vec<TextEdit>>> {
    let uri = params.text_document.uri;
    let document = self.documents.get(&uri)?;
    
    // 解析文档
    let ast = self.parser.parse(&document.text)?;
    
    // 格式化AST
    let formatted = self.formatter.format(&ast)?;
    
    // 创建编辑
    let edit = TextEdit {
        range: Range {
            start: Position { line: 0, character: 0 },
            end: Position {
                line: document.line_count() as u32,
                character: 0,
            },
        },
        new_text: formatted,
    };
    
    Ok(Some(vec![edit]))
}
```

---

## 🚀 实现计划

### Phase 1: 基础框架 (1周)

```rust
// 1. 创建项目
mkdir -p tools/az_lsp
cargo new tools/az_lsp --bin

// 2. 添加依赖
[dependencies]
tower-lsp = "0.20"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"

// 3. 实现基础服务器
pub struct AzLspServer {
    client: Client,
    documents: DashMap<Url, Document>,
}

#[tower_lsp::async_trait]
impl LanguageServer for AzLspServer {
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                completion_provider: Some(CompletionOptions::default()),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                // ... 其他功能
                ..Default::default()
            },
            ..Default::default()
        })
    }
    
    async fn initialized(&self, _: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "AZ lsp initialized!")
            .await;
    }
    
    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }
}
```

### Phase 2: 解析器 (1周)

```rust
// 1. 词法分析器
pub struct Lexer {
    source: String,
    position: usize,
}

impl Lexer {
    pub fn tokenize(&mut self) -> Result<Vec<Token>> {
        // 实现词法分析
    }
}

// 2. 语法分析器
pub struct Parser {
    tokens: Vec<Token>,
    current: usize,
}

impl Parser {
    pub fn parse(&mut self) -> Result<Program> {
        // 实现语法分析
    }
}
```

### Phase 3: 分析器 (1周)

```rust
// 1. 符号表
pub struct SymbolTable {
    scopes: Vec<Scope>,
    symbols: HashMap<String, Symbol>,
}

// 2. 类型检查器
pub struct TypeChecker {
    symbol_table: SymbolTable,
}

impl TypeChecker {
    pub fn check(&mut self, ast: &Program) -> Vec<TypeError> {
        // 实现类型检查
    }
}
```

### Phase 4: LSP功能 (2周)

```rust
// 实现所有LSP功能
impl LanguageServer for AzLspServer {
    async fn completion(&self, params: CompletionParams) -> Result<...> { }
    async fn hover(&self, params: HoverParams) -> Result<...> { }
    async fn goto_definition(&self, params: GotoDefinitionParams) -> Result<...> { }
    async fn references(&self, params: ReferenceParams) -> Result<...> { }
    async fn rename(&self, params: RenameParams) -> Result<...> { }
    async fn formatting(&self, params: DocumentFormattingParams) -> Result<...> { }
}
```

### Phase 5: 测试和优化 (1周)

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_completion() { }
    
    #[test]
    fn test_diagnostics() { }
    
    #[test]
    fn test_goto_definition() { }
}
```

---

## 📦 安装和使用

### 安装

```bash
# 从源码构建
git clone https://github.com/JuSanSuiYuan/az.git
cd az/tools/az_lsp
cargo build --release

# 安装到系统
cargo install --path .
```

### VSCode集成

```json
// .vscode/settings.json
{
    "az_lsp.server.path": "/path/to/az_lsp",
    "az_lsp.trace.server": "verbose"
}
```

### 配置

```toml
# az_lsp.toml
[server]
max_diagnostics = 100
completion_trigger_characters = [".", ":", ">"]

[formatting]
indent_size = 4
max_line_length = 100
```

---

## 🎯 功能对比

### 与其他LSP对比

| 功能 | rust-analyzer | clangd | AZ lsp |
|------|---------------|--------|-------|
| 代码补全 | ✅ 优秀 | ✅ 优秀 | ✅ 计划 |
| 诊断 | ✅ 优秀 | ✅ 优秀 | ✅ 计划 |
| 跳转定义 | ✅ 优秀 | ✅ 优秀 | ✅ 计划 |
| 查找引用 | ✅ 优秀 | ✅ 优秀 | ✅ 计划 |
| 重命名 | ✅ 优秀 | ✅ 优秀 | ✅ 计划 |
| 格式化 | ✅ rustfmt | ✅ clang-format | ✅ 计划 |
| 宏展开 | ✅ 支持 | ✅ 支持 | 📋 未来 |
| 内联提示 | ✅ 支持 | ✅ 支持 | 📋 未来 |

---

## 📊 性能目标

### 响应时间

| 操作 | 目标时间 | 说明 |
|------|---------|------|
| 代码补全 | <50ms | 即时响应 |
| 诊断 | <200ms | 实时反馈 |
| 跳转定义 | <100ms | 快速导航 |
| 格式化 | <500ms | 可接受延迟 |

### 内存使用

- **小项目** (<1000行): <50MB
- **中项目** (<10000行): <200MB
- **大项目** (>10000行): <500MB

---

## 🔮 未来计划

### v0.1.0 - 基础功能 (1个月)
- ✅ 基础LSP服务器
- ✅ 代码补全
- ✅ 语法诊断
- ✅ 跳转定义

### v0.2.0 - 增强功能 (2个月)
- ✅ 查找引用
- ✅ 重命名
- ✅ 悬停提示
- ✅ 代码格式化

### v0.3.0 - 高级功能 (3个月)
- ✅ 代码操作
- ✅ 内联提示
- ✅ 语义高亮
- ✅ 调用层次

### v1.0.0 - 生产就绪 (6个月)
- ✅ 完整功能
- ✅ 性能优化
- ✅ 稳定性保证
- ✅ 完整文档

---

## 📝 总结

### AZ lsp特点

✅ **现代化** - 基于Rust和tower-lsp  
✅ **高性能** - 快速响应，低内存占用  
✅ **功能完整** - 支持所有主要LSP功能  
✅ **易于集成** - 支持VSCode、Vim、Emacs等  
✅ **持续更新** - 随AZ语言发展而进化

### 开发状态

```
AZ lsp开发进度: ░░░░░░░░░░░░░░░░░░░░ 0%

📋 设计阶段: 100% ✅
📋 实现阶段: 0%
📋 测试阶段: 0%
📋 发布阶段: 0%

预计完成: 6个月
```

---

**AZ lsp - 为AZ语言提供一流的IDE体验！** 🚀

