# az_mod - AZ语言包管理器

## 概述

az_mod是AZ语言的官方包管理器，设计灵感来自pnpm，提供高效的依赖管理和workspace支持。

## 核心特性

### 1. pnpm风格的Workspace
- 单一仓库多包管理（monorepo）
- 包之间的依赖链接
- 统一的依赖管理

### 2. 硬链接优势
- 节省磁盘空间
- 加快安装速度
- 全局缓存共享

### 3. Git直连
- 直接从Git仓库获取依赖
- 支持tag、branch、commit
- 无需中心仓库

### 4. 快速并行
- 并行下载依赖
- 并行构建包
- 增量构建

## 架构设计

```
chim/
├── src/
│   ├── main.rs              # 主入口
│   ├── cli/                 # 命令行接口
│   │   ├── mod.rs
│   │   ├── init.rs          # chim init
│   │   ├── add.rs           # chim add
│   │   ├── install.rs       # chim install
│   │   ├── update.rs        # chim update
│   │   ├── build.rs         # chim build
│   │   └── publish.rs       # chim publish
│   ├── resolver/            # 依赖解析
│   │   ├── mod.rs
│   │   ├── graph.rs         # 依赖图
│   │   └── version.rs       # 版本解析
│   ├── fetcher/             # 依赖获取
│   │   ├── mod.rs
│   │   ├── git.rs           # Git直连
│   │   ├── http.rs          # HTTP下载
│   │   └── cache.rs         # 缓存管理
│   ├── linker/              # 硬链接管理
│   │   ├── mod.rs
│   │   └── store.rs         # 全局存储
│   ├── workspace/           # Workspace支持
│   │   ├── mod.rs
│   │   ├── config.rs        # 配置解析
│   │   └── graph.rs         # Workspace图
│   ├── builder/             # 构建系统
│   │   ├── mod.rs
│   │   ├── compiler.rs      # 调用az编译器
│   │   └── parallel.rs      # 并行构建
│   └── error.rs             # C3风格错误处理
├── Cargo.toml
└── README.md
```

## 命令

### 1. az_mod init
创建新项目

```bash
# 创建新项目
az_mod init my-project

# 创建库项目
az_mod init my-lib --lib

# 创建workspace
az_mod init my-workspace --workspace
```

生成的项目结构：
```
my-project/
├── package.az           # 包配置
├── az_mod-workspace.toml  # Workspace配置（可选）
├── src/
│   └── main.az
└── .gitignore
```

### az_mod add
添加依赖

```bash
# 添加依赖
az_mod add std@1.0.0

# 从Git添加
az_mod add http --git https://gitee.com/az_lang/http --tag v0.2.0

# 添加开发依赖
az_mod add test-framework --dev

# 添加本地依赖
az_mod add my-lib --path ../my-lib
```

### az_mod install
安装依赖

```bash
# 安装所有依赖
az_mod install

# 仅安装生产依赖
az_mod install --prod

# 冻结依赖版本
az_mod install --frozen
```

### az_mod update
更新依赖

```bash
# 更新所有依赖
az_mod update

# 更新特定依赖
az_mod update std

# 更新到最新主版本
az_mod update --latest
```

### az_mod build
构建项目

```bash
# 构建项目
az_mod build

# 发布构建
az_mod build --release

# 指定目标
az_mod build --target x86_64-linux-gnu

# 并行构建
az_mod build --jobs 8
```

### az_mod run
运行项目

```bash
# 运行主程序
az_mod run

# 运行特定二进制
az_mod run --bin my-tool

# 传递参数
az_mod run -- arg1 arg2
```

### az_mod test
运行测试

```bash
# 运行所有测试
az_mod test

# 运行特定测试
az_mod test test_name

# 并行测试
az_mod test --jobs 8
```

### az_mod publish
发布包

```bash
# 发布到仓库
az_mod publish

# 干运行
az_mod publish --dry-run
```

## package.az配置

```toml
# 包信息
name = "my-project"
version = "0.1.0"
description = "My AZ project"
authors = ["Your Name <you@example.com>"]
license = "MulanPSL-2.0"
repository = "https://gitee.com/username/my-project"

# 主文件
main = "src/main.az"

# 脚本
[scripts]
build = "az build"
test = "az test"
run = "az run"
fmt = "az fmt"

# 依赖
[dependencies]
std = "1.0.0"
http = { git = "https://gitee.com/az_lang/http", tag = "v0.2.0" }
json = { path = "../json" }

# 开发依赖
[dev-dependencies]
test-framework = "0.1.0"
benchmark = "0.2.0"

# Workspace配置
[workspace]
members = [
    "packages/*",
    "apps/*"
]

# 构建配置
[build]
target = "x86_64-unknown-linux-gnu"
opt-level = 2
lto = true
debug = false

# 编译器标志
[build.flags]
warnings = "all"
errors = ["unused-variable", "type-mismatch"]

# 特性标志
[features]
default = ["std"]
full = ["std", "http", "json"]
minimal = []

# 目标特定配置
[target.x86_64-linux-gnu]
linker = "lld"
rustflags = ["-C", "target-cpu=native"]

[target.aarch64-linux-gnu]
linker = "lld"
```

## az_mod-workspace.toml配置

```toml
# Workspace配置
packages = [
    'packages/*',
    'apps/*'
]

# 链接选项
link-workspace-packages = true

# az_mod特定设置
[az_mod]
# Git直连配置
git-fetch = true

# 包管理策略
hard-link = true
hoist = false
shamefully-hoist = false

# 工具链配置
linker = "lld"
debugger = "lldb"

# 缓存配置
[cache]
dir = "~/.chim/cache"
max-size = "10GB"
clean-after = "30d"

# 并行配置
[parallel]
download = 8
build = 4

# 代理配置
[proxy]
http = "http://proxy.example.com:8080"
https = "https://proxy.example.com:8080"
```

## 硬链接机制

### 全局存储结构

```
~/.az_mod/
├── store/                    # 全局包存储
│   ├── std@1.0.0/
│   │   ├── src/
│   │   └── package.az
│   ├── http@0.2.0/
│   │   ├── src/
│   │   └── package.az
│   └── ...
├── cache/                    # 下载缓存
│   ├── git/
│   └── http/
└── config.toml              # 全局配置
```

### 项目node_modules结构

```
my-project/
├── node_modules/            # 硬链接到全局存储
│   ├── std -> ~/.az_mod/store/std@1.0.0/
│   ├── http -> ~/.az_mod/store/http@0.2.0/
│   └── .az_mod/              # az_mod元数据
│       ├── lock.toml       # 锁文件
│       └── graph.json      # 依赖图
└── ...
```

### 硬链接优势

1. **节省空间**：相同版本的包只存储一次
2. **快速安装**：创建硬链接比复制文件快得多
3. **一致性**：所有项目使用相同的包副本
4. **安全性**：包内容不可修改（写时复制）

## Git直连机制

### 支持的Git URL格式

```toml
# HTTPS
http = { git = "https://gitee.com/az_lang/http" }

# SSH
http = { git = "git@gitee.com:az_lang/http.git" }

# 指定tag
http = { git = "https://gitee.com/az_lang/http", tag = "v0.2.0" }

# 指定branch
http = { git = "https://gitee.com/az_lang/http", branch = "main" }

# 指定commit
http = { git = "https://gitee.com/az_lang/http", rev = "abc123" }
```

### Git缓存

```
~/.az_mod/cache/git/
├── gitee.com/
│   └── az_lang/
│       └── http/
│           ├── .git/
│           └── refs/
│               ├── v0.1.0/
│               ├── v0.2.0/
│               └── main/
```

## 依赖解析算法

### 版本解析

使用语义化版本（SemVer）：

```
^1.2.3  := >=1.2.3 <2.0.0
~1.2.3  := >=1.2.3 <1.3.0
1.2.*   := >=1.2.0 <1.3.0
>=1.2.3 := >=1.2.3
```

### 冲突解决

1. **优先级**：直接依赖 > 间接依赖
2. **版本选择**：选择满足所有约束的最新版本
3. **冲突报告**：无法解决时报告详细错误

### 依赖图

```rust
struct DependencyGraph {
    nodes: HashMap<PackageId, Node>,
    edges: Vec<Edge>,
}

struct Node {
    id: PackageId,
    version: Version,
    dependencies: Vec<PackageId>,
}

struct Edge {
    from: PackageId,
    to: PackageId,
    constraint: VersionReq,
}
```

## 并行构建

### 构建图

```rust
struct BuildGraph {
    packages: Vec<Package>,
    dependencies: HashMap<PackageId, Vec<PackageId>>,
}

impl BuildGraph {
    fn topological_sort(&self) -> Vec<Vec<PackageId>> {
        // 返回可并行构建的包组
    }
}
```

### 并行策略

1. **拓扑排序**：确定构建顺序
2. **分层构建**：同一层的包可并行构建
3. **资源限制**：根据CPU核心数限制并行度

## 性能优化

### 1. 增量构建
- 跟踪文件修改时间
- 只重新构建修改的包
- 缓存编译结果

### 2. 并行下载
- 同时下载多个依赖
- 使用HTTP/2多路复用
- 断点续传支持

### 3. 缓存策略
- 全局包缓存
- 编译结果缓存
- Git仓库缓存

### 4. 硬链接
- 零拷贝安装
- 节省磁盘空间
- 提高IO性能

## C3风格错误处理

```rust
pub enum AzModError {
    ResolveError(String),
    FetchError(String),
    BuildError(String),
    LinkError(String),
}

pub type Result<T> = std::result::Result<T, AzModError>;

// 使用示例
fn install_package(name: &str) -> Result<()> {
    let package = fetch_package(name)?;
    let deps = resolve_dependencies(&package)?;
    build_package(&package)?;
    link_package(&package)?;
    Ok(())
}
```

## 与其他包管理器对比

| 特性 | az_mod | npm | pnpm | cargo |
|------|-------|-----|------|-------|
| 硬链接 | ✅ | ❌ | ✅ | ❌ |
| Workspace | ✅ | ✅ | ✅ | ✅ |
| Git直连 | ✅ | ✅ | ✅ | ✅ |
| 并行构建 | ✅ | ❌ | ✅ | ✅ |
| 增量构建 | ✅ | ❌ | ❌ | ✅ |
| 速度 | 快 | 慢 | 快 | 快 |

## 未来计划

- [ ] 中心仓库支持
- [ ] 包签名验证
- [ ] 自动依赖更新
- [ ] 漏洞扫描
- [ ] 许可证检查
- [ ] 包统计分析

## 许可证

本项目采用木兰宽松许可证2.0（Mulan Permissive License，Version 2）。
