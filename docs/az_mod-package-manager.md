# az mod包管理器设计

## 核心功能

az mod是AZ语言的包管理器，提供以下核心功能：

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

### 模块结构
```
az_mod/
├── src/
│   ├── main.rs              # 主入口
│   ├── cli/                 # 命令行接口
│   │   ├── mod.rs
│   │   ├── init.rs          # az mod init
│   │   ├── add.rs           # az mod add
│   │   ├── install.rs       # az mod install
│   │   ├── update.rs        # az mod update
│   │   ├── build.rs         # az mod build
│   │   └── publish.rs       # az mod publish
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

## 命令接口

### az mod init
创建新项目

```bash
# 创建新项目
az mod init my-project

# 创建库项目
az mod init my-lib --lib

# 创建workspace
az mod init my-workspace --workspace
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

### az mod add
添加依赖

```bash
# 添加依赖
az mod add std@1.0.0

# 从Git添加
az mod add http --git https://gitee.com/az_lang/http --tag v0.2.0

# 添加开发依赖
az mod add test-framework --dev

# 本地路径依赖
az mod add my-lib --path ../my-lib
```

### az mod install
安装依赖

```bash
# 安装所有依赖
az mod install

# 仅安装生产依赖
az mod install --prod

# 严格按照lock文件安装
az mod install --frozen
```

### az mod update
更新依赖

```bash
# 更新所有依赖
az mod update

# 更新特定依赖
az mod update std

# 更新到最新版本（忽略版本约束）
az mod update --latest
```

### az mod build
构建项目

```bash
# 构建项目
az mod build

# 发布模式构建
az mod build --release

# 指定目标平台
az mod build --target x86_64-linux-gnu

# 并行构建
az mod build --jobs 8
```

### az mod run
运行项目

```bash
# 运行主程序
az mod run

# 运行特定二进制
az mod run --bin my-tool

# 传递参数
az mod run -- arg1 arg2
```

### az mod test
运行测试

```bash
# 运行所有测试
az mod test

# 运行特定测试
az mod test test_name

# 并行测试
az mod test --jobs 8
```

### az mod publish
发布包

```bash
# 发布包
az mod publish

# 预览发布
az mod publish --dry-run
```

## 配置文件

### package.az配置
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

### az_mod-workspace.toml配置
```toml
# 工作区配置

# 指定工作区中的包
packages = [
  "apps/*",
  "packages/*"
]

# 是否链接工作区包
link-workspace-packages = true

# az mod特定设置
[az_mod]
  # Git直连配置
  git-fetch = true
  
  # 缓存目录
  dir = "~/.az_mod/cache"
  
  # 存储策略
  store = "hardlink"
  
  # Node.js兼容模式（可选）
  node-compat = false
```

## 全局存储设计

### 存储结构
```
~/.az_mod/
├── cache/              # 全局缓存
│   ├── registry/       # 注册表缓存
│   ├── git/            # Git仓库缓存
│   └── tarballs/       # 压缩包缓存
├── store/              # 全局存储
│   ├── std@1.0.0/      # 包内容
│   ├── http@0.2.0/     # 包内容
│   └── ...
└── config.toml         # 全局配置
```

### 项目结构
```
my-project/
├── node_modules/       # 符号链接目录
│   ├── std -> ~/.az_mod/store/std@1.0.0/
│   ├── http -> ~/.az_mod/store/http@0.2.0/
│   └── .az_mod/              # az mod元数据
├── package.az          # 包配置
├── src/                # 源代码
└── az_mod-lock.toml    # 锁定文件
```

## 依赖解析

### 版本规范
```
^1.2.3  # 兼容更新 (1.2.3 ≤ 版本 < 2.0.0)
~1.2.3  # 补丁更新 (1.2.3 ≤ 版本 < 1.3.0)
*       # 任意版本
>=1.2.3 <2.0.0  # 范围版本
1.x     # 1系列的任意版本
```

### Git依赖
```toml
[dependencies]
http = { git = "https://gitee.com/az_lang/http", tag = "v0.2.0" }
utils = { git = "https://gitee.com/az_lang/utils", branch = "master" }
logger = { git = "https://gitee.com/az_lang/logger", rev = "abc123" }
```

## 错误处理设计

### 错误类型
```rust
pub enum AzModError {
    // 配置错误
    ConfigError(String),
    
    // 网络错误
    NetworkError(String),
    
    // 依赖错误
    DependencyError(String),
    
    // 构建错误
    BuildError(String),
    
    // IO错误
    IOError(std::io::Error),
    
    // 其他错误
    Other(String),
}

pub type Result<T> = std::result::Result<T, AzModError>;
```

### 错误示例
```
Error: 依赖解析失败
├─ 包: http@0.3.0
├─ 问题: 与已安装的版本 http@0.2.0 冲突
└─ 解决方案: 运行 `az mod why http` 查看依赖树
```

## 与其他包管理器对比

| 特性 | az mod | npm | pnpm | cargo |
|------|--------|-----|------|-------|
| 硬链接存储 | ✅ | ❌ | ✅ | ✅ |
| Workspace支持 | ✅ | ✅ | ✅ | ✅ |
| 版本锁定 | ✅ | ✅ | ✅ | ✅ |
| Git依赖 | ✅ | ⚠️ | ⚠️ | ✅ |
| 并行构建 | ✅ | ❌ | ❌ | ✅ |
| AZ语言集成 | ✅ | ❌ | ❌ | ❌ |
| 磁盘效率 | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 安装速度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |