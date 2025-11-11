# AZ lsp - AZ Language Server Protocol

**Official Language Server for the AZ Programming Language**

---

## 🎯 Overview

AZ lsp is the official Language Server Protocol implementation for AZ, providing intelligent code completion, diagnostics, navigation, and more for IDEs and editors.

## ✨ Features

- ✅ **Code Completion** - Context-aware intelligent completion
- ✅ **Diagnostics** - Real-time syntax and semantic errors
- ✅ **Go to Definition** - Quick navigation to definitions
- ✅ **Find References** - Find all symbol usages
- ✅ **Hover Information** - Type and documentation on hover
- ✅ **Code Formatting** - Automatic code formatting
- ✅ **Rename** - Safe symbol renaming
- ✅ **Code Actions** - Quick fixes and refactorings

## 🚀 Installation

### From Source

```bash
git clone https://github.com/JuSanSuiYuan/az.git
cd az/tools/az_lsp
cargo build --release
cargo install --path .
```

### From Cargo

```bash
cargo install --path tools/az_lsp
```

## 📦 Editor Integration

### VSCode

Install the AZ extension from the marketplace:

```bash
code --install-extension az-lang.az-vscode
```

Or configure manually in `.vscode/settings.json`:

```json
{
    "az_lsp.server.path": "/path/to/az_lsp",
    "az_lsp.trace.server": "verbose"
}
```

### Vim/Neovim

Using `coc.nvim`:

```json
{
    "languageserver": {
        "az": {
            "command": "az_lsp",
            "filetypes": ["az"],
            "rootPatterns": ["package.az"]
        }
    }
}
```

Using `nvim-lspconfig`:

```lua
require'lspconfig'.az_lsp.setup{}
```

### Emacs

Using `lsp-mode`:

```elisp
(add-to-list 'lsp-language-id-configuration '(az-mode . "az"))
(lsp-register-client
 (make-lsp-client :new-connection (lsp-stdio-connection "az_lsp")
                  :major-modes '(az-mode)
                  :server-id 'az_lsp))
```

## ⚙️ Configuration

Create `az_lsp.toml` in your project root:

```toml
[server]
max_diagnostics = 100
completion_trigger_characters = [".", ":", ">"]

[formatting]
indent_size = 4
max_line_length = 100
use_tabs = false

[diagnostics]
enable_warnings = true
enable_hints = true
```

## 🔧 Development

### Prerequisites

- Rust 1.70+
- Cargo

### Build

```bash
cargo build
```

### Test

```bash
cargo test
```

### Run

```bash
cargo run
```

## 📚 Documentation

- [Design Document](../AZ_LSP_DESIGN.md)
- [API Reference](docs/api.md)
- [Contributing Guide](CONTRIBUTING.md)

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) first.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🔗 Links

- [AZ Language](https://github.com/JuSanSuiYuan/az)
- [Documentation](https://az-lang.org/docs)
- [Community](https://discord.gg/az-lang)

---

**AZ lsp - Bringing first-class IDE support to AZ!** 🚀
