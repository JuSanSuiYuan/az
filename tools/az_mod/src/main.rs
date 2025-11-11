// az_mod - AZ语言包管理器
// 使用Rust实现，提供高性能和安全性

use clap::{Parser, Subcommand};
use colored::*;
use std::path::PathBuf;

mod resolver;
mod fetcher;
mod linker;
mod workspace;
mod cache;
mod error;

use error::Result;

#[derive(Parser)]
#[command(name = "az_mod")]
#[command(about = "AZ语言包管理器az_mod", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// 初始化新项目
    Init {
        /// 项目名称
        name: String,
        /// 创建库项目
        #[arg(long)]
        lib: bool,
        /// 创建workspace
        #[arg(long)]
        workspace: bool,
    },
    /// 添加依赖
    Add {
        /// 包名
        package: String,
        /// Git仓库URL
        #[arg(long)]
        git: Option