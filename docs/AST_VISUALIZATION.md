# AZ语言AST可视化指南

## 简介

AZ编译器支持将抽象语法树(AST)可视化为图形表示，帮助开发者理解代码结构和调试逻辑错误。本指南将介绍如何使用可视化功能。

## 可视化类型

1. **AST可视化** - 显示程序的抽象语法树结构
2. **控制流图(CFG)** - 显示函数内的控制流结构
3. **函数调用图** - 显示程序中函数之间的调用关系

## 使用方法

### 方法1: 使用Python可视化工具（推荐）

AZ提供了一个Python脚本用于AST可视化，无需安装额外的编译器工具。

```bash
# 基本用法
python tools/ast_visualizer.py examples/visualize_test.az

# 指定输出文件
python tools/ast_visualizer.py examples/visualize_test.az -o my_ast.dot

# 同时生成PNG图像
python tools/ast_visualizer.py examples/visualize_test.az --png
```

### 方法2: 使用C++可视化工具

如果成功构建了C++工具：

```bash
# 可视化AST
./build/tools/visualize_ast examples/visualize_test.az

# 指定输出文件
./build/tools/visualize_ast examples/visualize_test.az -o my_ast.dot

# 生成控制流图
./build/tools/visualize_ast examples/visualize_test.az --cfg

# 生成函数调用图
./build/tools/visualize_ast examples/visualize_test.az --call-graph
```

## 安装Graphviz

要查看生成的图形，需要安装Graphviz工具：

1. 访问 [Graphviz官网](https://graphviz.org/download/)
2. 下载适用于您操作系统的安装包
3. 安装Graphviz
4. 将Graphviz的bin目录添加到系统PATH环境变量中

## 生成图像文件

使用Graphviz命令行工具将DOT文件转换为图像：

```bash
# 生成PNG图像
dot -Tpng ast.dot -o ast.png

# 生成SVG图像
dot -Tsvg ast.dot -o ast.svg

# 生成PDF文档
dot -Tpdf ast.dot -o ast.pdf
```

## 示例

给定以下AZ代码：

```az
fn factorial(n: int) int {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}

fn main() int {
    let x = 5;
    let result = factorial(x);
    println("Factorial of " + x + " is " + result);
    return 0;
}
```

生成的AST图将显示：
- 函数声明节点
- 变量声明节点
- 控制流节点（if语句）
- 表达式节点（二元运算、函数调用等）

## 故障排除

### 1. "dot命令未找到"错误

确保已安装Graphviz并将其添加到PATH环境变量中。

### 2. 中文显示问题

如果图形中的中文显示为乱码，请确保：
1. DOT文件使用UTF-8编码
2. Graphviz支持中文字体

可以在DOT文件头部添加字体设置：
```
digraph AST {
  node [fontname="SimHei"];
  edge [fontname="SimHei"];
  // ... 其他内容
}
```

## 高级用法

### 自定义可视化样式

可以通过修改可视化工具的代码来自定义图形样式：

1. 节点形状和颜色
2. 边的样式和标签
3. 图的布局方向

### 集成到开发环境

可以将可视化功能集成到IDE中：
1. 配置外部工具
2. 设置快捷键
3. 自动显示生成的图像

## 技术细节

### AST节点类型

可视化工具支持以下AST节点类型：

- **表达式节点**：整数、浮点数、字符串、布尔值、标识符、二元运算、一元运算、函数调用
- **语句节点**：变量声明、函数声明、返回语句、if语句、while语句、代码块

### 图形表示约定

- 矩形节点表示AST节点
- 有向边表示父子关系
- 边上的标签表示关系类型（如"condition"、"body"等）

## 贡献

如果您希望改进可视化功能：

1. Fork项目
2. 修改相关代码
3. 提交Pull Request

可以改进的方面：
- 支持更多AST节点类型
- 改进图形布局算法
- 添加交互式可视化功能