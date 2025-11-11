# AZ语言 - Match语句设计文档

## 概述

Match语句是AZ语言中强大的模式匹配功能，类似于其他现代语言（如Rust、Swift）中的match/switch表达式。它提供了一种清晰、安全的方式来处理多个条件分支。

## 语法

### 基本语法

```az
match <表达式> {
    <模式1> => <语句或表达式>,
    <模式2> => <语句或表达式>,
    ...
    _ => <默认语句或表达式>
}
```

### 完整语法

```az
match <表达式> {
    <模式1> if <守卫条件> => {
        // 代码块
    },
    <模式2> | <模式3> => <单行表达式>,
    _ => <默认处理>
}
```

## 模式类型

### 1. 字面量模式

匹配具体的值：

```az
match x {
    0 => println("零"),
    1 => println("一"),
    2 => println("二"),
    _ => println("其他")
}
```

### 2. 标识符模式

绑定变量：

```az
match x {
    n => println("值为: " + n)
}
```

### 3. 通配符模式

匹配任何值（不绑定）：

```az
match x {
    0 => println("零"),
    _ => println("非零")
}
```

### 4. 或模式

匹配多个值：

```az
match day {
    1 | 2 | 3 | 4 | 5 => println("工作日"),
    6 | 7 => println("周末"),
    _ => println("无效")
}
```

## 守卫条件

守卫条件允许在模式后添加额外的条件检查：

```az
match n {
    x if x > 0 => println("正数"),
    x if x < 0 => println("负数"),
    _ => println("零")
}
```

## 分支体

### 单行表达式

```az
match x {
    0 => return "零",
    1 => return "一",
    _ => return "其他"
}
```

### 代码块

```az
match x {
    0 => {
        println("处理零");
        return 0;
    },
    _ => {
        println("处理其他");
        return x * 2;
    }
}
```

## 完整性检查

编译器会检查match语句是否覆盖了所有可能的情况：

```az
// ✅ 正确：有通配符模式
match x {
    0 => println("零"),
    _ => println("其他")
}

// ❌ 错误：缺少默认分支（未来版本会检查）
match x {
    0 => println("零"),
    1 => println("一")
    // 缺少其他情况的处理
}
```

## 使用示例

### 示例1: 简单的值匹配

```az
fn describe_number(n: int) string {
    match n {
        0 => return "零",
        1 => return "一",
        2 => return "二",
        _ => return "其他数字"
    }
}
```

### 示例2: 使用或模式

```az
fn is_weekend(day: int) bool {
    match day {
        6 | 7 => return true,
        _ => return false
    }
}
```

### 示例3: 使用守卫条件

```az
fn classify_number(n: int) string {
    match n {
        0 => return "零",
        n if n > 0 => return "正数",
        n if n < 0 => return "负数",
        _ => return "未知"
    }
}
```

### 示例4: 复杂的状态机

```az
fn process_state(state: int, input: int) int {
    match state {
        0 => {
            match input {
                1 => return 1,
                2 => return 2,
                _ => return 0
            }
        },
        1 => {
            match input {
                1 => return 2,
                _ => return 0
            }
        },
        2 => return 0,
        _ => return 0
    }
}
```

### 示例5: 成绩评定

```az
fn get_grade(score: int) string {
    match score {
        90 | 91 | 92 | 93 | 94 | 95 | 96 | 97 | 98 | 99 | 100 => return "A",
        80 | 81 | 82 | 83 | 84 | 85 | 86 | 87 | 88 | 89 => return "B",
        70 | 71 | 72 | 73 | 74 | 75 | 76 | 77 | 78 | 79 => return "C",
        60 | 61 | 62 | 63 | 64 | 65 | 66 | 67 | 68 | 69 => return "D",
        _ => return "F"
    }
}
```

## 与其他语言的对比

### Rust

```rust
match x {
    0 => println!("zero"),
    1 | 2 => println!("one or two"),
    n if n > 10 => println!("big"),
    _ => println!("other")
}
```

### Swift

```swift
switch x {
case 0:
    print("zero")
case 1, 2:
    print("one or two")
case let n where n > 10:
    print("big")
default:
    print("other")
}
```

### AZ

```az
match x {
    0 => println("zero"),
    1 | 2 => println("one or two"),
    n if n > 10 => println("big"),
    _ => println("other")
}
```

## 实现细节

### AST结构

```az
// 模式类型
enum PatternKind {
    Literal,      // 字面量模式
    Identifier,   // 标识符模式
    Wildcard,     // 通配符模式
    Or            // 或模式
}

// 模式节点
struct Pattern {
    kind: PatternKind,
    literal: *Expr,        // 字面量模式
    name: string,          // 标识符模式
    patterns: []*Pattern   // 或模式
}

// Match分支
struct MatchArm {
    pattern: *Pattern,
    guard: *Expr,      // 可选的守卫条件
    body: *Stmt
}

// Match语句
struct MatchStmt {
    kind: StmtKind.Match,
    match_expr: *Expr,
    match_arms: []MatchArm
}
```

### 解析流程

1. **解析match关键字**
2. **解析被匹配的表达式**
3. **解析左花括号 {**
4. **循环解析match分支**：
   - 解析模式（支持或模式）
   - 解析可选的守卫条件（if）
   - 解析箭头 =>
   - 解析分支体（表达式或代码块）
5. **解析右花括号 }**

### 语义分析

1. **类型检查**：
   - 检查被匹配表达式的类型
   - 检查每个模式的类型是否与表达式类型兼容
   - 检查守卫条件的类型是否为bool

2. **完整性检查**：
   - 检查是否覆盖了所有可能的值
   - 检查是否有不可达的分支

3. **作用域管理**：
   - 为每个分支创建新的作用域
   - 处理模式中绑定的变量

### 代码生成

Match语句会被降级为一系列的if-else语句：

```az
// 源代码
match x {
    0 => println("zero"),
    1 | 2 => println("one or two"),
    _ => println("other")
}

// 降级后
if (x == 0) {
    println("zero");
} else if (x == 1 || x == 2) {
    println("one or two");
} else {
    println("other");
}
```

## 未来扩展

### 1. 结构体模式

```az
struct Point { x: int, y: int }

match point {
    Point { x: 0, y: 0 } => println("原点"),
    Point { x: 0, y } => println("Y轴上"),
    Point { x, y: 0 } => println("X轴上"),
    Point { x, y } => println("一般点")
}
```

### 2. 元组模式

```az
match (x, y) {
    (0, 0) => println("原点"),
    (0, _) => println("Y轴"),
    (_, 0) => println("X轴"),
    (x, y) => println("一般点")
}
```

### 3. 枚举模式

```az
enum Option<T> {
    Some(T),
    None
}

match opt {
    Some(value) => println("有值: " + value),
    None => println("无值")
}
```

### 4. 范围模式

```az
match score {
    0..60 => println("不及格"),
    60..70 => println("及格"),
    70..80 => println("中等"),
    80..90 => println("良好"),
    90..100 => println("优秀"),
    _ => println("无效分数")
}
```

### 5. 数组模式

```az
match arr {
    [] => println("空数组"),
    [x] => println("单元素"),
    [x, y] => println("两个元素"),
    [first, ..rest] => println("多个元素")
}
```

## 最佳实践

### 1. 总是使用通配符模式

确保match语句覆盖所有情况：

```az
// ✅ 好
match x {
    0 => handle_zero(),
    _ => handle_other()
}

// ❌ 不好（可能遗漏情况）
match x {
    0 => handle_zero()
}
```

### 2. 优先使用或模式

而不是重复代码：

```az
// ✅ 好
match day {
    1 | 2 | 3 | 4 | 5 => "工作日",
    6 | 7 => "周末",
    _ => "无效"
}

// ❌ 不好
match day {
    1 => "工作日",
    2 => "工作日",
    3 => "工作日",
    4 => "工作日",
    5 => "工作日",
    6 => "周末",
    7 => "周末",
    _ => "无效"
}
```

### 3. 合理使用守卫条件

守卫条件应该简洁明了：

```az
// ✅ 好
match n {
    x if x > 0 => "正数",
    x if x < 0 => "负数",
    _ => "零"
}

// ❌ 不好（守卫条件过于复杂）
match n {
    x if x > 0 && x < 100 && x % 2 == 0 => "小偶数",
    // ...
}
```

### 4. 按照可能性排序

将最常见的情况放在前面：

```az
// ✅ 好（假设大多数情况是正数）
match n {
    x if x > 0 => "正数",
    0 => "零",
    _ => "负数"
}
```

## 性能考虑

1. **编译时优化**：
   - 编译器会将match语句优化为跳转表或二分查找
   - 对于连续的整数值，使用跳转表
   - 对于稀疏的值，使用二分查找或哈希表

2. **运行时性能**：
   - 简单的match语句性能接近switch语句
   - 复杂的守卫条件可能影响性能
   - 嵌套match会增加分支预测失败的可能性

## 总结

Match语句是AZ语言中强大而灵活的控制流结构，它提供了：

- ✅ 清晰的语法
- ✅ 类型安全
- ✅ 完整性检查
- ✅ 灵活的模式匹配
- ✅ 守卫条件支持
- ✅ 良好的性能

通过合理使用match语句，可以编写出更加清晰、安全、高效的代码。
