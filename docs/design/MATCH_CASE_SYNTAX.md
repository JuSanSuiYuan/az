# AZ语言 Match Case 语法

**类似Python的match case语法**

---

## 🎯 设计理念

AZ的match case语法借鉴了Python 3.10+的模式匹配，提供简洁、直观的语法。

### 与Python对比

| 特性 | Python | AZ |
|------|--------|-----|
| 关键字 | `match`/`case` | `match`/`case` |
| 通配符 | `_` | `_` |
| 守卫 | `if` | `if` |
| 或模式 | `\|` | `,` 或多个case |
| 代码块 | 缩进 | `{}` |

---

## 📚 语法规则

### 基本语法

```az
match <expression> {
    case <pattern>:
        <statement>
    case <pattern>:
        <statement>
    case _:
        <statement>
}
```

### 完整语法

```az
match <expression> {
    case <pattern> [if <condition>]: [{ ]
        <statements>
    [ }]
    case <pattern1>, <pattern2>, ...: [{ ]
        <statements>
    [ }]
    case _: [{ ]
        <statements>
    [ }]
}
```

---

## 🔧 模式类型

### 1. 字面量模式

匹配具体的值：

```az
match x {
    case 0:
        println("零");
    case 1:
        println("一");
    case 42:
        println("答案");
    case _:
        println("其他");
}
```

### 2. 通配符模式

使用 `_` 匹配任何值：

```az
match x {
    case 0:
        println("零");
    case _:
        println("非零");  // 匹配所有其他值
}
```

### 3. 或模式（逗号分隔）

匹配多个值之一：

```az
match day {
    case 1, 2, 3, 4, 5:
        println("工作日");
    case 6, 7:
        println("周末");
    case _:
        println("无效");
}
```

### 4. 或模式（多个case）

使用多个连续的case（fall-through）：

```az
match day {
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
        println("工作日");
    case 6:
    case 7:
        println("周末");
    case _:
        println("无效");
}
```

### 5. 守卫条件

使用 `if` 添加额外条件：

```az
match n {
    case 0:
        println("零");
    case _ if n > 0:
        println("正数");
    case _ if n < 0:
        println("负数");
    case _:
        println("未知");
}
```

### 6. 变量绑定

捕获匹配的值：

```az
match x {
    case 0:
        println("零");
    case n if n > 0:
        println("正数: " + n);
    case n:
        println("负数: " + n);
}
```

---

## 📖 详细示例

### 示例1: 简单值匹配

```az
fn get_day_name(day: int) string {
    match day {
        case 1:
            return "Monday";
        case 2:
            return "Tuesday";
        case 3:
            return "Wednesday";
        case 4:
            return "Thursday";
        case 5:
            return "Friday";
        case 6:
            return "Saturday";
        case 7:
            return "Sunday";
        case _:
            return "Invalid";
    }
}
```

### 示例2: 使用代码块

```az
fn process_status(code: int) void {
    match code {
        case 200: {
            println("Success");
            log("Request completed successfully");
        }
        case 404: {
            println("Not Found");
            log("Resource not found");
        }
        case 500: {
            println("Server Error");
            log("Internal server error occurred");
        }
        case _: {
            println("Unknown Status");
            log("Unknown status code: " + code);
        }
    }
}
```

### 示例3: 守卫条件

```az
fn classify_temperature(temp: float) string {
    match temp {
        case _ if temp < 0.0:
            return "Freezing";
        case _ if temp < 10.0:
            return "Cold";
        case _ if temp < 20.0:
            return "Cool";
        case _ if temp < 30.0:
            return "Warm";
        case _ if temp < 40.0:
            return "Hot";
        case _:
            return "Extreme";
    }
}
```

### 示例4: 嵌套match

```az
fn process_input(type: string, value: int) string {
    match type {
        case "number": {
            match value {
                case 0:
                    return "Zero";
                case _ if value > 0:
                    return "Positive";
                case _:
                    return "Negative";
            }
        }
        case "boolean": {
            match value {
                case 0:
                    return "False";
                case 1:
                    return "True";
                case _:
                    return "Invalid boolean";
            }
        }
        case _:
            return "Unknown type";
    }
}
```

### 示例5: 字符串匹配

```az
fn execute_command(cmd: string) void {
    match cmd {
        case "start":
            println("Starting application...");
        case "stop":
            println("Stopping application...");
        case "restart":
            println("Restarting application...");
        case "status":
            println("Application is running");
        case "help": {
            println("Available commands:");
            println("  start   - Start the application");
            println("  stop    - Stop the application");
            println("  restart - Restart the application");
            println("  status  - Show application status");
            println("  help    - Show this help message");
        }
        case _:
            println("Unknown command: " + cmd);
    }
}
```

### 示例6: 范围匹配

```az
fn get_age_group(age: int) string {
    match age {
        case _ if age < 0:
            return "Invalid";
        case _ if age <= 12:
            return "Child";
        case _ if age <= 19:
            return "Teenager";
        case _ if age <= 59:
            return "Adult";
        case _ if age <= 120:
            return "Senior";
        case _:
            return "Invalid";
    }
}
```

### 示例7: 状态机

```az
struct StateMachine {
    state: int
}

fn transition(sm: *StateMachine, input: int) void {
    match sm.state {
        case 0: {
            match input {
                case 1:
                    sm.state = 1;
                case 2:
                    sm.state = 2;
                case _:
                    sm.state = 0;
            }
        }
        case 1: {
            match input {
                case 1:
                    sm.state = 2;
                case 2:
                    sm.state = 0;
                case _:
                    sm.state = 1;
            }
        }
        case 2: {
            sm.state = 0;
        }
        case _:
            sm.state = 0;
    }
}
```

---

## 🆚 与其他语言对比

### Python 3.10+

```python
match x:
    case 0:
        print("zero")
    case 1 | 2:
        print("one or two")
    case n if n > 10:
        print("big")
    case _:
        print("other")
```

### AZ

```az
match x {
    case 0:
        println("zero");
    case 1, 2:
        println("one or two");
    case n if n > 10:
        println("big");
    case _:
        println("other");
}
```

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

---

## ⚙️ 实现细节

### Token定义

```az
enum TokenType {
    // ...
    MATCH,    // match关键字
    CASE,     // case关键字
    // ...
}
```

### AST节点

```az
// Match语句
struct MatchStmt {
    expr: *Expr,           // 被匹配的表达式
    cases: []CaseArm       // case分支列表
}

// Case分支
struct CaseArm {
    patterns: []Pattern,   // 模式列表（支持多个）
    guard: *Expr,          // 可选的守卫条件
    body: *Stmt            // 分支体
}

// 模式
enum Pattern {
    Literal(value),        // 字面量
    Identifier(name),      // 标识符
    Wildcard              // 通配符 _
}
```

### 解析流程

```
1. 解析 match 关键字
2. 解析被匹配的表达式
3. 解析 { 
4. 循环解析 case 分支:
   a. 解析 case 关键字
   b. 解析模式（可能有多个，用逗号分隔）
   c. 可选：解析 if 守卫条件
   d. 解析 :
   e. 解析分支体（单语句或代码块）
5. 解析 }
```

### 代码生成

Match case会被降级为if-else链：

```az
// 源代码
match x {
    case 0:
        println("zero");
    case 1, 2:
        println("one or two");
    case _ if x > 10:
        println("big");
    case _:
        println("other");
}

// 生成的代码（概念）
if (x == 0) {
    println("zero");
} else if (x == 1 || x == 2) {
    println("one or two");
} else if (x > 10) {
    println("big");
} else {
    println("other");
}
```

---

## 🎯 最佳实践

### 1. 总是包含默认case

```az
// ✅ 好
match x {
    case 0:
        println("zero");
    case _:
        println("other");
}

// ❌ 不好（可能遗漏情况）
match x {
    case 0:
        println("zero");
}
```

### 2. 使用守卫条件处理范围

```az
// ✅ 好
match age {
    case _ if age < 18:
        return "Minor";
    case _ if age < 65:
        return "Adult";
    case _:
        return "Senior";
}

// ❌ 不好（难以维护）
match age {
    case 0, 1, 2, ..., 17:
        return "Minor";
    // ...
}
```

### 3. 按照可能性排序

```az
// ✅ 好（最常见的情况在前）
match status {
    case 200:
        return "OK";
    case 404:
        return "Not Found";
    case 500:
        return "Server Error";
    case _:
        return "Other";
}
```

### 4. 使用有意义的变量名

```az
// ✅ 好
match score {
    case s if s >= 90:
        return "A";
    case s if s >= 80:
        return "B";
    case _:
        return "F";
}

// ❌ 不好
match score {
    case _ if score >= 90:
        return "A";
    case _ if score >= 80:
        return "B";
    case _:
        return "F";
}
```

---

## 🚀 未来扩展

### 1. 结构体模式

```az
match point {
    case Point { x: 0, y: 0 }:
        println("Origin");
    case Point { x: 0, y: _ }:
        println("On Y axis");
    case Point { x: _, y: 0 }:
        println("On X axis");
    case _:
        println("Somewhere else");
}
```

### 2. 元组模式

```az
match (x, y) {
    case (0, 0):
        println("Origin");
    case (0, _):
        println("On Y axis");
    case (_, 0):
        println("On X axis");
    case _:
        println("Somewhere else");
}
```

### 3. 数组模式

```az
match arr {
    case []:
        println("Empty");
    case [x]:
        println("One element");
    case [x, y]:
        println("Two elements");
    case [x, ...rest]:
        println("Multiple elements");
}
```

### 4. 范围模式

```az
match x {
    case 0..10:
        println("0-9");
    case 10..20:
        println("10-19");
    case _:
        println("Other");
}
```

---

## 📊 性能考虑

### 编译时优化

1. **跳转表** - 连续整数值使用跳转表
2. **二分查找** - 稀疏整数值使用二分查找
3. **哈希表** - 字符串匹配使用哈希表

### 运行时性能

| 模式类型 | 时间复杂度 | 说明 |
|---------|-----------|------|
| 连续整数 | O(1) | 跳转表 |
| 稀疏整数 | O(log n) | 二分查找 |
| 字符串 | O(1) | 哈希表 |
| 守卫条件 | O(n) | 顺序检查 |

---

## 📝 总结

### AZ的Match Case特点

✅ **Python风格** - 使用case关键字  
✅ **简洁语法** - 清晰易读  
✅ **强大功能** - 支持守卫、嵌套、代码块  
✅ **类型安全** - 编译时检查  
✅ **高性能** - 编译时优化

### 与Python的区别

| 特性 | Python | AZ |
|------|--------|-----|
| 代码块 | 缩进 | `{}` |
| 或模式 | `\|` | `,` |
| 类型检查 | 运行时 | 编译时 |
| 性能 | 解释执行 | 编译优化 |

---

**AZ的Match Case - 结合Python的简洁和系统语言的性能！** 🚀

