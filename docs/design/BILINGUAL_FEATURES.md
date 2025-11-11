# AZ语言双语特性

**中英文混合编程支持**

---

## 🌏 概述

AZ语言支持中英文混合编程，允许开发者使用中文或英文关键字编写代码，并提供双语错误信息。

---

## 📚 中文关键字对照表

### 完整对照表

| 英文 | 中文 | 用途 |
|------|------|------|
| fn | 函数 | 函数定义 |
| return | 返回 | 返回语句 |
| if | 如果 | 条件判断 |
| else | 否则 | 否则分支 |
| for | 循环 | for循环 |
| while | 当 | while循环 |
| let | 令 | 不可变变量 |
| var | 设 | 可变变量 |
| const | 常 | 常量 |
| import | 导入 | 导入模块 |
| module | 模块 | 模块声明 |
| pub | 公开 | 公开可见性 |
| struct | 结构 | 结构体 |
| enum | 枚举 | 枚举类型 |
| match | 匹配 | 模式匹配 |
| case | 情况 | case分支 |
| comptime | 编译时 | 编译时执行 |

---

## 💡 使用示例

### 示例1: 纯中文编程

```az
函数 阶乘(n: int) int {
    如果 (n <= 1) {
        返回 1;
    }
    返回 n * 阶乘(n - 1);
}

函数 主函数() int {
    令 结果 = 阶乘(5);
    返回 结果;
}
```

### 示例2: 纯英文编程

```az
fn factorial(n: int) int {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}

fn main() int {
    let result = factorial(5);
    return result;
}
```

### 示例3: 中英文混合编程

```az
// 中文函数定义
函数 计算总和(arr: []int, len: int) int {
    设 总和 = 0;
    循环 (设 i = 0; i < len; i = i + 1) {
        总和 = 总和 + arr[i];
    }
    返回 总和;
}

// 英文函数定义
fn calculate_average(arr: []int, len: int) int {
    let sum = 计算总和(arr, len);  // 调用中文函数
    return sum / len;
}

// 主函数
fn main() int {
    令 数组 = [1, 2, 3, 4, 5];
    let avg = calculate_average(数组, 5);
    返回 avg;
}
```

### 示例4: 变量声明

```az
fn main() int {
    // 英文：不可变变量
    let x = 10;
    
    // 中文：不可变变量
    令 y = 20;
    
    // 英文：可变变量
    var a = 30;
    
    // 中文：可变变量
    设 b = 40;
    
    // 修改可变变量
    a = a + 1;
    b = b + 1;
    
    return x + y + a + b;
}
```

### 示例5: 控制流

```az
函数 判断大小(n: int) string {
    如果 (n > 100) {
        返回 "大";
    } 否则 {
        返回 "小";
    }
}

fn check_range(n: int) string {
    if (n < 0) {
        return "negative";
    } else {
        return "positive";
    }
}
```

### 示例6: 循环

```az
// 中文for循环
函数 求和() int {
    设 总和 = 0;
    循环 (设 i = 0; i < 10; i = i + 1) {
        总和 = 总和 + i;
    }
    返回 总和;
}

// 英文while循环
fn count_down() int {
    var n = 10;
    while (n > 0) {
        n = n - 1;
    }
    return n;
}
```

---

## 🔍 双语错误信息

### 错误信息格式

AZ编译器会输出双语错误信息：
1. 第一行：英文错误信息
2. 第二行：中文错误信息
3. 第三行：错误详情（英文）
4. 第四行：错误详情（中文，如果有翻译）

### 示例1: 语法错误

**代码**:
```az
fn main() int {
    let x = 10
    return x;
}
```

**错误输出**:
```
[Error] Syntax Error at test.az:3:11
[错误] 语法错误 在 test.az:3:11
  期望 ';'
```

### 示例2: 类型错误

**代码**:
```az
fn add(a: int, b: int) int {
    return a + b;
}

fn main() int {
    return add(1, "hello");
}
```

**错误输出**:
```
[Error] Type Error at test.az:6:18
[错误] 类型错误 在 test.az:6:18
  Type mismatch
  类型不匹配
```

### 错误类型对照

| 英文 | 中文 |
|------|------|
| Lexical Error | 词法错误 |
| Syntax Error | 语法错误 |
| Semantic Error | 语义错误 |
| Type Error | 类型错误 |
| Runtime Error | 运行时错误 |

### 常见错误信息翻译

| 英文 | 中文 |
|------|------|
| Expected ';' | 期望 ';' |
| Expected '(' | 期望 '(' |
| Expected ')' | 期望 ')' |
| Expected '{' | 期望 '{' |
| Expected '}' | 期望 '}' |
| Expected identifier | 期望标识符 |
| Expected type name | 期望类型名 |
| Expected expression | 期望表达式 |
| Undefined variable | 未定义的变量 |
| Type mismatch | 类型不匹配 |

---

## 🎯 最佳实践

### 1. 团队协作

**建议**: 团队内统一使用英文或中文

```az
// ✅ 好 - 统一使用中文
函数 计算(x: int) int {
    令 结果 = x * 2;
    返回 结果;
}

// ✅ 好 - 统一使用英文
fn calculate(x: int) int {
    let result = x * 2;
    return result;
}

// ⚠️ 可以但不推荐 - 混合使用
函数 calculate(x: int) int {
    let 结果 = x * 2;
    return 结果;
}
```

### 2. 变量命名

**建议**: 变量名可以使用中文或英文

```az
// 中文变量名
令 总和 = 0;
令 计数器 = 10;

// 英文变量名
let sum = 0;
let counter = 10;

// 混合（不推荐）
let 总和 = 0;
```

### 3. 函数命名

**建议**: 函数名保持一致的语言风格

```az
// ✅ 好 - 中文函数名
函数 计算总和() int { }
函数 获取数据() int { }

// ✅ 好 - 英文函数名
fn calculate_sum() int { }
fn get_data() int { }
```

### 4. 注释

**建议**: 注释可以使用任何语言

```az
// 这是中文注释
// This is an English comment

函数 示例() int {
    // 计算结果
    令 结果 = 42;
    返回 结果;
}
```

---

## 🌟 特色功能

### 1. 完全双语支持

- ✅ 所有关键字都有中文对应
- ✅ 可以自由混合使用
- ✅ 编译器完全支持

### 2. 双语错误信息

- ✅ 英文错误信息
- ✅ 中文错误信息
- ✅ 自动翻译常见错误

### 3. 灵活性

- ✅ 可以在同一文件中混合使用
- ✅ 可以在不同文件中使用不同语言
- ✅ 函数可以互相调用

---

## 📊 使用场景

### 适合使用中文的场景

1. **教学** - 帮助初学者理解
2. **中文团队** - 提高代码可读性
3. **领域特定** - 业务逻辑用中文更清晰

### 适合使用英文的场景

1. **国际协作** - 与国际团队合作
2. **开源项目** - 更广泛的受众
3. **技术文档** - 与英文文档保持一致

### 适合混合使用的场景

1. **过渡期** - 从中文向英文过渡
2. **特殊需求** - 某些部分需要特定语言
3. **学习目的** - 同时学习两种表达方式

---

## 🎓 学习资源

### 中文编程示例

查看 `examples/chinese_example.az` 获取完整的中文编程示例。

### 编译和运行

```bash
# 编译中文程序
python az.py examples/chinese_example.az -o chinese_program

# 运行
./chinese_program
```

---

## 📝 总结

### AZ语言双语特性

✅ **完整的中文关键字** - 17个关键字全部支持  
✅ **双语错误信息** - 英文+中文  
✅ **自由混合** - 可以在同一文件中混合使用  
✅ **零性能损失** - 编译后性能完全相同  
✅ **易于学习** - 降低编程学习门槛

### 独特优势

1. **降低学习门槛** - 中文母语者更容易理解
2. **提高可读性** - 业务逻辑用中文更清晰
3. **国际化支持** - 支持多语言开发
4. **灵活性** - 可以根据需要选择语言

---

**AZ语言 - 真正的双语编程语言！** 🌏🚀

