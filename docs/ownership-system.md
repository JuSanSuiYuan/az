# AZ语言所有权系统设计

## 核心理念

AZ语言的所有权系统借鉴了Rust的所有权概念，但进行了简化和优化，以适应更广泛的开发者群体。所有权系统的主要目标是：

1. **内存安全**：在编译时防止内存泄漏、悬空指针和数据竞争
2. **零运行时开销**：所有权检查在编译时完成，不引入运行时负担
3. **易用性**：相比Rust，提供更简化的所有权模型，降低学习门槛

## 基本原则

### 1. 所有权规则
- 每个值都有一个所有者（owner）
- 同一时刻只能有一个所有者
- 当所有者离开作用域时，值被自动释放

### 2. 移动语义
- 赋值或传递参数时，默认进行移动而非复制
- 移动后，原变量不再可用

### 3. 借用检查
- 通过引用（&T）临时借用所有权
- 同一时刻只能有一个可变引用（&mut T）或多个不可变引用（&T）
- 引用的生命周期必须短于所有者

## 语法设计

### 所有者变量
```az
let x = 5;        // x 是 i32 类型的所有者
let y = x;        // x 的值移动到 y，x 不再可用
// println(x);    // 编译错误：x 已被移动
```

### 引用和借用
```az
let x = 5;
let ref_x = &x;   // 不可变借用
let mut y = 10;
let ref_y = &mut y; // 可变借用
```

### 函数参数
```az
fn take_ownership(s: string) void {
    // s 成为此字符串的所有者
    println(s);
} // s 离开作用域，字符串被释放

fn borrow(s: &string) void {
    // 借用字符串，不获取所有权
    println(s);
} // s 离开作用域，但不释放字符串

fn main() int {
    let s = "Hello";
    borrow(&s);        // 借用 s
    take_ownership(s); // 移动 s
    return 0;
}
```

## 编译时检查

### 生命周期检查
编译器会自动分析引用的生命周期，确保引用不会超出所有者的生命周期：

```az
fn dangling_ref() &int {
    let x = 5;
    return &x;  // 编译错误：返回对局部变量的引用
}
```

### 借用检查规则
```az
fn main() int {
    let mut x = 5;
    let ref1 = &x;      // 不可变借用
    let ref2 = &x;      // 可以有多个不可变借用
    // let ref3 = &mut x; // 编译错误：不能同时有可变和不可变借用
    println(ref1, ref2);
    
    let ref3 = &mut x;  // 可变借用
    *ref3 = 10;         // 修改值
    // println(ref3);   // 编译错误：可变借用期间不能有其他借用
    return 0;
}
```

## 与GC的集成

### 所有权与GC的关系
AZ语言采用混合内存管理模型：
- 默认使用所有权系统管理内存
- AZGC作为可选的后备机制，处理复杂场景

### 手动控制选项
开发者可以通过注解控制内存管理方式：

```az
// 默认使用所有权管理
struct MyStruct {
    data: []int
}

// 明确指定使用GC管理
struct MyStructGC: #[gc]
    data: []int

// 作用域级别控制
fn main() int {
    gc_scope {  // 此作用域内使用GC
        let obj = MyStruct{data: [1, 2, 3]};
        // obj 由GC管理
    }  // 作用域结束，GC负责清理
    
    let obj2 = MyStruct{data: [4, 5, 6]};
    // obj2 由所有权系统管理
    return 0;
}
```

## 标准库支持

### 智能指针
```az
import std.ptr;

fn main() int {
    // Box<T> - 堆分配的独占指针
    let boxed = Box::new(42);
    println(boxed.value);
    
    // Rc<T> - 引用计数指针（需要启用GC）
    gc_scope {
        let shared = Rc::new(42);
        let shared2 = shared.clone();
        println(shared.value, shared2.value);
    }
    
    return 0;
}
```

### 集合类型
```az
import std.collections;

fn main() int {
    // Vec<T> - 动态数组，由所有权系统管理
    let mut vec = Vec::new();
    vec.push(1);
    vec.push(2);
    
    // 当vec离开作用域时，内部数据自动释放
    
    // HashMap<K, V> - 哈希映射
    let mut map = HashMap::new();
    map.insert("key", "value");
    
    return 0;
}
```

## 错误处理

当所有权规则被违反时，编译器会提供清晰的错误信息：

```
error: use of moved value
 --> main.az:5:5
  |
3 | let y = x;  // x 的值移动到 y
4 | println(x); // 尝试使用已移动的值
  |         ^ value used here after move
  |
  = note: move occurs because `x` has type `string`, which does not implement the `Copy` trait
```

## 最佳实践

1. **优先使用借用**：尽可能使用引用而非获取所有权
2. **合理设计API**：函数参数使用引用，返回值谨慎使用移动语义
3. **利用生命周期**：通过生命周期参数明确表达引用关系
4. **适时使用Clone**：对于小型数据结构，可以使用clone方法进行复制

## 未来扩展

1. **Copy trait**：为简单的数据类型实现自动复制
2. **生命周期省略**：在简单情况下自动推断生命周期
3. **借用检查器优化**：支持更复杂的借用场景
4. **与异步集成**：在异步环境中正确管理所有权