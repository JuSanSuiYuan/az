# AZGC垃圾回收器设计

## 核心特性

AZGC（AZ Garbage Collector）是AZ语言的垃圾回收器实现，具有以下核心特性：

### 1. 按需启用
- AZGC不是默认的内存管理机制
- 开发者可以选择性地在特定模块或作用域中启用
- 与所有权系统互补，而非竞争关系

### 2. 超低延迟
- 亚毫秒级的停顿时间（< 1ms）
- 并发执行，最大程度减少对应用程序的影响
- 适用于对响应时间敏感的应用场景

### 3. 智能触发策略
- 自动识别所有权系统难以处理的场景
- 在循环引用、复杂对象图等场景下自动启用
- 提供手动控制选项

## 技术架构

### 1. 分代收集
AZGC采用分代垃圾回收策略：

#### 年轻代（Young Generation）
- 使用复制算法（Copying Algorithm）
- 针对生命周期短的对象
- 高效回收大量临时对象

#### 老年代（Old Generation）
- 使用标记-压缩算法（Mark-Compact Algorithm）
- 针对生命周期长的对象
- 并发执行以减少停顿时间

### 2. 并发执行
AZGC的核心是并发执行机制：

#### 并发标记阶段
- 应用线程与GC线程同时运行
- 使用三色标记法（Tri-color Marking）
- 通过写屏障（Write Barrier）维护一致性

#### 并发压缩阶段
- 并发整理内存碎片
- 使用转发指针（Forwarding Pointer）技术
- 避免对象移动时的同步问题

### 3. 写屏障技术
为了确保并发执行的正确性，AZGC使用写屏障：

```az
// 伪代码示例
fn write_barrier(obj, field, value) void {
    // 记录对象引用变化
    if (is_gc_active()) {
        log_reference_change(obj, field, value);
    }
    // 执行实际写操作
    *field = value;
}
```

## 内存布局

### 堆结构
```
+------------------+ 
|   Eden Space     |  <- 年轻代
+------------------+
|   Survivor Space |  <- 年轻代
+------------------+
|   Old Generation |  <- 老年代
+------------------+
|   Large Object   |  <- 大对象区域
+------------------+
```

### 对象头设计
每个GC管理的对象都有对象头：

```c
struct ObjectHeader {
    uintptr_t mark_bit : 1;     // 标记位
    uintptr_t age : 4;          // 年龄（用于分代）
    uintptr_t size : 20;        // 对象大小
    uintptr_t type_id : 20;     // 类型标识
    // 其他元数据...
};
```

## 触发策略

### 自动触发条件
AZGC在以下情况下自动触发：

1. **内存压力**：堆使用率达到阈值（默认70%）
2. **循环引用检测**：检测到无法通过所有权系统管理的循环引用
3. **复杂对象图**：对象间引用关系过于复杂，难以静态分析

### 手动控制
开发者可以通过以下方式手动控制AZGC：

```az
// 在模块级别启用GC
#[gc(enable=true)]
module my_module;

// 在作用域级别启用GC
fn main() int {
    gc_scope {
        // 此作用域内的对象由AZGC管理
        let obj = MyObject::new();
        // 无需手动释放，GC会自动管理
    }  // 作用域结束，触发GC
    
    // 默认作用域使用所有权管理
    return 0;
}

// 对象级别控制
fn main() int {
    // 明确指定对象使用GC管理
    let gc_obj = gc_alloc(MyObject::new());
    
    // 默认使用所有权管理
    let owned_obj = MyObject::new();
    return 0;
}
```

## 性能优化

### 1. 增量回收
- 将GC工作分解为小的任务单元
- 避免长时间的停顿
- 根据应用负载动态调整回收频率

### 2. 并行处理
- 利用多核CPU并行执行GC任务
- 标记和压缩阶段均可并行化
- 动态调整工作线程数量

### 3. 预测性调度
- 基于历史数据预测GC时机
- 避免在关键业务时段触发GC
- 提供API供开发者提示GC时机

## 与标准库集成

### GC感知的数据结构
```az
import std.gc.collections;

fn main() int {
    // GcVec - 由GC管理的动态数组
    let vec = GcVec::new();
    vec.push("Hello");
    vec.push("World");
    // 无需手动释放，GC会自动管理
    
    // GcHashMap - 由GC管理的哈希映射
    let map = GcHashMap::new();
    map.insert("key", "value");
    
    return 0;
}
```

### 弱引用支持
```az
import std.gc.weakref;

fn main() int {
    gc_scope {
        let obj = MyObject::new();
        let weak_ref = WeakRef::new(&obj);
        
        // 使用对象...
        
        // 检查对象是否仍存在
        if (let strong_ref = weak_ref.upgrade()) {
            // 对象仍然存在
            println(strong_ref.value);
        } else {
            // 对象已被回收
            println("Object has been collected");
        }
    }
    
    return 0;
}
```

## 配置选项

### 运行时配置
```az
// chim.toml 或 package.az 中的配置
[gc]
enabled = true           # 启用GC
threshold = 0.7          # 触发GC的堆使用率阈值
parallel_threads = 4     # 并行GC线程数
concurrent = true        # 是否并发执行
```

### 编译时注解
```az
// 启用详细的GC日志
#[gc(verbose=true)]
module debug_module;

// 设置特定的GC策略
#[gc(strategy="low_latency")]
fn critical_function() void {
    // 执行关键任务
}
```

## 监控和调试

### GC统计信息
AZGC提供详细的统计信息：

```az
import std.gc.stats;

fn main() int {
    let stats = gc_stats();
    println("GC运行次数: ", stats.collections);
    println("回收内存: ", stats.reclaimed_bytes, " bytes");
    println("总停顿时间: ", stats.pause_time_ms, " ms");
    return 0;
}
```

### 内存分析工具
提供内存分析工具帮助开发者优化内存使用：

```az
import std.gc.profiler;

fn main() int {
    gc_profiler_start();
    
    // 执行一些操作...
    
    let report = gc_profiler_stop();
    report.print_summary();
    return 0;
}
```

## 与其他内存管理方式的对比

| 特性 | 所有权系统 | AZGC | 手动管理 |
|------|------------|------|----------|
| 内存安全 | ✅ 编译时保证 | ✅ 运行时保证 | ❌ 需要开发者保证 |
| 性能 | ⚡ 最高 | 🔧 高 | ⚡ 最高 |
| 易用性 | 🔧 中等 | ✅ 高 | ❌ 低 |
| 学习成本 | 🔧 中等 | ✅ 低 | ❌ 高 |
| 循环引用处理 | ❌ 不支持 | ✅ 支持 | ❌ 需要开发者处理 |

## 最佳实践

1. **默认使用所有权系统**：在大多数情况下，所有权系统提供更好的性能
2. **在复杂场景中启用AZGC**：当遇到循环引用或复杂对象图时，启用AZGC
3. **合理配置GC参数**：根据应用特点调整GC触发阈值和并发设置
4. **监控GC性能**：使用提供的工具监控GC行为，及时发现性能问题
5. **混合使用策略**：在同一项目中不同模块使用不同的内存管理方式

## 未来发展方向

1. **更智能的触发策略**：基于机器学习预测最佳GC时机
2. **区域化GC**：针对不同内存区域采用不同的回收策略
3. **跨语言互操作**：与其他语言的GC系统集成
4. **云原生优化**：针对容器化环境优化GC行为