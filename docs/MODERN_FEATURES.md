# AZ语言现代化编程特性

本文档详细介绍AZ语言的现代化编程特性，包括依赖注入、中间件模式、异步编程、LINQ查询和强类型系统。

## 目录

1. [依赖注入 (Dependency Injection)](#依赖注入-dependency-injection)
2. [中间件模式 (Middleware Pattern)](#中间件模式-middleware-pattern)
3. [异步编程 (Async Programming)](#异步编程-async-programming)
4. [LINQ查询 (LINQ-style Queries)](#linq查询-linq-style-queries)
5. [强类型系统 (Strong Type System)](#强类型系统-strong-type-system)

## 依赖注入 (Dependency Injection)

AZ标准库提供了完整的依赖注入框架，位于[di.az](file:///d:/编程项目/az/stdlib/di.az)模块中。

### 核心概念

- **ServiceLifetime**: 服务生命周期枚举
  - `Transient`: 每次请求都创建新实例
  - `Scoped`: 作用域内单例
  - `Singleton`: 全局单例

- **ServiceDescriptor**: 服务描述符，包含服务类型、实现和生命周期

- **ServiceProvider**: 服务提供者，负责服务解析

- **ServiceContainer**: 服务容器，负责服务注册和构建

### 使用示例

```az
import "di";

// 定义服务接口
interface IUserService {
    fn get_user_name(self: *Self, id: int) string;
}

// 实现服务
struct UserService {}

impl UserService : IUserService {
    fn new() UserService {
        return UserService{};
    }
    
    fn get_user_name(self: *Self, id: int) string {
        return "张三";
    }
}

// 服务注册
fn main() {
    let mut container = ServiceContainer::new();
    container.register_singleton<IUserService, UserService>(UserService::new());
    container.build();
    
    // 设置全局容器
    ServiceProvider::set_global_container(container);
    
    // 服务解析
    let user_service = ServiceProvider::resolve<IUserService>();
    println(user_service.get_user_name(1));
}
```

## 中间件模式 (Middleware Pattern)

AZ提供了中间件模式支持，位于[middleware.az](file:///d:/编程项目/az/stdlib/middleware.az)模块中，可用于构建可扩展的请求处理管道。

### 核心概念

- **Middleware**: 中间件接口，所有中间件必须实现此接口

- **MiddlewareContext**: 中间件上下文，包含请求和响应信息

- **MiddlewarePipeline**: 中间件管道，负责管理中间件链

### 使用示例

```az
import "middleware";

// 定义中间件
struct LoggingMiddleware {}

impl LoggingMiddleware : Middleware {
    fn new() LoggingMiddleware {
        return LoggingMiddleware{};
    }
    
    fn process(self: *Self, context: *MiddlewareContext) Result<void, Error> {
        println("处理请求开始...");
        let result = context.next();
        println("处理请求结束.");
        return result;
    }
}

struct AuthMiddleware {}

impl AuthMiddleware : Middleware {
    fn new() AuthMiddleware {
        return AuthMiddleware{};
    }
    
    fn process(self: *Self, context: *MiddlewareContext) Result<void, Error> {
        if context.request.headers.get("Authorization").is_some() {
            return context.next();
        } else {
            return Err(Error::new("未授权"));
        }
    }
}

// 使用中间件管道
fn main() {
    let mut pipeline = MiddlewarePipeline::new();
    pipeline.add_middleware(LoggingMiddleware::new());
    pipeline.add_middleware(AuthMiddleware::new());
    
    let request = HttpRequest::new();
    match pipeline.execute(&mut request) {
        Ok(_) => println("请求处理成功"),
        Err(e) => println("请求处理失败: {}", e.message())
    }
}
```

## 异步编程 (Async Programming)

AZ提供了基于Future/Promise模式的异步编程支持，位于[async.az](file:///d:/编程项目/az/stdlib/async.az)模块中。

### 核心概念

- **Task<T>**: 异步任务，表示一个可能还未完成的计算

- **Promise<T>**: 承诺，用于设置异步任务的结果

- **Future<T>**: 未来，用于获取异步任务的结果

- **AsyncStream<T>**: 异步流，支持流式数据处理

### 使用示例

```az
import "async";

// 创建异步任务
fn fetch_data_async(id: int) Task<string> {
    let promise = Promise<string>::new();
    let task = Task::from_promise(promise);
    
    // 模拟异步操作
    go fn() {
        // 模拟网络请求
        std::thread::sleep(std::time::Duration::from_millis(1000));
        promise.set_value(format("数据-{}", id));
    }();
    
    return task;
}

// 使用异步任务
fn main() {
    let task1 = fetch_data_async(1);
    let task2 = fetch_data_async(2);
    
    // 等待任务完成
    let result1 = task1.await();
    let result2 = task2.await();
    
    println("结果1: {}", result1.unwrap());
    println("结果2: {}", result2.unwrap());
}
```

## LINQ查询 (LINQ-style Queries)

AZ提供了LINQ风格的函数式查询操作，位于[linq.az](file:///d:/编程项目/az/stdlib/linq.az)模块中。

### 核心概念

- **Queryable<T>**: 可查询接口，提供各种查询操作

- **QueryBuilder<T>**: 查询构建器，用于构建复杂查询

- **Group<K, V>**: 分组结果

### 支持的操作

- `filter`: 过滤元素
- `map`: 映射元素
- `flat_map`: 展平映射
- `take`: 获取前N个元素
- `skip`: 跳过前N个元素
- `distinct`: 去重
- `sort`: 排序
- `group_by`: 分组
- `to_list`: 转换为列表

### 使用示例

```az
import "linq";

struct User {
    id: int,
    name: string,
    age: int,
    department: string,
}

fn main() {
    let users = [
        User{id: 1, name: "张三", age: 25, department: "工程部"},
        User{id: 2, name: "李四", age: 30, department: "市场部"},
        User{id: 3, name: "王五", age: 35, department: "工程部"},
    ];
    
    // 查询工程部的用户并获取姓名
    let engineer_names = users.as_queryable()
        .filter(fn(u: User) bool { return u.department == "工程部"; })
        .map(fn(u: User) string { return u.name; })
        .to_list();
    
    println("工程部员工:");
    for engineer_names {
        println("  - {}", it);
    }
    
    // 按部门分组统计人数
    let department_stats = users.as_queryable()
        .group_by(fn(u: User) string { return u.department; })
        .map(fn(g: Group<string, User>) (string, int) { 
            return (g.key, g.items.len()); 
        })
        .to_list();
    
    println("部门统计:");
    for department_stats {
        println("  - {}: {}人", it.0, it.1);
    }
}
```

## 强类型系统 (Strong Type System)

AZ具有静态强类型系统，支持类型推断和结构化数据输出。

### 特性

1. **静态类型检查**: 在编译时检查类型安全
2. **类型推断**: 自动推断变量类型
3. **结构化数据**: 支持结构体和枚举
4. **泛型支持**: 支持泛型编程

### 使用示例

```az
// 结构体定义
struct User {
    id: int,
    name: string,
    age: int,
    email: string,
}

// 枚举定义
enum UserRole {
    Admin,
    User,
    Guest,
}

// 泛型结构体
struct Result<T> {
    value: T,
    success: bool,
    error_message: string,
}

fn main() {
    // 类型推断
    let user = User{
        id: 1,
        name: "张三",
        age: 25,
        email: "zhangsan@example.com"
    };
    
    // 使用结构化数据
    println("用户信息:");
    println("  ID: {}", user.id);
    println("  姓名: {}", user.name);
    println("  年龄: {}", user.age);
    println("  邮箱: {}", user.email);
    
    // 泛型使用
    let success_result = Result<string>{
        value: "操作成功",
        success: true,
        error_message: ""
    };
    
    if success_result.success {
        println("结果: {}", success_result.value);
    }
}
```

## 总结

AZ语言的现代化编程特性为开发者提供了强大的工具来构建复杂且可维护的应用程序。这些特性包括：

1. **依赖注入**: 支持松耦合的设计和更好的可测试性
2. **中间件模式**: 支持可扩展的请求处理管道
3. **异步编程**: 支持高效的并发处理
4. **LINQ查询**: 提供函数式编程风格的数据查询
5. **强类型系统**: 确保类型安全和更好的代码质量

这些特性使AZ语言不仅适合系统编程，也适合构建复杂的应用程序。