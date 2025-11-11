# AZ标准库完善计划

**日期**: 2025年10月30日  
**目标**: 完善标准库，使AZ语言达到实用水平

---

## 📊 当前状态

### ✅ 已有框架
- stdlib/io.az - 基础I/O框架
- stdlib/string.az - 基础字符串框架
- stdlib/collections.az - 基础集合框架
- stdlib/error.az - 错误处理框架
- stdlib/math.az - 数学函数框架
- stdlib/fs.az - 文件系统框架
- stdlib/mem.az - 内存管理框架
- stdlib/os.az - 操作系统接口框架
- stdlib/time.az - 时间处理框架

### ⚠️ 需要完善
所有模块都只有框架，缺少实际实现

---

## 🎯 完善计划

### 阶段1: 核心标准库（Week 1-2）

#### 1.1 std.io - 输入输出 ⭐⭐⭐⭐⭐
**优先级**: 极高  
**时间**: 3天

**需要实现**:
```az
// 基础输出
fn print(s: string) void;
fn println(s: string) void;
fn eprint(s: string) void;  // 错误输出
fn eprintln(s: string) void;

// 格式化输出
fn printf(format: string, ...args) void;
fn sprintf(format: string, ...args) string;

// 基础输入
fn read_line() string;
fn read_char() char;
fn read_int() Result<int, ParseError>;
fn read_float() Result<float, ParseError>;

// 文件读写
fn read_file(path: string) Result<string, IOError>;
fn write_file(path: string, content: string) Result<void, IOError>;
fn append_file(path: string, content: string) Result<void, IOError>;
fn read_lines(path: string) Result<Vec<string>, IOError>;

// 文件操作
struct File;
fn open(path: string, mode: FileMode) Result<File, IOError>;
fn close(file: File) Result<void, IOError>;
fn read(file: File, buffer: []byte) Result<int, IOError>;
fn write(file: File, data: []byte) Result<int, IOError>;
fn seek(file: File, offset: int, whence: SeekMode) Result<int, IOError>;

// 缓冲I/O
struct BufReader;
struct BufWriter;
fn buf_reader(file: File) BufReader;
fn buf_writer(file: File) BufWriter;
```

#### 1.2 std.string - 字符串操作 ⭐⭐⭐⭐⭐
**优先级**: 极高  
**时间**: 3天

**需要实现**:
```az
// 基础操作
fn length(s: string) int;
fn is_empty(s: string) bool;
fn concat(a: string, b: string) string;
fn repeat(s: string, n: int) string;

// 大小写转换
fn to_upper(s: string) string;
fn to_lower(s: string) string;
fn to_title(s: string) string;

// 子字符串
fn substring(s: string, start: int, end: int) string;
fn slice(s: string, start: int, end: int) string;
fn take(s: string, n: int) string;
fn skip(s: string, n: int) string;

// 查找和匹配
fn find(s: string, sub: string) Option<int>;
fn rfind(s: string, sub: string) Option<int>;
fn contains(s: string, sub: string) bool;
fn starts_with(s: string, prefix: string) bool;
fn ends_with(s: string, suffix: string) bool;
fn count(s: string, sub: string) int;

// 分割和连接
fn split(s: string, sep: string) Vec<string>;
fn split_n(s: string, sep: string, n: int) Vec<string>;
fn split_whitespace(s: string) Vec<string>;
fn lines(s: string) Vec<string>;
fn join(parts: Vec<string>, sep: string) string;

// 修剪
fn trim(s: string) string;
fn trim_left(s: string) string;
fn trim_right(s: string) string;
fn trim_prefix(s: string, prefix: string) string;
fn trim_suffix(s: string, suffix: string) string;

// 替换
fn replace(s: string, old: string, new: string) string;
fn replace_n(s: string, old: string, new: string, n: int) string;
fn replace_all(s: string, old: string, new: string) string;

// 字符操作
fn chars(s: string) Vec<char>;
fn bytes(s: string) Vec<byte>;
fn char_at(s: string, index: int) Option<char>;

// 验证
fn is_alpha(s: string) bool;
fn is_numeric(s: string) bool;
fn is_alphanumeric(s: string) bool;
fn is_whitespace(s: string) bool;

// 格式化
fn format(template: string, ...args) string;
fn pad_left(s: string, width: int, fill: char) string;
fn pad_right(s: string, width: int, fill: char) string;
fn center(s: string, width: int, fill: char) string;

// 转换
fn to_int(s: string) Result<int, ParseError>;
fn to_float(s: string) Result<float, ParseError>;
fn to_bool(s: string) Result<bool, ParseError>;
fn from_int(n: int) string;
fn from_float(f: float) string;
fn from_bool(b: bool) string;
```

#### 1.3 std.error - 错误处理 ⭐⭐⭐⭐⭐
**优先级**: 极高  
**时间**: 2天

**需要实现**:
```az
// Result类型（已有框架，需完善）
enum Result<T, E> {
    Ok(T),
    Err(E)
}

impl Result<T, E> {
    fn is_ok() bool;
    fn is_err() bool;
    fn unwrap() T;
    fn unwrap_err() E;
    fn unwrap_or(default: T) T;
    fn unwrap_or_else(f: fn() T) T;
    fn expect(msg: string) T;
    fn map<U>(f: fn(T) U) Result<U, E>;
    fn map_err<F>(f: fn(E) F) Result<T, F>;
    fn and_then<U>(f: fn(T) Result<U, E>) Result<U, E>;
    fn or_else<F>(f: fn(E) Result<T, F>) Result<T, F>;
}

// Option类型（已有框架，需完善）
enum Option<T> {
    Some(T),
    None
}

impl Option<T> {
    fn is_some() bool;
    fn is_none() bool;
    fn unwrap() T;
    fn unwrap_or(default: T) T;
    fn unwrap_or_else(f: fn() T) T;
    fn expect(msg: string) T;
    fn map<U>(f: fn(T) U) Option<U>;
    fn and_then<U>(f: fn(T) Option<U>) Option<U>;
    fn or_else(f: fn() Option<T>) Option<T>;
    fn filter(f: fn(T) bool) Option<T>;
}

// 错误类型
struct Error {
    message: string,
    kind: ErrorKind,
    source: Option<Error>
}

enum ErrorKind {
    IOError,
    ParseError,
    TypeError,
    RuntimeError,
    NetworkError,
    FileNotFound,
    PermissionDenied,
    InvalidInput,
    Other
}

// 断言和panic
fn assert(condition: bool, msg: string) void;
fn assert_eq<T>(a: T, b: T, msg: string) void;
fn assert_ne<T>(a: T, b: T, msg: string) void;
fn panic(msg: string) void;
fn todo(msg: string) void;
fn unreachable(msg: string) void;
```

---

### 阶段2: 集合和数据结构（Week 2-3）

#### 2.1 std.collections - 集合类型 ⭐⭐⭐⭐
**优先级**: 高  
**时间**: 5天

**需要实现**:
```az
// Vec<T> - 动态数组
struct Vec<T>;

impl Vec<T> {
    fn new() Vec<T>;
    fn with_capacity(cap: int) Vec<T>;
    fn from_array(arr: []T) Vec<T>;
    
    fn push(item: T) void;
    fn pop() Option<T>;
    fn insert(index: int, item: T) void;
    fn remove(index: int) Option<T>;
    fn clear() void;
    
    fn get(index: int) Option<T>;
    fn set(index: int, item: T) bool;
    fn first() Option<T>;
    fn last() Option<T>;
    
    fn len() int;
    fn capacity() int;
    fn is_empty() bool;
    fn contains(item: T) bool;
    fn find(item: T) Option<int>;
    
    fn sort() void;
    fn reverse() void;
    fn filter(f: fn(T) bool) Vec<T>;
    fn map<U>(f: fn(T) U) Vec<U>;
    fn fold<U>(init: U, f: fn(U, T) U) U;
    
    fn as_slice() []T;
    fn to_array() []T;
}

// HashMap<K, V> - 哈希表
struct HashMap<K, V>;

impl HashMap<K, V> {
    fn new() HashMap<K, V>;
    fn with_capacity(cap: int) HashMap<K, V>;
    
    fn insert(key: K, value: V) Option<V>;
    fn get(key: K) Option<V>;
    fn remove(key: K) Option<V>;
    fn clear() void;
    
    fn contains_key(key: K) bool;
    fn len() int;
    fn is_empty() bool;
    
    fn keys() Vec<K>;
    fn values() Vec<V>;
    fn entries() Vec<(K, V)>;
}

// HashSet<T> - 集合
struct HashSet<T>;

impl HashSet<T> {
    fn new() HashSet<T>;
    fn with_capacity(cap: int) HashSet<T>;
    
    fn insert(item: T) bool;
    fn remove(item: T) bool;
    fn contains(item: T) bool;
    fn clear() void;
    
    fn len() int;
    fn is_empty() bool;
    
    fn union(other: HashSet<T>) HashSet<T>;
    fn intersection(other: HashSet<T>) HashSet<T>;
    fn difference(other: HashSet<T>) HashSet<T>;
    fn is_subset(other: HashSet<T>) bool;
    fn is_superset(other: HashSet<T>) bool;
}

// LinkedList<T> - 链表
struct LinkedList<T>;

impl LinkedList<T> {
    fn new() LinkedList<T>;
    
    fn push_front(item: T) void;
    fn push_back(item: T) void;
    fn pop_front() Option<T>;
    fn pop_back() Option<T>;
    
    fn front() Option<T>;
    fn back() Option<T>;
    
    fn len() int;
    fn is_empty() bool;
    fn clear() void;
}

// BTreeMap<K, V> - 有序映射
struct BTreeMap<K, V>;

// BTreeSet<T> - 有序集合
struct BTreeSet<T>;

// VecDeque<T> - 双端队列
struct VecDeque<T>;
```

---

### 阶段3: 文件系统和数学（Week 3-4）

#### 3.1 std.fs - 文件系统 ⭐⭐⭐⭐
**优先级**: 高  
**时间**: 3天

**需要实现**:
```az
// 文件操作
fn exists(path: string) bool;
fn is_file(path: string) bool;
fn is_dir(path: string) bool;
fn is_symlink(path: string) bool;

fn create_file(path: string) Result<File, IOError>;
fn remove_file(path: string) Result<void, IOError>;
fn copy_file(src: string, dst: string) Result<void, IOError>;
fn rename(old: string, new: string) Result<void, IOError>;

fn metadata(path: string) Result<Metadata, IOError>;
fn file_size(path: string) Result<int, IOError>;
fn modified_time(path: string) Result<Time, IOError>;

// 目录操作
fn create_dir(path: string) Result<void, IOError>;
fn create_dir_all(path: string) Result<void, IOError>;
fn remove_dir(path: string) Result<void, IOError>;
fn remove_dir_all(path: string) Result<void, IOError>;
fn read_dir(path: string) Result<Vec<DirEntry>, IOError>;

// 路径操作
fn join(parts: Vec<string>) string;
fn split(path: string) (string, string);
fn basename(path: string) string;
fn dirname(path: string) string;
fn extension(path: string) Option<string>;
fn absolute(path: string) Result<string, IOError>;
fn canonicalize(path: string) Result<string, IOError>;

// 权限
fn chmod(path: string, mode: int) Result<void, IOError>;
fn chown(path: string, uid: int, gid: int) Result<void, IOError>;

// 临时文件
fn temp_dir() string;
fn temp_file() Result<File, IOError>;
```

#### 3.2 std.math - 数学函数 ⭐⭐⭐
**优先级**: 中  
**时间**: 2天

**需要实现**:
```az
// 常量
const PI: float = 3.14159265358979323846;
const E: float = 2.71828182845904523536;
const SQRT2: float = 1.41421356237309504880;

// 基础运算
fn abs(x: int) int;
fn abs_f(x: float) float;
fn min(a: int, b: int) int;
fn max(a: int, b: int) int;
fn min_f(a: float, b: float) float;
fn max_f(a: float, b: float) float;
fn clamp(x: int, min: int, max: int) int;

// 幂和根
fn pow(base: float, exp: float) float;
fn sqrt(x: float) float;
fn cbrt(x: float) float;
fn exp(x: float) float;
fn log(x: float) float;
fn log10(x: float) float;
fn log2(x: float) float;

// 三角函数
fn sin(x: float) float;
fn cos(x: float) float;
fn tan(x: float) float;
fn asin(x: float) float;
fn acos(x: float) float;
fn atan(x: float) float;
fn atan2(y: float, x: float) float;

// 双曲函数
fn sinh(x: float) float;
fn cosh(x: float) float;
fn tanh(x: float) float;

// 取整
fn floor(x: float) float;
fn ceil(x: float) float;
fn round(x: float) float;
fn trunc(x: float) float;

// 其他
fn sign(x: float) int;
fn copysign(x: float, y: float) float;
fn hypot(x: float, y: float) float;
fn fmod(x: float, y: float) float;
```

---

### 阶段4: 系统和时间（Week 4-5）

#### 4.1 std.os - 操作系统接口 ⭐⭐⭐
**优先级**: 中  
**时间**: 3天

**需要实现**:
```az
// 环境变量
fn getenv(key: string) Option<string>;
fn setenv(key: string, value: string) Result<void, OSError>;
fn unsetenv(key: string) Result<void, OSError>;
fn environ() HashMap<string, string>;

// 进程
fn getpid() int;
fn getppid() int;
fn exit(code: int) void;
fn abort() void;

// 命令执行
fn exec(cmd: string, args: Vec<string>) Result<int, OSError>;
fn spawn(cmd: string, args: Vec<string>) Result<Process, OSError>;

struct Process {
    pid: int
}

impl Process {
    fn wait() Result<int, OSError>;
    fn kill() Result<void, OSError>;
    fn is_running() bool;
}

// 系统信息
fn hostname() Result<string, OSError>;
fn username() Result<string, OSError>;
fn home_dir() Result<string, OSError>;
fn current_dir() Result<string, OSError>;
fn set_current_dir(path: string) Result<void, OSError>;

// 平台信息
fn os_name() string;
fn os_version() string;
fn arch() string;
```

#### 4.2 std.time - 时间处理 ⭐⭐⭐
**优先级**: 中  
**时间**: 2天

**需要实现**:
```az
// 时间类型
struct Time {
    seconds: int,
    nanos: int
}

struct Duration {
    seconds: int,
    nanos: int
}

// 当前时间
fn now() Time;
fn unix_timestamp() int;

// Duration创建
fn seconds(n: int) Duration;
fn milliseconds(n: int) Duration;
fn microseconds(n: int) Duration;
fn nanoseconds(n: int) Duration;

// 时间操作
impl Time {
    fn add(d: Duration) Time;
    fn sub(d: Duration) Time;
    fn diff(other: Time) Duration;
    fn format(fmt: string) string;
    fn parse(s: string, fmt: string) Result<Time, ParseError>;
}

// Duration操作
impl Duration {
    fn as_seconds() int;
    fn as_millis() int;
    fn as_micros() int;
    fn as_nanos() int;
}

// 睡眠
fn sleep(d: Duration) void;
fn sleep_ms(ms: int) void;
```

---

### 阶段5: 高级功能（Week 5-6）

#### 5.1 std.net - 网络 ⭐⭐
**优先级**: 低  
**时间**: 5天

**需要实现**:
```az
// TCP
struct TcpListener;
struct TcpStream;

fn tcp_listen(addr: string) Result<TcpListener, NetError>;
fn tcp_connect(addr: string) Result<TcpStream, NetError>;

// UDP
struct UdpSocket;

fn udp_bind(addr: string) Result<UdpSocket, NetError>;

// HTTP客户端（简单版）
fn http_get(url: string) Result<string, NetError>;
fn http_post(url: string, body: string) Result<string, NetError>;
```

#### 5.2 std.json - JSON解析 ⭐⭐
**优先级**: 低  
**时间**: 3天

**需要实现**:
```az
enum JsonValue {
    Null,
    Bool(bool),
    Number(float),
    String(string),
    Array(Vec<JsonValue>),
    Object(HashMap<string, JsonValue>)
}

fn parse(s: string) Result<JsonValue, ParseError>;
fn stringify(value: JsonValue) string;
fn stringify_pretty(value: JsonValue) string;
```

#### 5.3 std.regex - 正则表达式 ⭐⭐
**优先级**: 低  
**时间**: 5天

**需要实现**:
```az
struct Regex;

fn compile(pattern: string) Result<Regex, RegexError>;

impl Regex {
    fn is_match(text: string) bool;
    fn find(text: string) Option<Match>;
    fn find_all(text: string) Vec<Match>;
    fn replace(text: string, replacement: string) string;
    fn replace_all(text: string, replacement: string) string;
}

struct Match {
    start: int,
    end: int,
    text: string
}
```

---

## 📊 实现优先级总结

### 🔴 第一优先级（Week 1-2）
1. std.io - 输入输出
2. std.string - 字符串操作
3. std.error - 错误处理

### 🟠 第二优先级（Week 2-3）
4. std.collections - 集合类型

### 🟡 第三优先级（Week 3-4）
5. std.fs - 文件系统
6. std.math - 数学函数

### 🟢 第四优先级（Week 4-5）
7. std.os - 操作系统接口
8. std.time - 时间处理

### 🔵 第五优先级（Week 5-6）
9. std.net - 网络
10. std.json - JSON解析
11. std.regex - 正则表达式

---

## 🎯 实现策略

### 策略1: 基于C库实现
```az
// 使用extern调用C标准库
extern "C" {
    fn strlen(s: *char) int;
    fn strcmp(a: *char, b: *char) int;
    fn malloc(size: int) *void;
    fn free(ptr: *void) void;
}

// AZ包装
fn string_length(s: string) int {
    return strlen(s.as_ptr());
}
```

**优点**:
- ✅ 实现快速
- ✅ 性能好
- ✅ 稳定可靠

**缺点**:
- ⚠️ 依赖C库
- ⚠️ 跨平台需要注意

### 策略2: 纯AZ实现
```az
// 纯AZ实现
fn string_length(s: string) int {
    var len = 0;
    var i = 0;
    while (s[i] != '\0') {
        len = len + 1;
        i = i + 1;
    }
    return len;
}
```

**优点**:
- ✅ 不依赖外部库
- ✅ 完全控制

**缺点**:
- ❌ 实现慢
- ❌ 可能有bug
- ❌ 性能可能不如C

### 推荐: 混合策略
- 核心功能用C库（性能关键）
- 高级功能用AZ实现（灵活性）

---

## 📝 实现检查清单

### Week 1
- [ ] std.io基础输出
- [ ] std.io基础输入
- [ ] std.io文件读写
- [ ] std.string基础操作
- [ ] std.string查找匹配

### Week 2
- [ ] std.string分割连接
- [ ] std.string转换
- [ ] std.error Result类型
- [ ] std.error Option类型
- [ ] std.collections Vec

### Week 3
- [ ] std.collections HashMap
- [ ] std.collections HashSet
- [ ] std.fs文件操作
- [ ] std.fs目录操作

### Week 4
- [ ] std.fs路径操作
- [ ] std.math基础运算
- [ ] std.math三角函数
- [ ] std.os环境变量

### Week 5
- [ ] std.os进程管理
- [ ] std.time时间类型
- [ ] std.time Duration
- [ ] 文档和测试

### Week 6
- [ ] std.net TCP/UDP
- [ ] std.json解析
- [ ] std.regex正则
- [ ] 完整测试

---

## 🎉 完成标准

### 最小可用（4周）
- ✅ std.io完整
- ✅ std.string完整
- ✅ std.error完整
- ✅ std.collections基础

### 中等可用（7周）
- ✅ 最小可用+
- ✅ std.fs完整
- ✅ std.math完整
- ✅ std.os基础

### 完全可用（10周）
- ✅ 中等可用+
- ✅ std.time完整
- ✅ std.net基础
- ✅ std.json基础

---

<div align="center">

**让我们开始完善AZ标准库！**

目标: 4周内达到最小可用

</div>
