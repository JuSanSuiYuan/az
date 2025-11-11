// AZ语言运行时标准库
// 提供基础的标准库功能

// 禁用Windows安全警告
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#define _CRT_NONSTDC_NO_DEPRECATE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>
#include <math.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#include <io.h>
#include <process.h>
#include <sys/stat.h>
#define mkdir(path, mode) _mkdir(path)
#define rmdir(path) _rmdir(path)
#define access(path, mode) _access(path, mode)
#define strdup _strdup
#else
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#endif

// ============================================================================
// std.io - 输入输出
// ============================================================================

// 已在生成的代码中定义
// void println(const char* str);
// void print(const char* str);

char* az_read_line() {
    char* line = (char*)malloc(1024);
    if (!line) return NULL;
    if (fgets(line, 1024, stdin) == NULL) {
        free(line);
        return NULL;
    }
    // 移除换行符
    size_t len = strlen(line);
    if (len > 0 && line[len - 1] == '\n') {
        line[len - 1] = '\0';
    }
    return line;
}

// ============================================================================
// std.string - 字符串操作
// ============================================================================

char* az_string_concat(const char* a, const char* b) {
    if (!a || !b) return NULL;
    size_t len = strlen(a) + strlen(b) + 1;
    char* result = (char*)malloc(len);
    if (!result) return NULL;
    strcpy(result, a);
    strcat(result, b);
    return result;
}

int az_string_length(const char* str) {
    if (!str) return 0;
    return (int)strlen(str);
}

char* az_string_substring(const char* str, int start, int end) {
    if (!str) return NULL;
    int len = end - start;
    if (len <= 0) return NULL;
    char* result = (char*)malloc(len + 1);
    if (!result) return NULL;
    strncpy(result, str + start, len);
    result[len] = '\0';
    return result;
}

bool az_string_equals(const char* a, const char* b) {
    if (!a || !b) return false;
    return strcmp(a, b) == 0;
}

char* az_string_to_upper(const char* str) {
    if (!str) return NULL;
    int len = strlen(str);
    char* result = (char*)malloc(len + 1);
    if (!result) return NULL;
    for (int i = 0; i < len; i++) {
        result[i] = toupper(str[i]);
    }
    result[len] = '\0';
    return result;
}

char* az_string_to_lower(const char* str) {
    if (!str) return NULL;
    int len = strlen(str);
    char* result = (char*)malloc(len + 1);
    if (!result) return NULL;
    for (int i = 0; i < len; i++) {
        result[i] = tolower(str[i]);
    }
    result[len] = '\0';
    return result;
}

// ============================================================================
// std.fs - 文件系统
// ============================================================================

char* az_read_file(const char* path) {
    if (!path) return NULL;
    
    FILE* file = fopen(path, "rb");
    if (!file) return NULL;
    
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    char* content = (char*)malloc(size + 1);
    if (!content) {
        fclose(file);
        return NULL;
    }
    
    fread(content, 1, size, file);
    content[size] = '\0';
    
    fclose(file);
    return content;
}

int az_write_file(const char* path, const char* content) {
    if (!path || !content) return -1;
    
    FILE* file = fopen(path, "w");
    if (!file) return -1;
    
    fputs(content, file);
    fclose(file);
    return 0;
}

bool az_file_exists(const char* path) {
    if (!path) return false;
    FILE* file = fopen(path, "r");
    if (file) {
        fclose(file);
        return true;
    }
    return false;
}

// ============================================================================
// std.collections - 动态数组
// ============================================================================

typedef struct {
    void** data;
    int size;
    int capacity;
} AzVec;

AzVec* az_vec_new() {
    AzVec* vec = (AzVec*)malloc(sizeof(AzVec));
    if (!vec) return NULL;
    vec->data = (void**)malloc(sizeof(void*) * 10);
    if (!vec->data) {
        free(vec);
        return NULL;
    }
    vec->size = 0;
    vec->capacity = 10;
    return vec;
}

void az_vec_push(AzVec* vec, void* item) {
    if (!vec) return;
    if (vec->size >= vec->capacity) {
        vec->capacity *= 2;
        void** new_data = (void**)realloc(vec->data, sizeof(void*) * vec->capacity);
        if (!new_data) return;
        vec->data = new_data;
    }
    vec->data[vec->size++] = item;
}

void* az_vec_get(AzVec* vec, int index) {
    if (!vec || index < 0 || index >= vec->size) return NULL;
    return vec->data[index];
}

int az_vec_len(AzVec* vec) {
    if (!vec) return 0;
    return vec->size;
}

void az_vec_free(AzVec* vec) {
    if (!vec) return;
    free(vec->data);
    free(vec);
}

// ============================================================================
// std.mem - 内存管理
// ============================================================================

void* az_malloc(size_t size) {
    return malloc(size);
}

void az_free(void* ptr) {
    free(ptr);
}

void* az_realloc(void* ptr, size_t size) {
    return realloc(ptr, size);
}

// ============================================================================
// std.math - 数学函数
// ============================================================================

double az_sqrt(double x) {
    return sqrt(x);
}

double az_pow(double x, double y) {
    return pow(x, y);
}

double az_abs(double x) {
    return fabs(x);
}

int az_abs_int(int x) {
    return abs(x);
}

// ============================================================================
// 工具函数
// ============================================================================

// 整数转字符串
char* az_int_to_string(int value) {
    char* str = (char*)malloc(32);
    if (!str) return NULL;
    snprintf(str, 32, "%d", value);
    return str;
}

// 字符串转整数
int az_string_to_int(const char* str) {
    if (!str) return 0;
    return atoi(str);
}

// 浮点数转字符串
char* az_float_to_string(double value) {
    char* str = (char*)malloc(32);
    if (!str) return NULL;
    snprintf(str, 32, "%f", value);
    return str;
}

// 字符串转浮点数
double az_string_to_float(const char* str) {
    if (!str) return 0.0;
    return atof(str);
}

// ============================================================================
// std.string - 高级字符串操作
// ============================================================================

// 字符串查找
int az_string_find(const char* str, const char* sub) {
    if (!str || !sub) return -1;
    const char* pos = strstr(str, sub);
    if (!pos) return -1;
    return (int)(pos - str);
}

// 字符串包含
bool az_string_contains(const char* str, const char* sub) {
    return az_string_find(str, sub) != -1;
}

// 字符串开头匹配
bool az_string_starts_with(const char* str, const char* prefix) {
    if (!str || !prefix) return false;
    size_t prefix_len = strlen(prefix);
    size_t str_len = strlen(str);
    if (prefix_len > str_len) return false;
    return strncmp(str, prefix, prefix_len) == 0;
}

// 字符串结尾匹配
bool az_string_ends_with(const char* str, const char* suffix) {
    if (!str || !suffix) return false;
    size_t suffix_len = strlen(suffix);
    size_t str_len = strlen(str);
    if (suffix_len > str_len) return false;
    return strcmp(str + str_len - suffix_len, suffix) == 0;
}

// 字符串去除空白
char* az_string_trim(const char* str) {
    if (!str) return NULL;
    
    // 跳过开头空白
    while (*str && isspace(*str)) str++;
    
    if (*str == '\0') {
        char* result = (char*)malloc(1);
        result[0] = '\0';
        return result;
    }
    
    // 找到结尾
    const char* end = str + strlen(str) - 1;
    while (end > str && isspace(*end)) end--;
    
    // 复制结果
    size_t len = end - str + 1;
    char* result = (char*)malloc(len + 1);
    if (!result) return NULL;
    strncpy(result, str, len);
    result[len] = '\0';
    return result;
}

// 字符串替换
char* az_string_replace(const char* str, const char* old, const char* new_str) {
    if (!str || !old || !new_str) return NULL;
    
    size_t old_len = strlen(old);
    size_t new_len = strlen(new_str);
    
    // 计算需要的空间
    int count = 0;
    const char* p = str;
    while ((p = strstr(p, old)) != NULL) {
        count++;
        p += old_len;
    }
    
    if (count == 0) {
        return strdup(str);
    }
    
    // 分配新字符串
    size_t result_len = strlen(str) + count * (new_len - old_len);
    char* result = (char*)malloc(result_len + 1);
    if (!result) return NULL;
    
    // 执行替换
    char* dst = result;
    p = str;
    while (*p) {
        const char* q = strstr(p, old);
        if (q == NULL) {
            strcpy(dst, p);
            break;
        }
        
        size_t len = q - p;
        strncpy(dst, p, len);
        dst += len;
        strcpy(dst, new_str);
        dst += new_len;
        p = q + old_len;
    }
    
    return result;
}

// 字符串重复
char* az_string_repeat(const char* str, int count) {
    if (!str || count <= 0) return NULL;
    
    size_t len = strlen(str);
    size_t total_len = len * count;
    char* result = (char*)malloc(total_len + 1);
    if (!result) return NULL;
    
    char* p = result;
    for (int i = 0; i < count; i++) {
        strcpy(p, str);
        p += len;
    }
    
    return result;
}

// 字符串反转
char* az_string_reverse(const char* str) {
    if (!str) return NULL;
    
    size_t len = strlen(str);
    char* result = (char*)malloc(len + 1);
    if (!result) return NULL;
    
    for (size_t i = 0; i < len; i++) {
        result[i] = str[len - 1 - i];
    }
    result[len] = '\0';
    
    return result;
}

// 字符串分割
char** az_string_split(const char* str, const char* delimiter, int* count) {
    if (!str || !delimiter || !count) return NULL;
    
    // 计算分割数量
    *count = 1;
    const char* p = str;
    while ((p = strstr(p, delimiter)) != NULL) {
        (*count)++;
        p += strlen(delimiter);
    }
    
    // 分配结果数组
    char** result = (char**)malloc(sizeof(char*) * (*count));
    if (!result) return NULL;
    
    // 执行分割
    char* str_copy = strdup(str);
    char* token = strtok(str_copy, delimiter);
    int i = 0;
    while (token != NULL && i < *count) {
        result[i++] = strdup(token);
        token = strtok(NULL, delimiter);
    }
    
    free(str_copy);
    return result;
}

// ============================================================================
// std.math - 扩展数学函数
// ============================================================================

#include <math.h>

double az_sin(double x) { return sin(x); }
double az_cos(double x) { return cos(x); }
double az_tan(double x) { return tan(x); }
double az_asin(double x) { return asin(x); }
double az_acos(double x) { return acos(x); }
double az_atan(double x) { return atan(x); }
double az_atan2(double y, double x) { return atan2(y, x); }

double az_exp(double x) { return exp(x); }
double az_log(double x) { return log(x); }
double az_log10(double x) { return log10(x); }
double az_log2(double x) { return log2(x); }

double az_floor(double x) { return floor(x); }
double az_ceil(double x) { return ceil(x); }
double az_round(double x) { return round(x); }
double az_trunc(double x) { return trunc(x); }

int az_max(int a, int b) { return a > b ? a : b; }
int az_min(int a, int b) { return a < b ? a : b; }
double az_max_float(double a, double b) { return a > b ? a : b; }
double az_min_float(double a, double b) { return a < b ? a : b; }

int az_clamp(int value, int min_val, int max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

double az_clamp_float(double value, double min_val, double max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

// ============================================================================
// std.fs - 扩展文件系统操作
// ============================================================================

// 追加到文件
int az_append_file(const char* path, const char* content) {
    if (!path || !content) return -1;
    
    FILE* file = fopen(path, "a");
    if (!file) return -1;
    
    fputs(content, file);
    fclose(file);
    return 0;
}

// 创建目录
int az_create_dir(const char* path) {
    if (!path) return -1;
    return mkdir(path, 0755);
}

// 删除目录
int az_remove_dir(const char* path) {
    if (!path) return -1;
    return rmdir(path);
}

// 删除文件
int az_remove_file(const char* path) {
    if (!path) return -1;
    return remove(path);
}

// 重命名文件
int az_rename_file(const char* old_path, const char* new_path) {
    if (!old_path || !new_path) return -1;
    return rename(old_path, new_path);
}

// 获取文件大小
long az_file_size(const char* path) {
    if (!path) return -1;
    
#ifdef _WIN32
    struct _stat st;
    if (_stat(path, &st) != 0) return -1;
    return st.st_size;
#else
    struct stat st;
    if (stat(path, &st) != 0) return -1;
    return st.st_size;
#endif
}

// 检查是否是目录
bool az_is_dir(const char* path) {
    if (!path) return false;
    
#ifdef _WIN32
    DWORD attrs = GetFileAttributesA(path);
    if (attrs == INVALID_FILE_ATTRIBUTES) return false;
    return (attrs & FILE_ATTRIBUTE_DIRECTORY) != 0;
#else
    struct stat st;
    if (stat(path, &st) != 0) return false;
    return S_ISDIR(st.st_mode);
#endif
}

// 检查是否是文件
bool az_is_file(const char* path) {
    if (!path) return false;
    
#ifdef _WIN32
    DWORD attrs = GetFileAttributesA(path);
    if (attrs == INVALID_FILE_ATTRIBUTES) return false;
    return (attrs & FILE_ATTRIBUTE_DIRECTORY) == 0;
#else
    struct stat st;
    if (stat(path, &st) != 0) return false;
    return S_ISREG(st.st_mode);
#endif
}

// ============================================================================
// std.os - 操作系统接口
// ============================================================================

// 获取环境变量
char* az_getenv(const char* name) {
    if (!name) return NULL;
    char* value = getenv(name);
    return value ? strdup(value) : NULL;
}

// 设置环境变量
int az_setenv(const char* name, const char* value) {
    if (!name || !value) return -1;
#ifdef _WIN32
    return _putenv_s(name, value);
#else
    return setenv(name, value, 1);
#endif
}

// 获取当前时间（Unix时间戳）
long az_time_now() {
    return (long)time(NULL);
}

// 睡眠（毫秒）
void az_sleep_millis(int millis) {
#ifdef _WIN32
    Sleep((DWORD)millis);
#else
    usleep(millis * 1000);
#endif
}

// 执行系统命令
int az_system(const char* command) {
    if (!command) return -1;
    return system(command);
}

// 获取进程ID
int az_getpid() {
#ifdef _WIN32
    return (int)GetCurrentProcessId();
#else
    return getpid();
#endif
}

// ============================================================================
// std.collections - 扩展集合操作
// ============================================================================

// Vec操作
void az_vec_insert(AzVec* vec, int index, void* item) {
    if (!vec || index < 0 || index > vec->size) return;
    
    if (vec->size >= vec->capacity) {
        vec->capacity *= 2;
        void** new_data = (void**)realloc(vec->data, sizeof(void*) * vec->capacity);
        if (!new_data) return;
        vec->data = new_data;
    }
    
    // 移动元素
    for (int i = vec->size; i > index; i--) {
        vec->data[i] = vec->data[i - 1];
    }
    
    vec->data[index] = item;
    vec->size++;
}

void* az_vec_remove(AzVec* vec, int index) {
    if (!vec || index < 0 || index >= vec->size) return NULL;
    
    void* item = vec->data[index];
    
    // 移动元素
    for (int i = index; i < vec->size - 1; i++) {
        vec->data[i] = vec->data[i + 1];
    }
    
    vec->size--;
    return item;
}

void az_vec_clear(AzVec* vec) {
    if (!vec) return;
    vec->size = 0;
}

// ============================================================================
// 随机数生成
// ============================================================================

// 设置随机数种子
void az_srand(unsigned int seed) {
    srand(seed);
}

// 生成随机数
int az_rand() {
    return rand();
}

// 生成范围内的随机数
int az_rand_range(int min, int max) {
    if (min >= max) return min;
    return min + rand() % (max - min);
}

// 生成随机浮点数 [0, 1)
double az_rand_float() {
    return (double)rand() / (double)RAND_MAX;
}
