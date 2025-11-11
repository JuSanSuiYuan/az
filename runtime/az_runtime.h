/*
 * AZ Runtime Library - Header
 * 提供AZ语言的运行时支持
 */

#ifndef AZ_RUNTIME_H
#define AZ_RUNTIME_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

// ============================================================================
// 基础类型定义
// ============================================================================

typedef const char* az_string;
typedef int64_t az_int;
typedef double az_float;
typedef bool az_bool;

// ============================================================================
// 内存管理
// ============================================================================

void* az_malloc(size_t size);
void az_free(void* ptr);
void* az_realloc(void* ptr, size_t size);

// ============================================================================
// 字符串操作
// ============================================================================

az_string az_string_concat(az_string a, az_string b);
az_int az_string_length(az_string s);
az_string az_string_substring(az_string s, az_int start, az_int end);
az_bool az_string_equals(az_string a, az_string b);
az_string az_string_from_int(az_int n);
az_string az_string_from_float(az_float f);

// ============================================================================
// 输入输出
// ============================================================================

void az_print(az_string s);
void az_println(az_string s);
void az_eprint(az_string s);
void az_eprintln(az_string s);

az_string az_read_line(void);
az_int az_read_int(void);
az_float az_read_float(void);

// ============================================================================
// 文件操作
// ============================================================================

typedef struct {
    FILE* handle;
    const char* path;
} az_file;

az_file* az_file_open(az_string path, az_string mode);
void az_file_close(az_file* file);
az_string az_file_read_all(az_string path);
void az_file_write_all(az_string path, az_string content);

// ============================================================================
// 动态数组 (Vec)
// ============================================================================

typedef struct {
    void* data;
    size_t length;
    size_t capacity;
    size_t element_size;
} az_vec;

az_vec* az_vec_new(size_t element_size);
void az_vec_push(az_vec* vec, void* element);
void* az_vec_get(az_vec* vec, size_t index);
void az_vec_free(az_vec* vec);

// ============================================================================
// 哈希表 (HashMap)
// ============================================================================

typedef struct az_hashmap_entry {
    void* key;
    void* value;
    struct az_hashmap_entry* next;
} az_hashmap_entry;

typedef struct {
    az_hashmap_entry** buckets;
    size_t bucket_count;
    size_t size;
} az_hashmap;

az_hashmap* az_hashmap_new(void);
void az_hashmap_insert(az_hashmap* map, void* key, void* value);
void* az_hashmap_get(az_hashmap* map, void* key);
void az_hashmap_free(az_hashmap* map);

// ============================================================================
// 错误处理
// ============================================================================

void az_panic(az_string message);
void az_assert(az_bool condition, az_string message);

// ============================================================================
// 数学函数
// ============================================================================

az_int az_abs(az_int x);
az_float az_abs_f(az_float x);
az_int az_min(az_int a, az_int b);
az_int az_max(az_int a, az_int b);
az_float az_sqrt(az_float x);
az_float az_pow(az_float base, az_float exp);

// ============================================================================
// 时间函数
// ============================================================================

az_int az_time_now(void);
void az_sleep_ms(az_int milliseconds);

#endif // AZ_RUNTIME_H
