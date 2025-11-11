/*
 * AZ Runtime Library - Implementation
 * 实现AZ语言的运行时支持
 */

#include "az_runtime.h"
#include <math.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

// ============================================================================
// 内存管理
// ============================================================================

void* az_malloc(size_t size) {
    void* ptr = malloc(size);
    if (ptr == NULL) {
        az_panic("Out of memory");
    }
    return ptr;
}

void az_free(void* ptr) {
    if (ptr != NULL) {
        free(ptr);
    }
}

void* az_realloc(void* ptr, size_t size) {
    void* new_ptr = realloc(ptr, size);
    if (new_ptr == NULL && size > 0) {
        az_panic("Out of memory");
    }
    return new_ptr;
}

// ============================================================================
// 字符串操作
// ============================================================================

az_string az_string_concat(az_string a, az_string b) {
    size_t len_a = strlen(a);
    size_t len_b = strlen(b);
    char* result = (char*)az_malloc(len_a + len_b + 1);
    
    strcpy(result, a);
    strcat(result, b);
    
    return result;
}

az_int az_string_length(az_string s) {
    return (az_int)strlen(s);
}

az_string az_string_substring(az_string s, az_int start, az_int end) {
    size_t len = strlen(s);
    
    if (start < 0) start = 0;
    if (end > (az_int)len) end = len;
    if (start >= end) return "";
    
    size_t sub_len = end - start;
    char* result = (char*)az_malloc(sub_len + 1);
    
    strncpy(result, s + start, sub_len);
    result[sub_len] = '\0';
    
    return result;
}

az_bool az_string_equals(az_string a, az_string b) {
    return strcmp(a, b) == 0;
}

az_string az_string_from_int(az_int n) {
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "%lld", (long long)n);
    
    size_t len = strlen(buffer);
    char* result = (char*)az_malloc(len + 1);
    strcpy(result, buffer);
    
    return result;
}

az_string az_string_from_float(az_float f) {
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "%f", f);
    
    size_t len = strlen(buffer);
    char* result = (char*)az_malloc(len + 1);
    strcpy(result, buffer);
    
    return result;
}

// ============================================================================
// 输入输出
// ============================================================================

void az_print(az_string s) {
    printf("%s", s);
    fflush(stdout);
}

void az_println(az_string s) {
    printf("%s\n", s);
    fflush(stdout);
}

void az_eprint(az_string s) {
    fprintf(stderr, "%s", s);
    fflush(stderr);
}

void az_eprintln(az_string s) {
    fprintf(stderr, "%s\n", s);
    fflush(stderr);
}

az_string az_read_line(void) {
    char buffer[1024];
    if (fgets(buffer, sizeof(buffer), stdin) != NULL) {
        // 移除换行符
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len-1] == '\n') {
            buffer[len-1] = '\0';
        }
        
        char* result = (char*)az_malloc(strlen(buffer) + 1);
        strcpy(result, buffer);
        return result;
    }
    return "";
}

az_int az_read_int(void) {
    az_int n;
    scanf("%lld", (long long*)&n);
    return n;
}

az_float az_read_float(void) {
    az_float f;
    scanf("%lf", &f);
    return f;
}

// ============================================================================
// 文件操作
// ============================================================================

az_file* az_file_open(az_string path, az_string mode) {
    FILE* handle = fopen(path, mode);
    if (handle == NULL) {
        return NULL;
    }
    
    az_file* file = (az_file*)az_malloc(sizeof(az_file));
    file->handle = handle;
    file->path = path;
    
    return file;
}

void az_file_close(az_file* file) {
    if (file != NULL && file->handle != NULL) {
        fclose(file->handle);
        az_free(file);
    }
}

az_string az_file_read_all(az_string path) {
    FILE* file = fopen(path, "r");
    if (file == NULL) {
        return NULL;
    }
    
    // 获取文件大小
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    // 读取内容
    char* buffer = (char*)az_malloc(size + 1);
    size_t read = fread(buffer, 1, size, file);
    buffer[read] = '\0';
    
    fclose(file);
    return buffer;
}

void az_file_write_all(az_string path, az_string content) {
    FILE* file = fopen(path, "w");
    if (file == NULL) {
        az_panic("Cannot open file for writing");
    }
    
    fwrite(content, 1, strlen(content), file);
    fclose(file);
}

// ============================================================================
// 动态数组 (Vec)
// ============================================================================

az_vec* az_vec_new(size_t element_size) {
    az_vec* vec = (az_vec*)az_malloc(sizeof(az_vec));
    vec->data = NULL;
    vec->length = 0;
    vec->capacity = 0;
    vec->element_size = element_size;
    return vec;
}

static void az_vec_grow(az_vec* vec) {
    size_t new_capacity = vec->capacity == 0 ? 4 : vec->capacity * 2;
    vec->data = az_realloc(vec->data, new_capacity * vec->element_size);
    vec->capacity = new_capacity;
}

void az_vec_push(az_vec* vec, void* element) {
    if (vec->length >= vec->capacity) {
        az_vec_grow(vec);
    }
    
    char* dest = (char*)vec->data + (vec->length * vec->element_size);
    memcpy(dest, element, vec->element_size);
    vec->length++;
}

void* az_vec_get(az_vec* vec, size_t index) {
    if (index >= vec->length) {
        az_panic("Vec index out of bounds");
    }
    
    return (char*)vec->data + (index * vec->element_size);
}

void az_vec_free(az_vec* vec) {
    if (vec != NULL) {
        if (vec->data != NULL) {
            az_free(vec->data);
        }
        az_free(vec);
    }
}

// ============================================================================
// 哈希表 (HashMap)
// ============================================================================

static size_t az_hash(void* key) {
    // 简单的哈希函数
    size_t h = (size_t)key;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

az_hashmap* az_hashmap_new(void) {
    az_hashmap* map = (az_hashmap*)az_malloc(sizeof(az_hashmap));
    map->bucket_count = 16;
    map->buckets = (az_hashmap_entry**)az_malloc(map->bucket_count * sizeof(az_hashmap_entry*));
    memset(map->buckets, 0, map->bucket_count * sizeof(az_hashmap_entry*));
    map->size = 0;
    return map;
}

void az_hashmap_insert(az_hashmap* map, void* key, void* value) {
    size_t hash = az_hash(key);
    size_t index = hash % map->bucket_count;
    
    // 查找是否已存在
    az_hashmap_entry* entry = map->buckets[index];
    while (entry != NULL) {
        if (entry->key == key) {
            entry->value = value;
            return;
        }
        entry = entry->next;
    }
    
    // 创建新条目
    az_hashmap_entry* new_entry = (az_hashmap_entry*)az_malloc(sizeof(az_hashmap_entry));
    new_entry->key = key;
    new_entry->value = value;
    new_entry->next = map->buckets[index];
    map->buckets[index] = new_entry;
    map->size++;
}

void* az_hashmap_get(az_hashmap* map, void* key) {
    size_t hash = az_hash(key);
    size_t index = hash % map->bucket_count;
    
    az_hashmap_entry* entry = map->buckets[index];
    while (entry != NULL) {
        if (entry->key == key) {
            return entry->value;
        }
        entry = entry->next;
    }
    
    return NULL;
}

void az_hashmap_free(az_hashmap* map) {
    if (map != NULL) {
        for (size_t i = 0; i < map->bucket_count; i++) {
            az_hashmap_entry* entry = map->buckets[i];
            while (entry != NULL) {
                az_hashmap_entry* next = entry->next;
                az_free(entry);
                entry = next;
            }
        }
        az_free(map->buckets);
        az_free(map);
    }
}

// ============================================================================
// 错误处理
// ============================================================================

void az_panic(az_string message) {
    fprintf(stderr, "[PANIC] %s\n", message);
    exit(1);
}

void az_assert(az_bool condition, az_string message) {
    if (!condition) {
        fprintf(stderr, "[ASSERTION FAILED] %s\n", message);
        exit(1);
    }
}

// ============================================================================
// 数学函数
// ============================================================================

az_int az_abs(az_int x) {
    return x < 0 ? -x : x;
}

az_float az_abs_f(az_float x) {
    return fabs(x);
}

az_int az_min(az_int a, az_int b) {
    return a < b ? a : b;
}

az_int az_max(az_int a, az_int b) {
    return a > b ? a : b;
}

az_float az_sqrt(az_float x) {
    return sqrt(x);
}

az_float az_pow(az_float base, az_float exp) {
    return pow(base, exp);
}

// ============================================================================
// 时间函数
// ============================================================================

az_int az_time_now(void) {
    return (az_int)time(NULL);
}

void az_sleep_ms(az_int milliseconds) {
#ifdef _WIN32
    Sleep((DWORD)milliseconds);
#else
    usleep(milliseconds * 1000);
#endif
}
