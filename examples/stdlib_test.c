#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// 内置函数
void println(const char* str) {
    printf("%s\n", str);
}

void print(const char* str) {
    printf("%s", str);
}

int main(void);

int main(void) {
    {
        println("=== AZ标准库测试 ===");
        println("");
        println("1. 字符串操作:");
        int str1 = "Hello";
        int str2 = " World";
        int combined = az_string_concat(str1, str2);
        println(("  连接: " + combined));
        int upper = az_string_to_upper(combined);
        println(("  大写: " + upper));
        int lower = az_string_to_lower(combined);
        println(("  小写: " + lower));
        println("");
        println("2. 数学运算:");
        int sqrt_val = az_sqrt(16.0);
        println(("  sqrt(16) = " + az_float_to_string(sqrt_val)));
        int pow_val = az_pow(2.0, 3.0);
        println(("  2^3 = " + az_float_to_string(pow_val)));
        int abs_val = az_abs_int((-42));
        println(("  abs(-42) = " + az_int_to_string(abs_val)));
        println("");
        println("3. 文件操作:");
        int content = "Hello from AZ!";
        int write_result = az_write_file("test.txt", content);
        if ((write_result == 0)) {
            {
                println("  文件写入成功");
                if (az_file_exists("test.txt")) {
                    {
                        println("  文件存在确认");
                        int read_content = az_read_file("test.txt");
                        println(("  读取内容: " + read_content));
                    }
                }
            }
        }
        println("");
        println("=== 测试完成 ===");
        return 0;
    }
}
