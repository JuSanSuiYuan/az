cmake_minimum_required(VERSION 3.10)
project(AZCompiler VERSION 0.4.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 查找LLVM
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# 查找MLIR
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION}")

# 查找OpenSSL
find_package(OpenSSL REQUIRED)

# 设置包含目录
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${OPENSSL_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)

# 设置定义
add_definitions(${LLVM_DEFINITIONS})
add_definitions(${MLIR_DEFINITIONS})

# 链接目录
link_directories(${LLVM_LIBRARY_DIRS})

# 添加子目录
add_subdirectory(lib)
add_subdirectory(tools)

# 创建主编译器可执行文件
add_executable(az_compiler
    tools/az_main.cpp
)

target_link_libraries(az_compiler
    PRIVATE
    AZBackend
    AZMLIR
    AZFrontend
    AZSupport
)

# 安装规则
install(TARGETS az_compiler
    RUNTIME DESTINATION bin
)

# 启用测试
enable_testing()
add_subdirectory(test)