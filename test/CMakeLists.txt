# AZ编译器测试

enable_testing()

# 查找GTest
find_package(GTest REQUIRED)

# 前端测试
add_executable(az_tests
    lexer_test.cpp
    parser_test.cpp
    sema_test.cpp
)

target_link_libraries(az_tests
    PRIVATE
    AZFrontend
    AZSupport
    LLVMSupport
)

# 添加前端测试
add_test(NAME LexerTest COMMAND az_tests lexer)
add_test(NAME ParserTest COMMAND az_tests parser)
add_test(NAME SemaTest COMMAND az_tests sema)

# 后端测试 - MLIR降级
add_executable(mlir_lowering_test
    Backend/MLIRLoweringTest.cpp
)

target_link_libraries(mlir_lowering_test
    PRIVATE
    AZBackend
    AZSupport
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME MLIRLoweringTest COMMAND mlir_lowering_test)

# 后端测试 - 优化器
add_executable(optimizer_test
    Backend/OptimizerTest.cpp
)

target_link_libraries(optimizer_test
    PRIVATE
    AZBackend
    AZSupport
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME OptimizerTest COMMAND optimizer_test)

# 后端测试 - 代码生成器
add_executable(codegen_test
    Backend/CodeGeneratorTest.cpp
)

target_link_libraries(codegen_test
    PRIVATE
    AZBackend
    AZSupport
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME CodeGeneratorTest COMMAND codegen_test)

# 后端集成测试
add_executable(integration_test
    Backend/IntegrationTest.cpp
)

target_link_libraries(integration_test
    PRIVATE
    AZBackend
    AZSupport
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME IntegrationTest COMMAND integration_test)

# 后端测试 - 调试信息
add_executable(debuginfo_test
    Backend/DebugInfoTest.cpp
)

target_link_libraries(debuginfo_test
    PRIVATE
    AZBackend
    AZSupport
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME DebugInfoTest COMMAND debuginfo_test)

# 后端测试 - JIT编译器
add_executable(jit_test
    Backend/JITTest.cpp
)

target_link_libraries(jit_test
    PRIVATE
    AZBackend
    AZSupport
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME JITTest COMMAND jit_test)

# 后端测试 - 缓存集成
add_executable(cache_integration_test
    Backend/CacheIntegrationTest.cpp
)

target_link_libraries(cache_integration_test
    PRIVATE
    AZBackend
    AZSupport
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME CacheIntegrationTest COMMAND cache_integration_test)

# 集成测试 - 编译示例程序
add_test(NAME CompileHello
    COMMAND az ${CMAKE_SOURCE_DIR}/examples/hello.az
)

add_test(NAME CompileVariables
    COMMAND az ${CMAKE_SOURCE_DIR}/examples/variables.az
)

add_test(NAME CompileFunctions
    COMMAND az ${CMAKE_SOURCE_DIR}/examples/functions.az
)

add_test(NAME CompileControlFlow
    COMMAND az ${CMAKE_SOURCE_DIR}/examples/control_flow.az
)

add_test(NAME CompileFibonacci
    COMMAND az ${CMAKE_SOURCE_DIR}/examples/fibonacci.az
)
