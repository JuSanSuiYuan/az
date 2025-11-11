# AZ编译器工具

# 主编译器
add_executable(az_compiler
    az_main.cpp
)

target_link_libraries(az_compiler
    PRIVATE
    AZFrontend
    AZSupport
    AZMLIR
    AZBackend
    AZLinker
    AZDebug
    MLIRAnalysis
    MLIRIR
    MLIRParser
    MLIRFuncDialect
    MLIRArithDialect
    MLIRSupport
    LLVMCore
    LLVMSupport
)

# AST可视化工具
add_executable(visualize_ast
    visualize_ast.cpp
)

target_link_libraries(visualize_ast
    PRIVATE
    AZFrontend
    AZSupport
)

# 安装
install(TARGETS az_compiler visualize_ast DESTINATION bin)