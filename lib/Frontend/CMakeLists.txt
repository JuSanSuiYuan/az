// AZ编译器库

# 前端库
add_library(AZFrontend
    Lexer.cpp
    Parser.cpp
    Token.cpp
    Sema.cpp
    AST.cpp
    ASTVisualizer.cpp
)

target_include_directories(AZFrontend PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(AZFrontend
    LLVMSupport
)