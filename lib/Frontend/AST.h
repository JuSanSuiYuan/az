// AZ编译器 - 抽象语法树(AST)定义

#ifndef AZ_FRONTEND_AST_H
#define AZ_FRONTEND_AST_H

#include <vector>
#include <memory>
#include <string>
#include <variant>

namespace az {
namespace frontend {

// 前向声明
class Expr;
class Stmt;
class Pattern;

// 表达式类型
enum class ExprKind {
    IntLiteral,
    FloatLiteral,
    StringLiteral,
    CharLiteral,
    BoolLiteral,
    Identifier,
    Binary,
    Unary,
    Call,
    Index,
    Member,
    ArrayLiteral,
    StructLiteral
};

// 表达式节点
struct Expr {
    ExprKind kind;
    
    // 根据kind不同，使用不同的字段
    union {
        int int_value;
        double float_value;
        bool bool_value;
        char char_value;
    };
    
    std::string string_value;
    std::string name;
    std::string op;
    
    std::unique_ptr<Expr> left;
    std::unique_ptr<Expr> right;
    std::unique_ptr<Expr> operand;
    std::unique_ptr<Expr> callee;
    std::vector<std::unique_ptr<Expr>> arguments;
    std::unique_ptr<Expr> object;
    std::string member;
    std::vector<std::unique_ptr<Expr>> elements;
    std::vector<std::pair<std::string, std::unique_ptr<Expr>>> fields;
    
    Expr(ExprKind k) : kind(k) {}
    virtual ~Expr() = default;
};

// 语句类型
enum class StmtKind {
    Expression,
    VarDecl,
    FuncDecl,
    Return,
    If,
    While,
    For,
    Block,
    Match,
    Import,
    // 新增：结构体和枚举声明
    StructDecl,
    EnumDecl
};

// 结构体字段
struct StructField {
    std::string name;
    std::string type_name;
    bool is_public;
};

// 枚举变体
struct EnumVariant {
    std::string name;
    std::unique_ptr<Expr> value;  // 可选的值
};

// 语句节点
struct Stmt {
    StmtKind kind;
    
    // 通用字段
    std::unique_ptr<Expr> expr;
    std::string name;
    std::string type_name;
    bool is_mutable;
    std::unique_ptr<Expr> initializer;
    std::vector<Param> params;
    std::string return_type;
    std::unique_ptr<Stmt> body;
    std::unique_ptr<Expr> condition;
    std::unique_ptr<Stmt> then_branch;
    std::unique_ptr<Stmt> else_branch;
    std::vector<std::unique_ptr<Stmt>> statements;
    
    // Match语句字段
    std::unique_ptr<Expr> match_expr;
    std::vector<MatchArm> match_arms;
    
    // 导入语句字段
    std::string import_path;
    
    // 新增：结构体和枚举声明字段
    bool is_public;
    std::vector<StructField> struct_fields;
    std::vector<EnumVariant> enum_variants;
    
    Stmt(StmtKind k) : kind(k), is_mutable(false), is_public(false) {}
    virtual ~Stmt() = default;
};

// 函数参数
struct Param {
    std::string name;
    std::string type_name;
};

// 模式类型
enum class PatternKind {
    Literal,      // 字面量模式: 1, "hello"
    Identifier,   // 标识符模式: x
    Wildcard,     // 通配符模式: _
    Or,           // 或模式: 1 | 2 | 3
    Struct,       // 结构体模式: Point { x, y }
    Enum          // 枚举模式: Some(x)
};

// 模式节点
struct Pattern {
    PatternKind kind;
    
    // 根据kind不同，使用不同的字段
    std::unique_ptr<Expr> literal;        // 字面量模式
    std::string name;                     // 标识符模式
    std::vector<std::unique_ptr<Pattern>> patterns;  // 或模式
    std::string struct_name;              // 结构体模式
    std::vector<std::pair<std::string, std::unique_ptr<Pattern>>> struct_fields;  // 结构体字段模式
    std::string enum_name;                // 枚举模式
    std::string variant_name;             // 枚举变体名
    std::unique_ptr<Pattern> enum_pattern; // 枚举变体内的模式
    
    Pattern(PatternKind k) : kind(k) {}
    virtual ~Pattern() = default;
};

// Match分支
struct MatchArm {
    std::unique_ptr<Pattern> pattern;
    std::unique_ptr<Expr> guard;      // 可选的守卫条件
    std::unique_ptr<Stmt> body;
};

// 程序根节点
struct Program {
    std::vector<std::unique_ptr<Stmt>> statements;
    
    Program() = default;
    Program(std::vector<std::unique_ptr<Stmt>> stmts) 
        : statements(std::move(stmts)) {}
};

// 表达式创建辅助函数
std::unique_ptr<Expr> make_int_expr(int value);
std::unique_ptr<Expr> make_float_expr(double value);
std::unique_ptr<Expr> make_string_expr(const std::string& value);
std::unique_ptr<Expr> make_bool_expr(bool value);
std::unique_ptr<Expr> make_char_expr(char value);
std::unique_ptr<Expr> make_identifier_expr(const std::string& name);
std::unique_ptr<Expr> make_binary_expr(const std::string& op, 
                                       std::unique_ptr<Expr> left, 
                                       std::unique_ptr<Expr> right);
std::unique_ptr<Expr> make_unary_expr(const std::string& op, 
                                      std::unique_ptr<Expr> operand);
std::unique_ptr<Expr> make_call_expr(std::unique_ptr<Expr> callee, 
                                     std::vector<std::unique_ptr<Expr>> arguments);

// 语句创建辅助函数
std::unique_ptr<Stmt> make_expr_stmt(std::unique_ptr<Expr> expr);
std::unique_ptr<Stmt> make_var_decl(const std::string& name, 
                                    const std::string& type_name, 
                                    bool is_mutable, 
                                    std::unique_ptr<Expr> initializer);
std::unique_ptr<Stmt> make_func_decl(const std::string& name,
                                     std::vector<Param> params,
                                     const std::string& return_type,
                                     std::unique_ptr<Stmt> body);
std::unique_ptr<Stmt> make_return_stmt(std::unique_ptr<Expr> expr);
std::unique_ptr<Stmt> make_block_stmt(std::vector<std::unique_ptr<Stmt>> statements);
std::unique_ptr<Stmt> make_if_stmt(std::unique_ptr<Expr> condition,
                                   std::unique_ptr<Stmt> then_branch,
                                   std::unique_ptr<Stmt> else_branch);
std::unique_ptr<Stmt> make_while_stmt(std::unique_ptr<Expr> condition,
                                      std::unique_ptr<Stmt> body);
std::unique_ptr<Stmt> make_for_stmt(const std::string& var_name,
                                    std::unique_ptr<Expr> initializer,
                                    std::unique_ptr<Expr> condition,
                                    std::unique_ptr<Expr> increment,
                                    std::unique_ptr<Stmt> body);
std::unique_ptr<Stmt> make_match_stmt(std::unique_ptr<Expr> expr,
                                      std::vector<MatchArm> arms);
std::unique_ptr<Stmt> make_import_stmt(const std::string& path);

// 新增：结构体和枚举声明创建辅助函数
std::unique_ptr<Stmt> make_struct_decl(const std::string& name,
                                       std::vector<StructField> fields,
                                       bool is_public);
std::unique_ptr<Stmt> make_enum_decl(const std::string& name,
                                     std::vector<EnumVariant> variants,
                                     bool is_public);

// 模式创建辅助函数
std::unique_ptr<Pattern> make_literal_pattern(std::unique_ptr<Expr> literal);
std::unique_ptr<Pattern> make_identifier_pattern(const std::string& name);
std::unique_ptr<Pattern> make_wildcard_pattern();
std::unique_ptr<Pattern> make_or_pattern(std::vector<std::unique_ptr<Pattern>> patterns);
std::unique_ptr<Pattern> make_struct_pattern(const std::string& struct_name,
                                             std::vector<std::pair<std::string, std::unique_ptr<Pattern>>> fields);
std::unique_ptr<Pattern> make_enum_pattern(const std::string& enum_name,
                                           const std::string& variant_name,
                                           std::unique_ptr<Pattern> pattern);

// Match分支创建辅助函数
MatchArm make_match_arm(std::unique_ptr<Pattern> pattern,
                        std::unique_ptr<Expr> guard,
                        std::unique_ptr<Stmt> body);

} // namespace frontend
} // namespace az

#endif // AZ_FRONTEND_AST_H