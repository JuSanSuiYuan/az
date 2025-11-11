// AZ编译器 - 抽象语法树(AST)实现

#include "AZ/Frontend/AST.h"
#include <memory>

namespace az {
namespace frontend {

// 表达式创建辅助函数
std::unique_ptr<Expr> make_int_expr(int value) {
    auto expr = std::make_unique<Expr>(ExprKind::IntLiteral);
    expr->int_value = value;
    return expr;
}

std::unique_ptr<Expr> make_float_expr(double value) {
    auto expr = std::make_unique<Expr>(ExprKind::FloatLiteral);
    expr->float_value = value;
    return expr;
}

std::unique_ptr<Expr> make_string_expr(const std::string& value) {
    auto expr = std::make_unique<Expr>(ExprKind::StringLiteral);
    expr->string_value = value;
    return expr;
}

std::unique_ptr<Expr> make_bool_expr(bool value) {
    auto expr = std::make_unique<Expr>(ExprKind::BoolLiteral);
    expr->bool_value = value;
    return expr;
}

std::unique_ptr<Expr> make_char_expr(char value) {
    auto expr = std::make_unique<Expr>(ExprKind::CharLiteral);
    expr->char_value = value;
    return expr;
}

std::unique_ptr<Expr> make_identifier_expr(const std::string& name) {
    auto expr = std::make_unique<Expr>(ExprKind::Identifier);
    expr->name = name;
    return expr;
}

std::unique_ptr<Expr> make_binary_expr(const std::string& op, 
                                       std::unique_ptr<Expr> left, 
                                       std::unique_ptr<Expr> right) {
    auto expr = std::make_unique<Expr>(ExprKind::Binary);
    expr->op = op;
    expr->left = std::move(left);
    expr->right = std::move(right);
    return expr;
}

std::unique_ptr<Expr> make_unary_expr(const std::string& op, 
                                      std::unique_ptr<Expr> operand) {
    auto expr = std::make_unique<Expr>(ExprKind::Unary);
    expr->op = op;
    expr->operand = std::move(operand);
    return expr;
}

std::unique_ptr<Expr> make_call_expr(std::unique_ptr<Expr> callee, 
                                     std::vector<std::unique_ptr<Expr>> arguments) {
    auto expr = std::make_unique<Expr>(ExprKind::Call);
    expr->callee = std::move(callee);
    expr->arguments = std::move(arguments);
    return expr;
}

// 新增：数组字面量表达式创建辅助函数
std::unique_ptr<Expr> make_array_literal_expr(std::vector<std::unique_ptr<Expr>> elements) {
    auto expr = std::make_unique<Expr>(ExprKind::ArrayLiteral);
    expr->elements = std::move(elements);
    return expr;
}

// 新增：结构体字面量表达式创建辅助函数
std::unique_ptr<Expr> make_struct_literal_expr(const std::string& struct_name,
                                              std::vector<std::pair<std::string, std::unique_ptr<Expr>>> fields) {
    auto expr = std::make_unique<Expr>(ExprKind::StructLiteral);
    expr->string_value = struct_name;
    expr->fields = std::move(fields);
    return expr;
}

// 语句创建辅助函数
std::unique_ptr<Stmt> make_expr_stmt(std::unique_ptr<Expr> expr) {
    auto stmt = std::make_unique<Stmt>(StmtKind::Expression);
    stmt->expr = std::move(expr);
    return stmt;
}

std::unique_ptr<Stmt> make_var_decl(const std::string& name, 
                                    const std::string& type_name, 
                                    bool is_mutable, 
                                    std::unique_ptr<Expr> initializer) {
    auto stmt = std::make_unique<Stmt>(StmtKind::VarDecl);
    stmt->name = name;
    stmt->type_name = type_name;
    stmt->is_mutable = is_mutable;
    stmt->initializer = std::move(initializer);
    return stmt;
}

std::unique_ptr<Stmt> make_func_decl(const std::string& name,
                                     std::vector<Param> params,
                                     const std::string& return_type,
                                     std::unique_ptr<Stmt> body) {
    auto stmt = std::make_unique<Stmt>(StmtKind::FuncDecl);
    stmt->name = name;
    stmt->params = std::move(params);
    stmt->return_type = return_type;
    stmt->body = std::move(body);
    return stmt;
}

std::unique_ptr<Stmt> make_return_stmt(std::unique_ptr<Expr> expr) {
    auto stmt = std::make_unique<Stmt>(StmtKind::Return);
    stmt->expr = std::move(expr);
    return stmt;
}

std::unique_ptr<Stmt> make_block_stmt(std::vector<std::unique_ptr<Stmt>> statements) {
    auto stmt = std::make_unique<Stmt>(StmtKind::Block);
    stmt->statements = std::move(statements);
    return stmt;
}

std::unique_ptr<Stmt> make_if_stmt(std::unique_ptr<Expr> condition,
                                   std::unique_ptr<Stmt> then_branch,
                                   std::unique_ptr<Stmt> else_branch) {
    auto stmt = std::make_unique<Stmt>(StmtKind::If);
    stmt->condition = std::move(condition);
    stmt->then_branch = std::move(then_branch);
    stmt->else_branch = std::move(else_branch);
    return stmt;
}

std::unique_ptr<Stmt> make_while_stmt(std::unique_ptr<Expr> condition,
                                      std::unique_ptr<Stmt> body) {
    auto stmt = std::make_unique<Stmt>(StmtKind::While);
    stmt->condition = std::move(condition);
    stmt->body = std::move(body);
    return stmt;
}

std::unique_ptr<Stmt> make_for_stmt(const std::string& var_name,
                                    std::unique_ptr<Expr> initializer,
                                    std::unique_ptr<Expr> condition,
                                    std::unique_ptr<Expr> increment,
                                    std::unique_ptr<Stmt> body) {
    // For语句可以转换为while语句
    // 这里简化实现，实际应该创建专门的ForStmt节点
    auto stmt = std::make_unique<Stmt>(StmtKind::For);
    // 可以添加专门的字段来存储for循环的组件
    return stmt;
}

std::unique_ptr<Stmt> make_match_stmt(std::unique_ptr<Expr> expr,
                                      std::vector<MatchArm> arms) {
    auto stmt = std::make_unique<Stmt>(StmtKind::Match);
    stmt->match_expr = std::move(expr);
    stmt->match_arms = std::move(arms);
    return stmt;
}

std::unique_ptr<Stmt> make_import_stmt(const std::string& path) {
    auto stmt = std::make_unique<Stmt>(StmtKind::Import);
    stmt->import_path = path;
    return stmt;
}

// 新增：结构体和枚举声明创建辅助函数
std::unique_ptr<Stmt> make_struct_decl(const std::string& name,
                                       std::vector<StructField> fields,
                                       bool is_public) {
    auto stmt = std::make_unique<Stmt>(StmtKind::StructDecl);
    stmt->name = name;
    stmt->struct_fields = std::move(fields);
    stmt->is_public = is_public;
    return stmt;
}

std::unique_ptr<Stmt> make_enum_decl(const std::string& name,
                                     std::vector<EnumVariant> variants,
                                     bool is_public) {
    auto stmt = std::make_unique<Stmt>(StmtKind::EnumDecl);
    stmt->name = name;
    stmt->enum_variants = std::move(variants);
    stmt->is_public = is_public;
    return stmt;
}

// 模式创建辅助函数
std::unique_ptr<Pattern> make_literal_pattern(std::unique_ptr<Expr> literal) {
    auto pattern = std::make_unique<Pattern>(PatternKind::Literal);
    pattern->literal = std::move(literal);
    return pattern;
}

std::unique_ptr<Pattern> make_identifier_pattern(const std::string& name) {
    auto pattern = std::make_unique<Pattern>(PatternKind::Identifier);
    pattern->name = name;
    return pattern;
}

std::unique_ptr<Pattern> make_wildcard_pattern() {
    return std::make_unique<Pattern>(PatternKind::Wildcard);
}

std::unique_ptr<Pattern> make_or_pattern(std::vector<std::unique_ptr<Pattern>> patterns) {
    auto pattern = std::make_unique<Pattern>(PatternKind::Or);
    pattern->patterns = std::move(patterns);
    return pattern;
}

std::unique_ptr<Pattern> make_struct_pattern(const std::string& struct_name,
                                             std::vector<std::pair<std::string, std::unique_ptr<Pattern>>> fields) {
    auto pattern = std::make_unique<Pattern>(PatternKind::Struct);
    pattern->struct_name = struct_name;
    pattern->struct_fields = std::move(fields);
    return pattern;
}

std::unique_ptr<Pattern> make_enum_pattern(const std::string& enum_name,
                                           const std::string& variant_name,
                                           std::unique_ptr<Pattern> pattern) {
    auto p = std::make_unique<Pattern>(PatternKind::Enum);
    p->enum_name = enum_name;
    p->variant_name = variant_name;
    p->enum_pattern = std::move(pattern);
    return p;
}

// Match分支创建辅助函数
MatchArm make_match_arm(std::unique_ptr<Pattern> pattern,
                        std::unique_ptr<Expr> guard,
                        std::unique_ptr<Stmt> body) {
    MatchArm arm;
    arm.pattern = std::move(pattern);
    arm.guard = std::move(guard);
    arm.body = std::move(body);
    return arm;
}

} // namespace frontend
} // namespace az