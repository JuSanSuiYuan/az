// AZ编译器主程序

#include "AZ/Backend/LLVMBackend.h"
#include "AZ/Backend/TargetManager.h"
#include "AZ/Frontend/Lexer.h"
#include "AZ/Frontend/Parser.h"
#include "AZ/Frontend/Sema.h"
#include "AZ/Frontend/ASTVisualizer.h"
#include "AZ/MLIR/MLIRGen.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"

#include <iostream>
#include <fstream>
#include <filesystem>

using namespace az;

class AZCompiler {
public:
    AZCompiler() : context_() {
        // 注册方言
        context_.loadDialect<mlir::func::FuncDialect>();
        context_.loadDialect<mlir::arith::ArithDialect>();
        context_.loadDialect<mlir::scf::SCFDialect>();
        context_.loadDialect<mlir::cf::ControlFlowDialect>();
        context_.loadDialect<mlir::memref::MemRefDialect>();
        context_.loadDialect<mlir::LLVM::LLVMDialect>();
    }
    
    int compile(const std::string& sourceFile, 
                const std::string& outputFile,
                const backend::LLVMBackend::Options& options) {
        
        std::cout << "编译信息: 开始编译 " << sourceFile << std::endl;
        
        // 1. 读取源文件
        std::ifstream file(sourceFile);
        if (!file.is_open()) {
            std::cerr << "错误: 无法打开源文件: " << sourceFile << std::endl;
            return 1;
        }
        
        std::string sourceCode((std::istreambuf_iterator<char>(file)),
                               std::istreambuf_iterator<char>());
        file.close();
        
        std::cout << "编译信息: 源文件读取完成，大小 " << sourceCode.length() << " 字节" << std::endl;
        
        // 2. 词法分析
        std::cout << "编译信息: 开始词法分析" << std::endl;
        frontend::Lexer lexer(sourceCode, sourceFile);
        auto tokensResult = lexer.tokenize();
        if (tokensResult.isErr()) {
            std::cerr << "词法分析错误: " 
                      << tokensResult.error().message 
                      << " (文件: " << tokensResult.error().filename
                      << ", 行: " << tokensResult.error().line
                      << ", 列: " << tokensResult.error().column << ")" << std::endl;
            return 1;
        }
        
        std::cout << "编译信息: 词法分析完成，生成 " << tokensResult.value().size() << " 个词法单元" << std::endl;
        
        // 3. 语法分析
        std::cout << "编译信息: 开始语法分析" << std::endl;
        frontend::Parser parser(std::move(tokensResult.value()), sourceFile);
        auto programResult = parser.parse();
        if (programResult.isErr()) {
            std::cerr << "语法分析错误: " 
                      << programResult.error().message 
                      << " (文件: " << programResult.error().filename
                      << ", 行: " << programResult.error().line
                      << ", 列: " << programResult.error().column << ")" << std::endl;
            return 1;
        }
        
        std::cout << "编译信息: 语法分析完成" << std::endl;
        
        // 4. 语义分析
        std::cout << "编译信息: 开始语义分析" << std::endl;
        frontend::SemanticAnalyzer sema;
        auto semaResult = sema.analyze(programResult.value().get());
        if (semaResult.isErr()) {
            std::cerr << "语义分析错误: " 
                      << semaResult.error().message 
                      << " (文件: " << semaResult.error().filename
                      << ", 行: " << semaResult.error().line
                      << ", 列: " << semaResult.error().column << ")" << std::endl;
            return 1;
        }
        
        std::cout << "编译信息: 语义分析完成" << std::endl;
        
        // 5. MLIR生成
        std::cout << "编译信息: 开始MLIR生成" << std::endl;
        mlir::MLIRGen mlirGen(context_);
        auto module = mlirGen.generate(programResult.value().get());
        if (!module) {
            std::cerr << "MLIR生成失败" << std::endl;
            return 1;
        }
        
        std::cout << "编译信息: MLIR生成完成" << std::endl;
        
        // 6. LLVM后端处理
        std::cout << "编译信息: 开始LLVM后端处理" << std::endl;
        backend::LLVMBackend llvmBackend(context_);
        llvmBackend.setOptions(options);
        llvmBackend.setSourceFilename(sourceFile);
        
        auto compileResult = llvmBackend.compile(module, outputFile);
        if (compileResult.isErr()) {
            std::cerr << "编译错误: " 
                      << compileResult.error().message 
                      << " (文件: " << compileResult.error().filename << ")" << std::endl;
            return 1;
        }
        
        std::cout << "编译信息: 编译成功完成，输出文件: " << compileResult.value() << std::endl;
        return 0;
    }
    
    int jitRun(const std::string& sourceFile,
               const std::vector<std::string>& args) {
        
        std::cout << "运行信息: 开始JIT运行 " << sourceFile << std::endl;
        
        // 1. 读取源文件
        std::ifstream file(sourceFile);
        if (!file.is_open()) {
            std::cerr << "错误: 无法打开源文件: " << sourceFile << std::endl;
            return 1;
        }
        
        std::string sourceCode((std::istreambuf_iterator<char>(file)),
                               std::istreambuf_iterator<char>());
        file.close();
        
        // 2. 词法分析
        frontend::Lexer lexer(sourceCode, sourceFile);
        auto tokensResult = lexer.tokenize();
        if (tokensResult.isErr()) {
            std::cerr << "词法分析错误: " 
                      << tokensResult.error().message 
                      << " (文件: " << tokensResult.error().filename
                      << ", 行: " << tokensResult.error().line
                      << ", 列: " << tokensResult.error().column << ")" << std::endl;
            return 1;
        }
        
        // 3. 语法分析
        frontend::Parser parser(std::move(tokensResult.value()), sourceFile);
        auto programResult = parser.parse();
        if (programResult.isErr()) {
            std::cerr << "语法分析错误: " 
                      << programResult.error().message 
                      << " (文件: " << programResult.error().filename
                      << ", 行: " << programResult.error().line
                      << ", 列: " << programResult.error().column << ")" << std::endl;
            return 1;
        }
        
        // 4. 语义分析
        frontend::SemanticAnalyzer sema;
        auto semaResult = sema.analyze(programResult.value().get());
        if (semaResult.isErr()) {
            std::cerr << "语义分析错误: " 
                      << semaResult.error().message 
                      << " (文件: " << semaResult.error().filename
                      << ", 行: " << semaResult.error().line
                      << ", 列: " << semaResult.error().column << ")" << std::endl;
            return 1;
        }
        
        // 5. MLIR生成
        mlir::MLIRGen mlirGen(context_);
        auto module = mlirGen.generate(programResult.value().get());
        if (!module) {
            std::cerr << "MLIR生成失败" << std::endl;
            return 1;
        }
        
        // 6. JIT运行
        backend::LLVMBackend llvmBackend(context_);
        auto runResult = llvmBackend.jitCompileAndRun(module, args);
        if (runResult.isErr()) {
            std::cerr << "JIT执行错误: " 
                      << runResult.error().message 
                      << " (文件: " << runResult.error().filename << ")" << std::endl;
            return 1;
        }
        
        std::cout << "运行信息: JIT执行完成，退出码: " << runResult.value() << std::endl;
        return runResult.value();
    }
    
    void printTargets() {
        backend::TargetManager targetManager;
        auto targetsResult = targetManager.getAvailableTargets();
        if (targetsResult.isOk()) {
            std::cout << "可用目标:" << std::endl;
            for (const auto& target : targetsResult.value()) {
                std::cout << "  " << target << std::endl;
            }
        } else {
            std::cerr << "获取目标列表失败: " << targetsResult.error().message << std::endl;
        }
    }
    
    void printNativeTarget() {
        backend::TargetManager targetManager;
        auto tripleResult = targetManager.getNativeTargetTriple();
        if (tripleResult.isOk()) {
            std::cout << "本机目标三元组: " << tripleResult.value() << std::endl;
        } else {
            std::cerr << "获取本机目标三元组失败: " << tripleResult.error().message << std::endl;
        }
        
        auto cpuResult = targetManager.getNativeCPU();
        if (cpuResult.isOk()) {
            std::cout << "本机CPU: " << cpuResult.value() << std::endl;
        } else {
            std::cerr << "获取本机CPU失败: " << cpuResult.error().message << std::endl;
        }
        
        auto featuresResult = targetManager.getNativeFeatures();
        if (featuresResult.isOk()) {
            std::cout << "本机特性: " << featuresResult.value() << std::endl;
        } else {
            std::cerr << "获取本机特性失败: " << featuresResult.error().message << std::endl;
        }
    }

private:
    mlir::MLIRContext context_;
};

void printUsage(const char* programName) {
    std::cout << "用法: " << programName << " [选项] <源文件>" << std::endl;
    std::cout << "选项:" << std::endl;
    std::cout << "  -o <文件>          输出文件" << std::endl;
    std::cout << "  -c                 仅编译" << std::endl;
    std::cout << "  -g                 生成调试信息" << std::endl;
    std::cout << "  -O0,-O1,-O2,-O3    优化级别" << std::endl;
    std::cout << "  -Os,-Oz            大小优化" << std::endl;
    std::cout << "  --jit              使用JIT运行" << std::endl;
    std::cout << "  --aot              使用AOT编译" << std::endl;
    std::cout << "  --emit-llvm        输出LLVM IR" << std::endl;
    std::cout << "  --emit-asm         输出汇编代码" << std::endl;
    std::cout << "  --emit-bc          输出bitcode" << std::endl;
    std::cout << "  --visualize-ast    可视化AST为Graphviz格式" << std::endl;
    std::cout << "  --visualize-cfg    可视化控制流图" << std::endl;
    std::cout << "  --visualize-call   可视化函数调用图" << std::endl;
    std::cout << "  --target <三元组>  目标三元组" << std::endl;
    std::cout << "  --static           静态链接" << std::endl;
    std::cout << "  --lto              启用LTO" << std::endl;
    std::cout << "  --pie              生成PIE" << std::endl;
    std::cout << "  --strip            去除符号表" << std::endl;
    std::cout << "  --linker <类型>    指定链接器类型 (lld, mold, gcc, clang, msvc)" << std::endl;
    std::cout << "  --print-targets    显示可用目标" << std::endl;
    std::cout << "  --print-native     显示本机目标信息" << std::endl;
    std::cout << "  -h, --help         显示此帮助" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }
    
    // 解析命令行参数
    std::string sourceFile;
    std::string outputFile = "a.out";
    std::string visualizeOutput = "ast.dot";
    bool jitMode = false;
    bool visualizeAST = false;
    bool visualizeCFG = false;
    bool visualizeCallGraph = false;
    backend::LLVMBackend::Options options;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "-o") {
            if (i + 1 < argc) {
                outputFile = argv[++i];
            } else {
                std::cerr << "错误: -o 需要一个参数" << std::endl;
                return 1;
            }
        } else if (arg == "-c") {
            options.outputType = backend::LLVMBackend::OutputType::Object;
        } else if (arg == "-g") {
            options.debugInfo = true;
        } else if (arg == "-O0") {
            options.optLevel = backend::LLVMBackend::OptLevel::O0;
        } else if (arg == "-O1") {
            options.optLevel = backend::LLVMBackend::OptLevel::O1;
        } else if (arg == "-O2") {
            options.optLevel = backend::LLVMBackend::OptLevel::O2;
        } else if (arg == "-O3") {
            options.optLevel = backend::LLVMBackend::OptLevel::O3;
        } else if (arg == "-Os") {
            options.optLevel = backend::LLVMBackend::OptLevel::Os;
        } else if (arg == "-Oz") {
            options.optLevel = backend::LLVMBackend::OptLevel::Oz;
        } else if (arg == "--jit") {
            jitMode = true;
        } else if (arg == "--aot") {
            options.enableAOT = true;
            options.outputType = backend::LLVMBackend::OutputType::AOTExecutable;
        } else if (arg == "--emit-llvm") {
            options.outputType = backend::LLVMBackend::OutputType::LLVMIR;
        } else if (arg == "--emit-asm") {
            options.outputType = backend::LLVMBackend::OutputType::Assembly;
        } else if (arg == "--emit-bc") {
            options.outputType = backend::LLVMBackend::OutputType::Bitcode;
        } else if (arg == "--visualize-ast") {
            visualizeAST = true;
        } else if (arg == "--visualize-cfg") {
            visualizeCFG = true;
        } else if (arg == "--visualize-call") {
            visualizeCallGraph = true;
        } else if (arg == "--target") {
            if (i + 1 < argc) {
                options.targetTriple = argv[++i];
            } else {
                std::cerr << "错误: --target 需要一个参数" << std::endl;
                return 1;
            }
        } else if (arg == "--static") {
            options.staticLink = true;
        } else if (arg == "--lto") {
            options.lto = true;
        } else if (arg == "--pie") {
            options.pie = true;
        } else if (arg == "--strip") {
            options.strip = true;
        } else if (arg == "--linker") {
            if (i + 1 < argc) {
                std::string linkerType = argv[++i];
                if (linkerType == "lld") {
                    options.linkerType = backend::LLVMBackend::LinkerType::LLD;
                } else if (linkerType == "mold") {
                    options.linkerType = backend::LLVMBackend::LinkerType::MOLD;
                } else if (linkerType == "gcc") {
                    options.linkerType = backend::LLVMBackend::LinkerType::GCC;
                } else if (linkerType == "clang") {
                    options.linkerType = backend::LLVMBackend::LinkerType::CLANG;
                } else if (linkerType == "msvc") {
                    options.linkerType = backend::LLVMBackend::LinkerType::MSVC;
                } else {
                    std::cerr << "错误: 未知的链接器类型: " << linkerType << std::endl;
                    return 1;
                }
            } else {
                std::cerr << "错误: --linker 需要一个参数" << std::endl;
                return 1;
            }
        } else if (arg == "--print-targets") {
            AZCompiler compiler;
            compiler.printTargets();
            return 0;
        } else if (arg == "--print-native") {
            AZCompiler compiler;
            compiler.printNativeTarget();
            return 0;
        } else if (arg.substr(0, 1) == "-") {
            std::cerr << "错误: 未知选项: " << arg << std::endl;
            return 1;
        } else {
            sourceFile = arg;
        }
    }
    
    if (sourceFile.empty()) {
        std::cerr << "错误: 未指定源文件" << std::endl;
        return 1;
    }
    
    if (!std::filesystem::exists(sourceFile)) {
        std::cerr << "错误: 源文件不存在: " << sourceFile << std::endl;
        return 1;
    }
    
    // 如果只是可视化，不需要完整的编译流程
    if (visualizeAST || visualizeCFG || visualizeCallGraph) {
        // 读取源文件
        std::ifstream file(sourceFile);
        if (!file.is_open()) {
            std::cerr << "错误: 无法打开源文件: " << sourceFile << std::endl;
            return 1;
        }
        
        std::string sourceCode((std::istreambuf_iterator<char>(file)),
                               std::istreambuf_iterator<char>());
        file.close();
        
        // 词法分析
        frontend::Lexer lexer(sourceCode, sourceFile);
        auto tokensResult = lexer.tokenize();
        if (tokensResult.isErr()) {
            std::cerr << "词法分析错误: " 
                      << tokensResult.error().message 
                      << " (文件: " << tokensResult.error().filename
                      << ", 行: " << tokensResult.error().line
                      << ", 列: " << tokensResult.error().column << ")" << std::endl;
            return 1;
        }
        
        // 语法分析
        frontend::Parser parser(std::move(tokensResult.value()), sourceFile);
        auto programResult = parser.parse();
        if (programResult.isErr()) {
            std::cerr << "语法分析错误: " 
                      << programResult.error().message 
                      << " (文件: " << programResult.error().filename
                      << ", 行: " << programResult.error().line
                      << ", 列: " << programResult.error().column << ")" << std::endl;
            return 1;
        }
        
        // 可视化AST
        if (visualizeAST) {
            frontend::ASTVisualizer visualizer;
            if (visualizer.visualizeAST(programResult.value().get(), visualizeOutput)) {
                std::cout << "✅ AST已可视化到文件: " << visualizeOutput << std::endl;
            } else {
                std::cerr << "❌ AST可视化失败" << std::endl;
                return 1;
            }
        }
        
        // 可视化函数调用图
        if (visualizeCallGraph) {
            frontend::ASTVisualizer visualizer;
            std::string callGraphOutput = "callgraph.dot";
            if (visualizer.visualizeCallGraph(programResult.value().get(), callGraphOutput)) {
                std::cout << "✅ 函数调用图已可视化到文件: " << callGraphOutput << std::endl;
            } else {
                std::cerr << "❌ 函数调用图可视化失败" << std::endl;
                return 1;
            }
        }
        
        // 可视化控制流图
        if (visualizeCFG) {
            frontend::ASTVisualizer visualizer;
            std::string cfgOutput = "cfg.dot";
            // 为简化起见，我们只可视化第一个函数的控制流图
            if (!programResult.value()->statements.empty()) {
                if (visualizer.visualizeControlFlow(programResult.value()->statements[0].get(), cfgOutput)) {
                    std::cout << "✅ 控制流图已可视化到文件: " << cfgOutput << std::endl;
                } else {
                    std::cerr << "❌ 控制流图可视化失败" << std::endl;
                    return 1;
                }
            }
        }
        
        return 0;
    }
    
    // 创建编译器并执行
    AZCompiler compiler;
    
    if (jitMode) {
        // 收集JIT参数
        std::vector<std::string> jitArgs;
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--") {
                for (int j = i + 1; j < argc; ++j) {
                    jitArgs.push_back(argv[j]);
                }
                break;
            }
        }
        
        return compiler.jitRun(sourceFile, jitArgs);
    } else {
        return compiler.compile(sourceFile, outputFile, options);
    }
}