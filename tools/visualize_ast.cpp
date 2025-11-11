// AZ AST可视化工具
// 独立工具，用于生成AST的Graphviz表示

#include "AZ/Frontend/Lexer.h"
#include "AZ/Frontend/Parser.h"
#include "AZ/Frontend/ASTVisualizer.h"

#include <iostream>
#include <fstream>
#include <filesystem>

using namespace az::frontend;

void printUsage(const char* programName) {
    std::cout << "用法: " << programName << " [选项] <源文件>" << std::endl;
    std::cout << "选项:" << std::endl;
    std::cout << "  -o <文件>          输出文件 (默认: ast.dot)" << std::endl;
    std::cout << "  --cfg              生成控制流图" << std::endl;
    std::cout << "  --call-graph       生成函数调用图" << std::endl;
    std::cout << "  -h, --help         显示此帮助" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }
    
    // 解析命令行参数
    std::string sourceFile;
    std::string outputFile = "ast.dot";
    bool generateCFG = false;
    bool generateCallGraph = false;
    
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
        } else if (arg == "--cfg") {
            generateCFG = true;
        } else if (arg == "--call-graph") {
            generateCallGraph = true;
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
    Lexer lexer(sourceCode, sourceFile);
    auto tokensResult = lexer.tokenize();
    if (tokensResult.isErr()) {
        std::cerr << "词法分析错误: " 
                  << tokensResult.error().message 
                  << " (文件: " << tokensResult.error().filename
                  << ", 行: " << tokensResult.error().line
                  << ", 列: " << tokensResult.error().column << ")" << std::endl;
        return 1;
    }
    
    std::cout << "✅ 词法分析完成，生成 " << tokensResult.value().size() << " 个词法单元" << std::endl;
    
    // 语法分析
    Parser parser(std::move(tokensResult.value()), sourceFile);
    auto programResult = parser.parse();
    if (programResult.isErr()) {
        std::cerr << "语法分析错误: " 
                  << programResult.error().message 
                  << " (文件: " << programResult.error().filename
                  << ", 行: " << programResult.error().line
                  << ", 列: " << programResult.error().column << ")" << std::endl;
        return 1;
    }
    
    std::cout << "✅ 语法分析完成" << std::endl;
    
    // 可视化AST
    ASTVisualizer visualizer;
    if (visualizer.visualizeAST(programResult.value().get(), outputFile)) {
        std::cout << "✅ AST已可视化到文件: " << outputFile << std::endl;
    } else {
        std::cerr << "❌ AST可视化失败" << std::endl;
        return 1;
    }
    
    // 如果需要生成控制流图
    if (generateCFG) {
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
    
    // 如果需要生成函数调用图
    if (generateCallGraph) {
        std::string callGraphOutput = "callgraph.dot";
        if (visualizer.visualizeCallGraph(programResult.value().get(), callGraphOutput)) {
            std::cout << "✅ 函数调用图已可视化到文件: " << callGraphOutput << std::endl;
        } else {
            std::cerr << "❌ 函数调用图可视化失败" << std::endl;
            return 1;
        }
    }
    
    std::cout << "\n要查看生成的图形，请使用Graphviz工具:" << std::endl;
    std::cout << "  dot -Tpng " << outputFile << " -o ast.png" << std::endl;
    std::cout << "  start ast.png" << std::endl;
    
    return 0;
}