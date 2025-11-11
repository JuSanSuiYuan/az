// AZ编译器 - 链接器集成实现

#include "AZ/Backend/Linker.h"

#include <filesystem>
#include <cstdlib>

// lld头文件
#include "lld/Common/Driver.h"
#include "lld/Common/ErrorHandler.h"
#include "llvm/Support/CommandLine.h"

namespace az {
namespace backend {

Linker::Linker() {
}

Result<void> Linker::link(const LinkOptions& options) {
    // 验证输入文件存在
    for (const auto& objFile : options.objectFiles) {
        if (!std::filesystem::exists(objFile)) {
            return Result<void>::Err(
                makeError("目标文件不存在: " + objFile));
        }
    }

    // 根据链接器类型调用相应的链接函数
    switch (options.linkerType) {
    case LinkerType::LLD:
        {
            // 构建lld参数
            auto args = buildLldArgs(options);
            // 调用lld
            return invokeLld(args);
        }
    case LinkerType::MOLD:
        {
            // 构建mold参数
            auto args = buildMoldArgs(options);
            // 调用mold
            return invokeMold(args);
        }
    default:
        {
            // 对于其他链接器类型，使用命令行调用
            // 构建lld参数作为默认选项
            auto args = buildLldArgs(options);
            // 调用lld
            return invokeLld(args);
        }
    }
}

std::vector<std::string> Linker::buildLldArgs(const LinkOptions& options) {
    std::vector<std::string> args;

    // 添加程序名
    args.push_back("lld");

    // 输出文件
    args.push_back("-o");
    args.push_back(options.outputPath);

    // 目标文件
    for (const auto& objFile : options.objectFiles) {
        args.push_back(objFile);
    }

    // 库搜索路径
    for (const auto& libPath : options.libraryPaths) {
        args.push_back("-L" + libPath);
    }

    // 链接库
    for (const auto& lib : options.libraries) {
        args.push_back("-l" + lib);
    }

    // 静态链接
    if (options.staticLink) {
        args.push_back("-static");
    }

    // LTO
    if (options.lto) {
        args.push_back("-flto");
    }

    // PIE
    if (options.pie) {
        args.push_back("-pie");
    }

    // strip symbols
    if (options.strip) {
        args.push_back("-strip-all");
    }

    // 共享库
    if (options.shared) {
        args.push_back("-shared");
    }

    return args;
}

std::vector<std::string> Linker::buildMoldArgs(const LinkOptions& options) {
    std::vector<std::string> args;

    // 添加程序名
    args.push_back("mold");

    // 输出文件
    args.push_back("-o");
    args.push_back(options.outputPath);

    // 目标文件
    for (const auto& objFile : options.objectFiles) {
        args.push_back(objFile);
    }

    // 库搜索路径
    for (const auto& libPath : options.libraryPaths) {
        args.push_back("-L" + libPath);
    }

    // 链接库
    for (const auto& lib : options.libraries) {
        args.push_back("-l" + lib);
    }

    // 静态链接
    if (options.staticLink) {
        args.push_back("-static");
    }

    // LTO
    if (options.lto) {
        args.push_back("-flto");
    }

    // PIE
    if (options.pie) {
        args.push_back("-pie");
    }

    // strip symbols
    if (options.strip) {
        args.push_back("--strip-all");
    }

    // 共享库
    if (options.shared) {
        args.push_back("-shared");
    }

    return args;
}

Result<void> Linker::invokeLld(const std::vector<std::string>& args) {
    // 将std::string转换为const char*
    std::vector<const char*> cArgs;
    for (const auto& arg : args) {
        cArgs.push_back(arg.c_str());
    }
    
    // 调用lld链接器
    bool success = false;
    
    // 根据平台选择合适的链接器
#ifdef _WIN32
    // Windows使用COFF链接器
    success = lld::coff::link(cArgs, llvm::outs(), llvm::errs(), false, false);
#elif __APPLE__
    // macOS使用MachO链接器
    success = lld::macho::link(cArgs, llvm::outs(), llvm::errs(), false, false);
#else
    // Linux使用ELF链接器
    success = lld::elf::link(cArgs, llvm::outs(), llvm::errs(), false, false);
#endif
    
    if (!success) {
        return Result<void>::Err(
            makeError("LLD链接失败"));
    }
    
    return Result<void>::Ok();
}

Result<void> Linker::invokeMold(const std::vector<std::string>& args) {
    // 将std::string转换为const char*
    std::vector<const char*> cArgs;
    for (const auto& arg : args) {
        cArgs.push_back(arg.c_str());
    }
    
    // 构建命令行
    std::string command = "mold";
    for (size_t i = 1; i < cArgs.size(); ++i) {  // 跳过第一个参数（程序名）
        command += " ";
        // 如果参数包含空格，需要加引号
        std::string argStr(cArgs[i]);
        if (argStr.find(' ') != std::string::npos) {
            command += "\"" + argStr + "\"";
        } else {
            command += argStr;
        }
    }
    
    // 执行命令
    int result = std::system(command.c_str());
    
    if (result != 0) {
        return Result<void>::Err(
            makeError("mold链接失败，退出码: " + std::to_string(result)));
    }
    
    return Result<void>::Ok();
}

Result<std::string> Linker::findSystemLibrary(
    const std::string& libName,
    const std::vector<std::string>& searchPaths) {
    
    // 尝试不同的库文件扩展名
    std::vector<std::string> extensions = {".a", ".so", ".dll", ".dylib"};

    for (const auto& path : searchPaths) {
        for (const auto& ext : extensions) {
            std::string libPath = path + "/lib" + libName + ext;
            if (std::filesystem::exists(libPath)) {
                return Result<std::string>::Ok(libPath);
            }
        }
    }

    return Result<std::string>::Err(
        makeError("找不到库: " + libName));
}

Result<void> Linker::createStaticLibrary(
    const std::vector<std::string>& objectFiles,
    const std::string& outputPath) {
    
    // 验证输入文件存在
    for (const auto& objFile : objectFiles) {
        if (!std::filesystem::exists(objFile)) {
            return Result<void>::Err(
                makeError("目标文件不存在: " + objFile));
        }
    }

    // 构建ar命令
    std::string command = "ar rcs " + outputPath;
    for (const auto& objFile : objectFiles) {
        command += " " + objFile;
    }
    
    // 执行命令
    int result = std::system(command.c_str());
    
    if (result != 0) {
        return Result<void>::Err(
            makeError("创建静态库失败，退出码: " + std::to_string(result)));
    }
    
    return Result<void>::Ok();
}

Result<void> Linker::createSharedLibrary(
    const std::vector<std::string>& objectFiles,
    const std::string& outputPath,
    const std::vector<std::string>& libraryPaths,
    const std::vector<std::string>& libraries) {
    
    // 验证输入文件存在
    for (const auto& objFile : objectFiles) {
        if (!std::filesystem::exists(objFile)) {
            return Result<void>::Err(
                makeError("目标文件不存在: " + objFile));
        }
    }

    // 构建链接命令
    std::string command;
    
#ifdef _WIN32
    command = "lld-link";  // Windows使用lld-link
#elif __APPLE__
    command = "ld64.lld";  // macOS使用ld64.lld
#else
    command = "ld.lld";    // Linux使用ld.lld
#endif
    
    command += " -shared -o " + outputPath;
    
    // 添加目标文件
    for (const auto& objFile : objectFiles) {
        command += " " + objFile;
    }
    
    // 添加库搜索路径
    for (const auto& libPath : libraryPaths) {
        command += " -L" + libPath;
    }
    
    // 添加链接库
    for (const auto& lib : libraries) {
        command += " -l" + lib;
    }
    
    // 执行命令
    int result = std::system(command.c_str());
    
    if (result != 0) {
        return Result<void>::Err(
            makeError("创建共享库失败，退出码: " + std::to_string(result)));
    }
    
    return Result<void>::Ok();
}

Linker::LinkerType Linker::detectLinker() {
    // 检查mold是否可用
    int result = std::system("mold --version > /dev/null 2>&1");
    if (result == 0) {
        return LinkerType::MOLD;
    }
    
    // 检查lld是否可用
    result = std::system("lld --version > /dev/null 2>&1");
    if (result == 0) {
        return LinkerType::LLD;
    }
    
    // 检查clang是否可用
    result = std::system("clang --version > /dev/null 2>&1");
    if (result == 0) {
        return LinkerType::CLANG;
    }
    
    // 检查gcc是否可用
    result = std::system("gcc --version > /dev/null 2>&1");
    if (result == 0) {
        return LinkerType::GCC;
    }
    
#ifdef _WIN32
    // 检查MSVC是否可用
    result = std::system("cl /? > /dev/null 2>&1");
    if (result == 0) {
        return LinkerType::MSVC;
    }
#endif
    
    // 默认返回LLD
    return LinkerType::LLD;
}

CompileError Linker::makeError(const std::string& message) {
    return CompileError{
        ErrorKind::UnknownError,
        message,
        0, 0, ""
    };
}

} // namespace backend
} // namespace az