// AZ编译器 - 链接器集成头文件

#ifndef AZ_BACKEND_LINKER_H
#define AZ_BACKEND_LINKER_H

#include "AZ/Support/Result.h"
#include "AZ/Frontend/Error.h"

#include <string>
#include <vector>

namespace az {
namespace backend {

class Linker {
public:
    // 链接器类型枚举
    enum class LinkerType {
        LLD,   // LLVM linker
        MOLD,  // mold linker
        GCC,   // GNU linker
        CLANG, // Clang linker
        MSVC   // MSVC linker
    };
    
    struct LinkOptions {
        std::vector<std::string> objectFiles;
        std::vector<std::string> libraryPaths;
        std::vector<std::string> libraries;
        std::string outputPath;
        bool staticLink = false;
        bool lto = false;
        bool pie = false;
        bool strip = false;
        bool shared = false;
        std::string targetTriple;
        LinkerType linkerType = LinkerType::LLD; // 默认使用LLD
    };

    Linker();
    
    /// 链接目标文件
    Result<void> link(const LinkOptions& options);
    
    /// 创建静态库
    Result<void> createStaticLibrary(
        const std::vector<std::string>& objectFiles,
        const std::string& outputPath);
    
    /// 创建共享库
    Result<void> createSharedLibrary(
        const std::vector<std::string>& objectFiles,
        const std::string& outputPath,
        const std::vector<std::string>& libraryPaths = {},
        const std::vector<std::string>& libraries = {});
    
    /// 查找系统库
    Result<std::string> findSystemLibrary(
        const std::string& libName,
        const std::vector<std::string>& searchPaths);
    
    /// 检测可用的链接器
    static LinkerType detectLinker();

private:
    std::vector<std::string> buildLldArgs(const LinkOptions& options);
    std::vector<std::string> buildMoldArgs(const LinkOptions& options);
    Result<void> invokeLld(const std::vector<std::string>& args);
    Result<void> invokeMold(const std::vector<std::string>& args);
    CompileError makeError(const std::string& message);
};

} // namespace backend
} // namespace az

#endif // AZ_BACKEND_LINKER_H