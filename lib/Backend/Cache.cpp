// AZ编译器 - 编译缓存实现

#include "AZ/Backend/Cache.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <chrono>

// 用于计算哈希
#include <iomanip>
#include <openssl/sha.h>

namespace az {
namespace backend {

CompilationCache::CompilationCache(const std::string& cacheDir)
    : cacheDir_(cacheDir) {
    
    // 创建缓存目录
    std::filesystem::create_directories(cacheDir_);
}

Result<bool> CompilationCache::hasCache(const std::string& sourceFile) {
    // 计算源文件哈希
    auto hashResult = computeHash(sourceFile);
    if (hashResult.isErr()) {
        return Result<bool>::Err(hashResult.error());
    }
    
    auto hash = hashResult.value();
    
    // 检查缓存文件是否存在
    auto cachePath = getCachePath(hash);
    bool exists = std::filesystem::exists(cachePath);
    
    // 如果缓存存在，验证其有效性
    if (exists) {
        auto validResult = isCacheValid(sourceFile, cachePath);
        if (validResult.isErr() || !validResult.value()) {
            // 缓存无效，删除它
            std::filesystem::remove(cachePath);
            return Result<bool>::Ok(false);
        }
    }
    
    return Result<bool>::Ok(exists);
}

Result<std::string> CompilationCache::getCachedObjectFile(
    const std::string& sourceFile) {
    
    // 计算源文件哈希
    auto hashResult = computeHash(sourceFile);
    if (hashResult.isErr()) {
        return Result<std::string>::Err(hashResult.error());
    }
    
    auto hash = hashResult.value();
    
    // 获取缓存路径
    auto cachePath = getCachePath(hash);
    
    // 检查缓存是否存在
    if (!std::filesystem::exists(cachePath)) {
        return Result<std::string>::Err(
            makeError("缓存不存在"));
    }
    
    // 验证缓存有效性
    auto validResult = isCacheValid(sourceFile, cachePath);
    if (validResult.isErr()) {
        return Result<std::string>::Err(validResult.error());
    }
    
    if (!validResult.value()) {
        // 缓存无效，删除它
        std::filesystem::remove(cachePath);
        return Result<std::string>::Err(
            makeError("缓存已过期"));
    }
    
    return Result<std::string>::Ok(cachePath);
}

Result<void> CompilationCache::saveToCache(
    const std::string& sourceFile,
    const std::string& objectFile) {
    
    // 计算源文件哈希
    auto hashResult = computeHash(sourceFile);
    if (hashResult.isErr()) {
        return Result<void>::Err(hashResult.error());
    }
    
    auto hash = hashResult.value();
    
    // 获取缓存路径
    auto cachePath = getCachePath(hash);
    
    // 复制目标文件到缓存
    try {
        std::filesystem::copy_file(
            objectFile,
            cachePath,
            std::filesystem::copy_options::overwrite_existing
        );
        
        // 创建元数据文件
        auto metadataPath = cachePath + ".meta";
        std::ofstream metadataFile(metadataPath);
        if (metadataFile.is_open()) {
            // 写入源文件路径和时间戳
            auto ftime = std::filesystem::last_write_time(sourceFile);
            auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                ftime - std::filesystem::file_time_type::clock::now() + 
                std::chrono::system_clock::now()
            );
            auto time = std::chrono::system_clock::to_time_t(sctp);
            
            metadataFile << sourceFile << std::endl;
            metadataFile << time << std::endl;
            metadataFile.close();
        }
    } catch (const std::filesystem::filesystem_error& e) {
        return Result<void>::Err(
            makeError("保存缓存失败: " + std::string(e.what())));
    }
    
    return Result<void>::Ok();
}

void CompilationCache::clearCache() {
    // 删除缓存目录中的所有文件
    try {
        std::filesystem::remove_all(cacheDir_);
        std::filesystem::create_directories(cacheDir_);
    } catch (const std::filesystem::filesystem_error&) {
        // 忽略错误
    }
}

void CompilationCache::cleanupCache(size_t maxSizeMB) {
    // 获取缓存目录中的所有文件
    std::vector<std::pair<std::filesystem::path, std::filesystem::file_time_type>> files;
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(cacheDir_)) {
            if (entry.is_regular_file() && entry.path().extension() == ".o") {
                files.emplace_back(entry.path(), entry.last_write_time());
            }
        }
    } catch (const std::filesystem::filesystem_error&) {
        // 忽略错误
        return;
    }
    
    // 按修改时间排序（最旧的在前）
    std::sort(files.begin(), files.end(), 
              [](const auto& a, const auto& b) {
                  return a.second < b.second;
              });
    
    // 计算总大小并删除最旧的文件直到满足大小限制
    uintmax_t totalSize = 0;
    for (const auto& file : files) {
        try {
            totalSize += std::filesystem::file_size(file.first);
        } catch (const std::filesystem::filesystem_error&) {
            // 忽略错误
        }
    }
    
    uintmax_t maxSizeBytes = maxSizeMB * 1024 * 1024;
    for (const auto& file : files) {
        if (totalSize <= maxSizeBytes) {
            break;
        }
        
        try {
            uintmax_t fileSize = std::filesystem::file_size(file.first);
            std::filesystem::remove(file.first);
            std::filesystem::remove(file.first.string() + ".meta");
            totalSize -= fileSize;
        } catch (const std::filesystem::filesystem_error&) {
            // 忽略错误
        }
    }
}

Result<std::string> CompilationCache::computeHash(const std::string& filePath) {
    // 使用SHA256计算文件哈希
    std::ifstream file(filePath, std::ios::binary);
    if (!file) {
        return Result<std::string>::Err(
            makeError("无法打开文件: " + filePath));
    }

    // 读取文件内容
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();
    file.close();

    // 计算SHA256哈希
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, content.c_str(), content.size());
    SHA256_Final(hash, &sha256);

    // 转换为十六进制字符串
    std::stringstream ss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
    }
    
    return Result<std::string>::Ok(ss.str());
}

Result<bool> CompilationCache::isCacheValid(
    const std::string& sourceFile, 
    const std::string& cacheFile) {
    
    // 检查元数据文件
    auto metadataPath = cacheFile + ".meta";
    if (!std::filesystem::exists(metadataPath)) {
        return Result<bool>::Ok(false);
    }
    
    // 读取元数据
    std::ifstream metadataFile(metadataPath);
    if (!metadataFile.is_open()) {
        return Result<bool>::Ok(false);
    }
    
    std::string storedSourceFile;
    std::time_t storedTimestamp;
    
    std::getline(metadataFile, storedSourceFile);
    metadataFile >> storedTimestamp;
    metadataFile.close();
    
    // 检查源文件路径是否匹配
    if (storedSourceFile != sourceFile) {
        return Result<bool>::Ok(false);
    }
    
    // 检查源文件是否已被修改
    try {
        auto ftime = std::filesystem::last_write_time(sourceFile);
        auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
            ftime - std::filesystem::file_time_type::clock::now() + 
            std::chrono::system_clock::now()
        );
        auto currentTime = std::chrono::system_clock::to_time_t(sctp);
        
        if (currentTime > storedTimestamp) {
            return Result<bool>::Ok(false);
        }
    } catch (const std::filesystem::filesystem_error&) {
        return Result<bool>::Ok(false);
    }
    
    return Result<bool>::Ok(true);
}

std::string CompilationCache::getCachePath(const std::string& hash) {
    return cacheDir_ + "/" + hash + ".o";
}

CompileError CompilationCache::makeError(const std::string& message) {
    return CompileError{
        ErrorKind::UnknownError,
        message,
        0, 0, ""
    };
}

} // namespace backend
} // namespace az