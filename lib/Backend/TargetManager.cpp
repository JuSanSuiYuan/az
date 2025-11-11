// AZ编译器 - 目标管理器实现

#include "AZ/Backend/TargetManager.h"

#include "llvm/ADT/Triple.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/Host.h"

namespace az {
namespace backend {

TargetManager::TargetManager() {
    // 初始化所有目标
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();
}

Result<std::unique_ptr<llvm::TargetMachine>> TargetManager::createTargetMachine(
    const std::string& targetTriple,
    const std::string& cpu,
    const std::string& features,
    llvm::CodeGenOpt::Level optLevel,
    llvm::Reloc::Model relocModel,
    llvm::CodeModel::Model codeModel) {
    
    // 使用指定的目标或本机目标
    std::string triple = targetTriple.empty() 
        ? llvm::sys::getDefaultTargetTriple() 
        : targetTriple;

    // 查找目标
    std::string error;
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget(triple, error);
    if (!target) {
        return Result<std::unique_ptr<llvm::TargetMachine>>::Err(
            makeError("无法找到目标: " + error));
    }

    // 创建目标机器
    llvm::TargetOptions targetOptions;
    
    // 设置重定位模型
    if (relocModel == llvm::Reloc::Model::PIC_) {
        targetOptions.RelaxELFRelocations = true;
    }
    
    // 获取CPU和特性
    std::string targetCPU = cpu;
    std::string targetFeatures = features;
    
    // 如果未指定CPU，尝试获取目标的默认CPU
    if (targetCPU.empty()) {
        if (triple == llvm::sys::getDefaultTargetTriple()) {
            // 本机目标，使用主机CPU
            targetCPU = llvm::sys::getHostCPUName().str();
        } else {
            // 非本机目标，使用通用CPU
            targetCPU = "generic";
        }
    }
    
    // 创建目标机器
    auto targetMachine = std::unique_ptr<llvm::TargetMachine>(
        target->createTargetMachine(
            triple,
            targetCPU,
            targetFeatures,
            targetOptions,
            relocModel,
            codeModel,
            optLevel
        )
    );

    if (!targetMachine) {
        return Result<std::unique_ptr<llvm::TargetMachine>>::Err(
            makeError("无法创建目标机器"));
    }

    return Result<std::unique_ptr<llvm::TargetMachine>>::Ok(
        std::move(targetMachine));
}

Result<std::vector<std::string>> TargetManager::getAvailableTargets() {
    std::vector<std::string> targets;
    
    // 获取所有可用目标
    for (auto it = llvm::TargetRegistry::targets().begin();
         it != llvm::TargetRegistry::targets().end();
         ++it) {
        targets.push_back(it->getName());
    }
    
    return Result<std::vector<std::string>>::Ok(std::move(targets));
}

Result<std::string> TargetManager::getNativeTargetTriple() {
    return Result<std::string>::Ok(llvm::sys::getDefaultTargetTriple());
}

Result<std::string> TargetManager::getNativeCPU() {
    return Result<std::string>::Ok(llvm::sys::getHostCPUName().str());
}

Result<std::string> TargetManager::getNativeFeatures() {
    // 获取本机特性
    llvm::StringMap<bool> hostFeatures;
    if (llvm::sys::getHostCPUFeatures(hostFeatures)) {
        std::string features;
        for (const auto& feature : hostFeatures) {
            if (feature.second) {
                if (!features.empty()) {
                    features += ",";
                }
                features += "+" + feature.first().str();
            }
        }
        return Result<std::string>::Ok(features);
    }
    return Result<std::string>::Ok("");
}

bool TargetManager::isTargetSupported(const std::string& targetName) {
    std::string error;
    return llvm::TargetRegistry::lookupTarget(targetName, error) != nullptr;
}

Result<llvm::Triple> TargetManager::parseTriple(const std::string& tripleStr) {
    llvm::Triple triple(tripleStr);
    
    // 验证三元组
    if (triple.getArch() == llvm::Triple::UnknownArch) {
        return Result<llvm::Triple>::Err(
            makeError("未知的架构: " + tripleStr));
    }
    
    return Result<llvm::Triple>::Ok(triple);
}

std::string TargetManager::getTargetArchName(llvm::Triple::ArchType arch) {
    switch (arch) {
    case llvm::Triple::x86:
        return "x86";
    case llvm::Triple::x86_64:
        return "x86_64";
    case llvm::Triple::arm:
        return "arm";
    case llvm::Triple::aarch64:
        return "aarch64";
    case llvm::Triple::mips:
        return "mips";
    case llvm::Triple::mips64:
        return "mips64";
    case llvm::Triple::powerpc:
        return "powerpc";
    case llvm::Triple::powerpc64:
        return "powerpc64";
    case llvm::Triple::sparc:
        return "sparc";
    case llvm::Triple::sparcv9:
        return "sparcv9";
    case llvm::Triple::systemz:
        return "systemz";
    default:
        return "unknown";
    }
}

std::string TargetManager::getTargetOSName(llvm::Triple::OSType os) {
    switch (os) {
    case llvm::Triple::Linux:
        return "linux";
    case llvm::Triple::Win32:
        return "windows";
    case llvm::Triple::MacOSX:
        return "macos";
    case llvm::Triple::FreeBSD:
        return "freebsd";
    case llvm::Triple::OpenBSD:
        return "openbsd";
    case llvm::Triple::NetBSD:
        return "netbsd";
    case llvm::Triple::DragonFly:
        return "dragonfly";
    case llvm::Triple::Solaris:
        return "solaris";
    case llvm::Triple::AIX:
        return "aix";
    default:
        return "unknown";
    }
}

std::vector<std::string> TargetManager::getSupportedCPUList(const std::string& targetTriple) {
    // 这是一个简化的实现，实际项目中可能需要更复杂的逻辑
    llvm::Triple triple(targetTriple);
    
    switch (triple.getArch()) {
    case llvm::Triple::x86:
    case llvm::Triple::x86_64:
        return {"generic", "i386", "i486", "i586", "i686", "pentium", "pentium-mmx", 
                "pentiumpro", "pentium2", "pentium3", "pentium4", "prescott", "nocona", 
                "core2", "penryn", "bonnell", "atom", "silvermont", "slm", "goldmont", 
                "goldmont-plus", "tremont", "nehalem", "corei7", "westmere", "sandybridge", 
                "corei7-avx", "ivybridge", "core-avx-i", "haswell", "core-avx2", "broadwell", 
                "skylake", "skylake-avx512", "skx", "cascadelake", "cooperlake", "icelake-client", 
                "icelake-server", "tigerlake", "sapphirerapids", "knl", "knm", "lakemont", 
                "k6", "k6-2", "k6-3", "athlon", "athlon-tbird", "athlon-4", "athlon-xp", 
                "athlon-mp", "k8", "opteron", "athlon64", "athlon-fx", "k8-sse3", "opteron-sse3", 
                "athlon64-sse3", "amdfam10", "barcelona", "bdver1", "bdver2", "bdver3", "bdver4", 
                "btver1", "btver2", "x86-64", "geode"};
    
    case llvm::Triple::arm:
        return {"generic", "armv2", "armv2a", "armv3", "armv3m", "armv4", "armv4t", "armv5", 
                "armv5t", "armv5te", "armv5tej", "armv6", "armv6j", "armv6k", "armv6z", 
                "armv6zk", "armv6t2", "armv6m", "armv7", "armv7a", "armv7r", "armv7m", 
                "armv7em", "armv7s", "armv7k", "armv8", "armv8a", "armv8r", "armv8m.base", 
                "armv8m.main", "armv8.1a", "armv8.2a", "armv8.3a", "armv8.4a", "armv8.5a", 
                "armv8.6a", "ep9312", "strongarm", "strongarm110", "strongarm1100", "strongarm1110", 
                "xscale", "iwmmxt", "xscale", "arm1136jf-s", "arm1156t2f-s", "arm1176jzf-s", 
                "mpcore", "mpcorenovfp", "cortex-a5", "cortex-a7", "cortex-a8", "cortex-a9", 
                "cortex-a12", "cortex-a15", "cortex-a17", "cortex-a32", "cortex-a35", "cortex-a53", 
                "cortex-a57", "cortex-a72", "cortex-a73", "cortex-a75", "cortex-a76", "cortex-a76ae", 
                "cortex-a77", "cortex-a78", "cortex-a78ae", "cortex-a78c", "cortex-a710", 
                "cortex-a715", "cortex-x1", "cortex-x2", "cortex-x3", "neoverse-n1", "neoverse-e1", 
                "neoverse-n2", "neoverse-v1", "neoverse-v2", "sc000", "sc300", "cortex-m0", 
                "cortex-m0plus", "cortex-m1", "cortex-m3", "cortex-m4", "cortex-m7", "cortex-m23", 
                "cortex-m33", "cortex-m35p", "cortex-m55", "cortex-m85", "cortex-r4", "cortex-r4f", 
                "cortex-r5", "cortex-r7", "cortex-r8", "cortex-r52", "cortex-r52+", "cortex-r82"};
    
    case llvm::Triple::aarch64:
        return {"generic", "cortex-a34", "cortex-a35", "cortex-a53", "cortex-a55", "cortex-a57", 
                "cortex-a65", "cortex-a65ae", "cortex-a72", "cortex-a73", "cortex-a75", "cortex-a76", 
                "cortex-a76ae", "cortex-a77", "cortex-a78", "cortex-a78ae", "cortex-a78c", 
                "cortex-a710", "cortex-a715", "cortex-x1", "cortex-x2", "cortex-x3", "cyclone", 
                "exynos-m3", "exynos-m4", "exynos-m5", "falkor", "kryo", "neoverse-n1", "neoverse-n2", 
                "neoverse-e1", "neoverse-v1", "neoverse-v2", "saphira", "thunderx", "thunderx2t99", 
                "thunderx3t110", "thunderxt81", "thunderxt83", "thunderxt88", "tsv110"};
    
    default:
        return {"generic"};
    }
}

Result<std::vector<std::string>> TargetManager::getTargetFeatures(const std::string& targetTriple) {
    // 获取目标支持的特性
    llvm::Triple triple(targetTriple);
    
    // 这是一个简化的实现，实际项目中可能需要查询目标的具体特性
    switch (triple.getArch()) {
    case llvm::Triple::x86:
    case llvm::Triple::x86_64:
        return Result<std::vector<std::string>>::Ok(
            std::vector<std::string>{"sse", "sse2", "sse3", "ssse3", "sse4.1", "sse4.2", 
                                   "avx", "avx2", "avx512f", "avx512cd", "avx512bw", 
                                   "avx512dq", "avx512vl", "popcnt", "bmi", "bmi2"});
    
    case llvm::Triple::arm:
        return Result<std::vector<std::string>>::Ok(
            std::vector<std::string>{"neon", "vfp2", "vfp3", "vfp4", "thumb", "thumb2"});
    
    case llvm::Triple::aarch64:
        return Result<std::vector<std::string>>::Ok(
            std::vector<std::string>{"neon", "crc", "crypto", "fullfp16", "fp16fml", 
                                   "lse", "rdm", "sha2", "sha3", "sm4"});
    
    default:
        return Result<std::vector<std::string>>::Ok(std::vector<std::string>());
    }
}

CompileError TargetManager::makeError(const std::string& message) {
    return CompileError{
        ErrorKind::UnknownError,
        message,
        0, 0, ""
    };
}

} // namespace backend
} // namespace az