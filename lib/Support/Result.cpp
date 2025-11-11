// AZ编译器 - Result类型实现

#include "AZ/Support/Result.h"
#include <iostream>

namespace az {
namespace support {

void CompileError::report() const {
    const char* kindName = "Unknown";
    switch (kind) {
        case ErrorKind::LexerError:
            kindName = "词法错误";
            break;
        case ErrorKind::ParserError:
            kindName = "语法错误";
            break;
        case ErrorKind::SemanticError:
            kindName = "语义错误";
            break;
        case ErrorKind::TypeError:
            kindName = "类型错误";
            break;
        case ErrorKind::RuntimeError:
            kindName = "运行时错误";
            break;
        case ErrorKind::IOError:
            kindName = "IO错误";
            break;
        case ErrorKind::UnknownError:
            kindName = "未知错误";
            break;
    }
    
    std::cerr << "[错误] " << kindName << " 在 " << filename
              << ":" << line << ":" << column << std::endl;
    std::cerr << "  " << message << std::endl;
}

} // namespace support
} // namespace az
