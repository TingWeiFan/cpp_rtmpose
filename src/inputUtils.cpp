#include "inputUtils.h"

#include <algorithm>
#include <cctype>
#include <filesystem>

InputType detectInputType(const std::string& path) {
    if (path.rfind("rtsp://", 0) == 0 || path.rfind("rtmp://", 0) == 0 ||
        path.rfind("http://", 0) == 0 || path.rfind("https://", 0) == 0) {
        return InputType::STREAM;
    }

    // 是否為純數字字串
    if (!path.empty() &&
        std::all_of(path.begin(), path.end(), [](unsigned char c) { return std::isdigit(c); })) {
        return InputType::STREAM;
    }

    // 取副檔名
    std::string ext = std::filesystem::path(path).extension().string();

    // 將指定範圍內的每一個元素經過某種運算後，將結果寫入指定位置
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
        return InputType::IMAGE;
    }

    if (ext == ".mp4" || ext == ".avi" || ext == ".mov" || ext == ".mkv") {
        return InputType::VIDEO;
    }

    return InputType::UNKNOWN;
}
