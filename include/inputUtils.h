#pragma once

#include <string>

enum class InputType { IMAGE, VIDEO, STREAM, UNKNOWN };

InputType detectInputType(const std::string& path);