#pragma once

#include <string>
#include <fstream>
#include <cassert>
#include <vector>

bool file2buffer(const std::string &modelPath, std::string &bytes);

template<typename T>
bool file2buffer(const std::string &modelPath, std::vector<T> &bytes)
{
    assert(sizeof(T) == 1);
    std::ifstream file(modelPath, std::ios::binary | std::ios::ate);

    if (!file.eof() && !file.fail()) {
        file.seekg(0, std::ios_base::end);
        std::streampos fileSize = file.tellg();
        bytes.resize(fileSize);

        file.seekg(0, std::ios_base::beg);
        file.read(static_cast<char*>(&bytes[0]), fileSize);
        return true;
    } else {
        return false;
    }
}



