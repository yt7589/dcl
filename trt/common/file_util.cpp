#include "file_util.hpp"

bool file2buffer(const std::string &modelPath, std::string& bytes)
{
    std::vector<char> tmp;
    auto re = file2buffer(modelPath, tmp);
    if (re)
    {
        bytes.assign(tmp.begin(), tmp.end());
        return true;
    } else
    {
        return false;
    }
}
