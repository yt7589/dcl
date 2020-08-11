#include <iostream>
#include <vector>
#include <algorithm>

const int BRAND_NUM = 3;
const int BMY_NUM = 5;
// batchSize = 2

int main()
{
    // prepare data
    std::vector<std::vector<float>> net_outputs = std::vector<std::vector<float>>{};
    std::vector<float> brand_outputs = std::vector<float>{0.1, 0.7, 0.2, 0.05, 0.05, 0.9};
    net_outputs.emplace_back(brand_outputs);
    std::vector<float> bmy_outputs = std::vector<float>{0.1, 0.1, 0.6, 0.1, 0.1, 0.2, 0.2, 0.05, 0.05, 0.5};
    net_outputs.emplace_back(bmy_outputs);
    //
    for (int in=0; in < net_outputs[0].size() / BRAND_NUM; ++in)
    {
        auto maxPositionBrand = std::max_element(net_outputs[0].begin() + in * BRAND_NUM, net_outputs[0].begin() + (in + 1) * BRAND_NUM);
        auto indexBrand = maxPositionBrand - (net_outputs[0].begin() + in * BRAND_NUM);
        std::cout<<"in="<<in<<"; max="<<*maxPositionBrand<<"; idx="<<indexBrand<<std::endl;
    }
    return 0;
}
