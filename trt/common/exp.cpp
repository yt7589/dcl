#include <iostream>
#include <vector>
#include <algorithm>

const int BRAND_NUM = 3;
const int BMY_NUM = 5;
// batchSize = 2
std::vector<std::vector<int>> brandBmys = std::vector<std::vector<int>>{
    {0, 1},
    {1, 2, 3},
    {4}
};

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
        std::cout<<"Brand: in="<<in<<"; max="<<*maxPositionBrand<<"; idx="<<indexBrand<<std::endl;
        std::vector<int> bmyIdxs = brandBmys[indexBrand];
        float maxBmyVal = -0.1f;
        int maxBmyIdx = -1;
        for (int bmyIdx=0; bmyIdx<bmyIdxs.size(); bmyIdx++)
        {
            std::cout<<bmyIdxs[bmyIdx]<<std::endl;
            if (net_outputs[1][in*BMY_NUM + bmyIdxs[bmyIdx]] > maxBmyVal)
            {
                maxBmyIdx = bmyIdxs[bmyIdx];
                maxBmyVal = net_outputs[1][in*BMY_NUM + bmyIdxs[bmyIdx]];
            }
        }
        std::cout<<"idx="<<maxBmyIdx<<"; pos="<<maxBmyVal<<";"<<std::endl;
    }
    return 0;
}
