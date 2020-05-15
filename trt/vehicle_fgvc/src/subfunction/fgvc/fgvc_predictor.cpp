#include "centernetpredictor.hpp"

#include <iostream>

#include "api_global.hpp"
#include "fgvc_predictor_api.hpp"

#include "opencv2/core/core.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "imgutils.hpp"

using Det = FgvcPredictorAPI::Det;


void FgvcPredictor::setConfig(const InputConfig &iconfig, WrapperConfig &config) const
{
    /* 
	    constexpr static float mean[]= {0.408, 0.447, 0.470};
    constexpr static float std[] = {0.289, 0.274, 0.278};
	*/
    auto mean = 0.485*225;
    auto xFactor = 1. / (0.225 * 255.);
    mean = 0.0;
    xFactor = 1./255.;
    //config.inputLayers.push_back("input_images:0");


    /*
    output1   
    data 0       -1 2 224 224

    output1 3    -1 100               score 
    output2 5    -1 100                inds
    output3 4    -1 100             clses
    output4 6    -1 100                 ys
    output5 7    -1 100                xs

    output6 1    -1 2 56 56          reg
    output7 2    -1 2 56 56            wh
    */

    config.outputLayers = std::vector<std::string>{"output1","output2",
                                    "output3","output4",
                                    "output5","output6",
                                    "output7"}; //596 , 698
    //config.outputLayers.resize(1);        
    //config.outputLayers[0] = "517";                        
    config.inputShape = std::vector<int>{-1, 3, IMG_W, IMG_H};
    //config.modelInputType = 0; //0:  float; 1  uint8;
    config.meanBGR = std::vector<float>{0.408*255, 0.447*255, 0.470*255};
    config.xFactorBGR = std::vector<float>{1. / (0.289 * 255.), 1. / (0.274 * 255.), 1. / (0.278 * 255.)};
    config.meanBGR = std::vector<float>{mean, mean, mean};
    config.xFactorBGR = std::vector<float>{xFactor, xFactor, xFactor};
    config.swapBR = false;

    BasePredictor::setConfig(iconfig, config);
}




void FgvcPredictor::postProcess(const std::vector<cv::Mat> &images, \
std::vector<std::vector<float>> &net_outputs, \
std::vector<std::vector<float>> &out_results) const 
{

    float thres = 0.3;
    int topk = 100;
    std::cout << "postProcess " << std::endl;
    std::cout << " net_outputs[0] " << net_outputs[0].size() << std::endl;
    assert(net_outputs.size() == 7);

    const auto & reg = net_outputs[0];
    const auto & wh = net_outputs[1];
    const auto & score = net_outputs[2];
    const auto & clses = net_outputs[3];
    const auto & inds = net_outputs[4];
    const auto & ys = net_outputs[5];
    const auto & xs = net_outputs[6];

    int batchsize = inds.size()/topk;
    int heightwidth = reg.size()/batchsize/2;

    out_results.resize(batchsize);

    for (int batchindex = 0; batchindex < batchsize; ++batchindex)
    {
        std::cout << " batchindex " << batchindex << std::endl;
        auto score_ptr = score.data() + topk*(batchindex);
        auto reg_ptr = reg.data() + heightwidth*2*(batchindex);
        auto wh_ptr = wh.data()  + heightwidth*2*(batchindex);
        
        auto clses_ptr = clses.data()  + topk*(batchindex);
        int* inds_ptr = (int*)inds.data()  + topk*(batchindex);
        auto ys_ptr = ys.data() + topk*(batchindex);
        auto xs_ptr = xs.data() + topk*(batchindex);

        std::vector <Det> dets;
        for (int i = 0; i < topk; ++i)
        {
            if (score_ptr[i] <thres)
            {
                break; 
            }
            int index = inds_ptr[i];
            float xs_float = xs_ptr[i] + reg_ptr[index];
            float ys_float = ys_ptr[i] + reg_ptr[index+heightwidth];

            Det det;
            det.prob = score_ptr[i]; 
 
            det.x1 = 4*(xs_float - wh_ptr[index]/2.0);
            det.y1 = 4*(ys_float - wh_ptr[index+heightwidth]/2.0);
            det.x2 = 4*(xs_float + wh_ptr[index]/2.0);
            det.y2 = 4*(ys_float + wh_ptr[index+heightwidth]/2.0);
			det.classes = clses_ptr[i];

            dets.push_back(det);
        }
        out_results[batchindex].resize(dets.size()* sizeof(Det)/sizeof(float));
        memcpy((char*)out_results[batchindex].data(), (char*)dets.data(), dets.size()* sizeof(Det));
    }
}

std::vector<float> FgvcVehiclePredictor::preProcess(const std::vector<cv::Mat> &images,
									const WrapperConfig& conf) const
{
	auto imgs = ImgUtils::resizeAndPadding(images, conf.inputShape[2], conf.inputShape[3]);
	
	return BasePredictor::preProcess(imgs, conf);
}


void FgvcVehiclePredictor::postProcess(const std::vector<cv::Mat> &images, \
std::vector<std::vector<float>> &net_outputs, \
std::vector<std::vector<float>> &out_results) const 
{
	
	auto &img = images[0];
	int input_w = 512;
    int input_h = 512;
    float scale = std::min(float(input_w) / img.cols, float(input_h) / img.rows);
    float dx = (input_w - scale * img.cols) / 2;
    float dy = (input_h - scale * img.rows) / 2;
     

		
		
		

    float thres = 0.3;
    int topk = 100;
    std::cout << "postProcess " << std::endl;
    std::cout << " net_outputs[0] " << net_outputs[0].size() << std::endl;
    assert(net_outputs.size() == 7);

    const auto & reg = net_outputs[0];
    const auto & wh = net_outputs[1];
    const auto & score = net_outputs[2];
    const auto & clses = net_outputs[3];
    const auto & inds = net_outputs[4];
    const auto & ys = net_outputs[5];
    const auto & xs = net_outputs[6];

    int batchsize = inds.size()/topk;
    int heightwidth = reg.size()/batchsize/2;

    out_results.resize(batchsize);

    for (int batchindex = 0; batchindex < batchsize; ++batchindex)
    {
        std::cout << " batchindex " << batchindex << std::endl;
        auto score_ptr = score.data() + topk*(batchindex);
        auto reg_ptr = reg.data() + heightwidth*2*(batchindex);
        auto wh_ptr = wh.data()  + heightwidth*2*(batchindex);
        
        auto clses_ptr = (int*)clses.data()  + topk*(batchindex);
        int* inds_ptr = (int*)inds.data()  + topk*(batchindex);
        auto ys_ptr = ys.data() + topk*(batchindex);
        auto xs_ptr = xs.data() + topk*(batchindex);

        std::vector <Det> dets;
        for (int i = 0; i < topk; ++i)
        {
            if (score_ptr[i] <thres)
            {
                break; 
            }
            int index = inds_ptr[i];
            float xs_float = xs_ptr[i] + reg_ptr[index];
            float ys_float = ys_ptr[i] + reg_ptr[index+heightwidth];

            Det det;
            det.prob = score_ptr[i]; 
 
            det.x1 = 4*(xs_float - wh_ptr[index]/2.0);
            det.y1 = 4*(ys_float - wh_ptr[index+heightwidth]/2.0);
            det.x2 = 4*(xs_float + wh_ptr[index]/2.0);
            det.y2 = 4*(ys_float + wh_ptr[index+heightwidth]/2.0);
			
			
			det.x1 = (det.x1 - dx) / scale;
			det.y1 = (det.y1 - dy) / scale;
			det.x2 = (det.x2 - dx) / scale;
			det.y2 = (det.y2 - dy) / scale;
			
	
			det.classes = clses_ptr[i];

            dets.push_back(det);
        }
        out_results[batchindex].resize(dets.size()* sizeof(Det)/sizeof(float));
        memcpy((char*)out_results[batchindex].data(), (char*)dets.data(), dets.size()* sizeof(Det));
    }
}

void FgvcVehiclePredictor::setConfig(const InputConfig &iconfig, WrapperConfig &config) const
{
    /* 
	    constexpr static float mean[]= {0.408, 0.447, 0.470};
    constexpr static float std[] = {0.289, 0.274, 0.278};
	*/
    auto mean = 0.485*225;
    auto xFactor = 1. / (0.225 * 255.);
    mean = 0.0;
    xFactor = 1./255.;
    //config.inputLayers.push_back("input_images:0");


    /*
    output1   
    data 0       -1 2 224 224

    output1 3    -1 100               score 
    output2 5    -1 100                inds
    output3 4    -1 100             clses
    output4 6    -1 100                 ys
    output5 7    -1 100                xs

    output6 1    -1 2 56 56          reg
    output7 2    -1 2 56 56            wh
    */

    config.outputLayers = std::vector<std::string>{"output1","output2",
                                    "output3","output4",
                                    "output5","output6",
                                    "output7"}; //596 , 698
    //config.outputLayers.resize(1);        
    //config.outputLayers[0] = "517";                        
    config.inputShape = std::vector<int>{-1, 3, 512, 512};
    //config.modelInputType = 0; //0:  float; 1  uint8;
    config.meanBGR = std::vector<float>{0.408*255, 0.447*255, 0.470*255};
    config.xFactorBGR = std::vector<float>{1. / (0.289 * 255.), 1. / (0.274 * 255.), 1. / (0.278 * 255.)};
    config.meanBGR = std::vector<float>{mean, mean, mean};
    config.xFactorBGR = std::vector<float>{xFactor, xFactor, xFactor};
    config.swapBR = false;

    BasePredictor::setConfig(iconfig, config);
}