
#include "centernet_predictor_api.hpp"
#include "centernetpredictor.hpp"

void CenterNetPredictorAPI::set(std::unique_ptr<BasePredictor>& net)
{
	net = std::unique_ptr<BasePredictor>(new CenterNetPredictor());
}



void HelmetPredictorAPI::set(std::unique_ptr<BasePredictor>& net)
{
	net = std::unique_ptr<BasePredictor>(new HelmetPredictor());
}