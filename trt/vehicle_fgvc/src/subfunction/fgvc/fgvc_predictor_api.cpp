
#include "fgvc_predictor_api.hpp"
#include "fgvc_predictor.hpp"

void FgvcPredictorAPI::set(std::unique_ptr<BasePredictor>& net)
{
	net = std::unique_ptr<BasePredictor>(new FgvcPredictor());
}



void FgvcVehiclePredictorAPI::set(std::unique_ptr<BasePredictor>& net)
{
	net = std::unique_ptr<BasePredictor>(new FgvcVehiclePredictor());
}