#pragma once
#include "predictor_api.hpp"


class FgvcPredictorAPI :public PredictorAPI//临时方案
{
public:
	struct Det{
    float x1; 
    float y1;
    float x2;
    float y2;
    float prob;
	int classes;
};
private:	

	virtual void set(std::unique_ptr<BasePredictor>& net) override;
};

class VgvcVehiclePredictorAPI :public FgvcPredictorAPI
{
	private:	

	virtual void set(std::unique_ptr<BasePredictor>& net) override;
};