#include "tensorrtwrapper.hpp"
#include "tensorrtpredictor.hpp"

#include <memory>



void TensorrtPredictor::setWrapper(std::unique_ptr<BaseWrapper>& pWrapper)
{
	pWrapper = std::unique_ptr<BaseWrapper>(new TensorRTWrapper()); 
}
