
#pragma once
#include "HUBaseLayer.h"

namespace TenTrans{

class HUOutputLayer: public HUBaseLayer 
{
public:
	HUOutputLayer(HUPtr<HUConfig> options, HUPtr<HUMemPool> memoryPool, HUPtr<HUDevice> device, cnpy::npz_t modelNpz, bool isEncoder=false);
	~HUOutputLayer();

	void Init();
	HUPtr<HUTensor> Forward(HUPtr<HUTensor> input);

    /* for kernel fusion */
    HUPtr<HUTensor> MultiplyW(HUPtr<HUTensor> input);
    HUPtr<HUTensor> GetBias() { return this->b_; };
		
public:
	HUPtr<HUTensor> W_;
	HUPtr<HUTensor> b_;
	string prefix_;
    bool isSharedOutEmbed_;
    bool isSharedAllEmbed_;
};

}
