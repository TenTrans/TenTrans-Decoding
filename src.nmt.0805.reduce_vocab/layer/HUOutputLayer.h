
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
    HUPtr<HUTensor> MultiplyW(HUPtr<HUTensor> input, const std::vector<size_t> &tgtEmbIdx, bool isFirstTime=false);
    HUPtr<HUTensor> GetBias() { return this->useShortList_ ? this->new_b_ : this->b_; };
    HUPtr<HUTensor> GetW() { return this->useShortList_ ? this->new_W_ : this->W_; };
		
public:
	HUPtr<HUTensor> W_;
	HUPtr<HUTensor> b_;
	string prefix_;
    bool isSharedOutEmbed_;
    bool isSharedAllEmbed_;

    // vocabulary selection
    bool useShortList_ = false;
    HUPtr<HUTensor> new_W_;
    HUPtr<HUTensor> new_b_;
};

}
