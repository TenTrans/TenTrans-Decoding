
#pragma once 

#include "HUBaseLayer.h"
#include "HUTensorOP.h"
#include "HUTensorUtil.h"

namespace TenTrans{

class HULayerNorm : public HUBaseLayer{
    
public:
    HULayerNorm(HUPtr<HUConfig> options, HUPtr<HUMemPool> memoryPool, HUPtr<HUDevice> device, cnpy::npz_t modelNpz, bool isEncoder, int layerId, bool isFFN, bool isContext=false);
    ~HULayerNorm();

    void Init();
	HUPtr<HUTensor> Forward(HUPtr<HUTensor> in, float eps=1e-12);
    /* LayerNorm((in + bias) + x), fuse residual network */
    HUPtr<HUTensor> AddBiasInputForward(HUPtr<HUTensor> in, HUPtr<HUTensor> x, const HUPtr<HUTensor> previousBias, float eps=1e-12);
    
public:
    std::string prefix_;
	HUPtr<HUTensor> ln_bias_;
	HUPtr<HUTensor> ln_scale_;

	int layerId_;                   // < 0: embedding layer normalization, >= 0: encoder/decoder layer normalization 
	bool isFFN_;                    // FFN layer ?
	bool isContext_;                // Self-Attention layer or Cross-Attention layer ?
};

}
