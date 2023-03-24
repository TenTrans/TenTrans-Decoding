#pragma once
#include "HUBaseLayer.h"
#include "HUMultiHeadAttention.h"
#include "HUFFNLayer.h"
#include "HULayerNorm.h"
#include <cuda_runtime.h>
#include <time.h>

namespace TenTrans{

class HUEncoderLayer : public HUBaseLayer{

public:
	HUEncoderLayer(HUPtr<HUConfig> options, HUPtr<HUMemPool> memoryPool, HUPtr<HUDevice> device, cnpy::npz_t modelNpz, int layerId, bool isEncoder=true);
	void Init();
	HUPtr<HUTensor> Forward(HUPtr<HUTensor> batchEmbedding, HUPtr<HUTensor> batchMask, EncoderSelfAttentionBuffer &params);

    HUPtr<HUTensor> Forward_V2(HUPtr<HUTensor> batchEmbedding, HUPtr<HUTensor> batchMask);
	~HUEncoderLayer();

public:
	HUPtr<HUMultiHeadAttention> attentionLayer_;
	HUPtr<HUFFNLayer> ffnLayer_;
	HUPtr<HULayerNorm> lnLayer1_; 
	HUPtr<HULayerNorm> lnLayer2_;
	int layerId_;
    bool isNormalizeBefore_;
};
}
