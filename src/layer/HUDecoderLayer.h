
#pragma once
#include<iostream>
#include "HUBaseLayer.h"
#include "HUMultiHeadAttention.h"
#include "HUFFNLayer.h"
#include "HULayerNorm.h"
#include "HUDecoderState.h"

namespace TenTrans{

class HUDecoderLayer : public HUBaseLayer{

public:
	HUDecoderLayer(HUPtr<HUConfig> options, HUPtr<HUMemPool> memoryPool, HUPtr<HUDevice> device, cnpy::npz_t modelNpz, int layerId, bool isEncoder=false);
	void Init();
	HUPtr<HUTensor> Forward(HUPtr<HUTensor> embedding, HUPtr<HUTensor> selfAttMask, State& decoderState, State prevDecoderState, int position, HUPtr<HUTensor> encoderContext, HUPtr<HUTensor> encoderMask, HUPtr<HUTensor> lengths, const std::vector<uint8_t> &isAllDoneCopy, uint8_t* isAllDone);

    HUPtr<HUTensor> Forward_V2(HUPtr<HUTensor> embedding, HUPtr<HUTensor> selfAttMask, State& decoderState, State prevDecoderState, int position, HUPtr<HUTensor> encoderContext, HUPtr<HUTensor> encoderMask, HUPtr<HUTensor> lengths, const std::vector<uint8_t> &isAllDoneCopy, uint8_t* isAllDone);
	~HUDecoderLayer();

public:
    HUPtr<HUMultiHeadAttention> attentionLayer_;    // self-attention & target-source attention
    HUPtr<HUFFNLayer> ffnLayer_;                    // FFN layer
    HUPtr<HULayerNorm> lnLayer1_;                   // layer-norm after self-attention
    HUPtr<HULayerNorm> lnLayer2_;                   // layer-norm after target-source attention
	HUPtr<HULayerNorm> lnLayer3_;                   // layer-norm after FFN layer
    int layerId_;
    bool isNormalizeBefore_;
};

}

