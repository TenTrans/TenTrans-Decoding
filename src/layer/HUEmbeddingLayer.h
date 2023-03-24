#pragma once
#include "HUBaseLayer.h"
#include "HUGlobal.h"
#include "HUTensor.h"
#include "HUTensorOP.h"
#include "HUTensorUtil.h"
#include "HULayerNorm.h"
#include <string>
#include <math.h>
#include <iostream>

namespace TenTrans
{
class HUEmbeddingLayer: public HUBaseLayer 
{
public:
	HUEmbeddingLayer(HUPtr<HUConfig> options, HUPtr<HUMemPool> memoryPool, HUPtr<HUDevice> device, cnpy::npz_t modelNpz, bool isEncoder);
	~HUEmbeddingLayer();

	void Forward(HUPtr<HUBatch> batch, HUPtr<HUTensor> &batchEmbedding, HUPtr<HUTensor> &batchMask);
	HUPtr<HUTensor> AddPositinoalEmbeddings(HUPtr<HUTensor> input, int start=0);

    HUPtr<HUTensor> ForwardDecoder(HUPtr<HUTensor> encoderOutput, std::vector<size_t> &embIdx, int beamSize, size_t startPos);
    HUPtr<HUTensor> ForwardDecoder_V2(HUPtr<HUTensor> encoderOutput, std::vector<size_t> &embIdx, int beamSize);
    HUPtr<HUTensor> ForwardLayerNorm(HUPtr<HUTensor> input);

	void Init();
    void InitPosEmbeddings();
		
public:
    HUPtr<HUTensor> wordEmbedding_;           // word embedding
    HUPtr<HUTensor> posEmbedding_;            // positional embedding (sinusoidal or learned)
    HUPtr<HULayerNorm> lnLayer_;              // embedding layer normalization

    bool useEmbedScale_;
    bool isSharedEmbed_;
};

}
