
#pragma once
#include <iostream>
#include "HUBaseLayer.h"
#include "HUEncoder.h"
#include "HUDecoder.h"
#include "HUOutputLayer.h"

#include <cuda_runtime.h>

namespace TenTrans
{

class HUEncoderDecoder : public HUBaseLayer 
{
public:
	HUEncoderDecoder(HUPtr<HUConfig> options, HUPtr<HUMemPool> memoryPool, HUPtr<HUDevice> device, cnpy::npz_t modelNpz, bool isEncoder=false);
	~HUEncoderDecoder();

	void Init();
    HUPtr<HUDecoderState> PrepareForDecoding(HUPtr<HUBatch> batch);
	HUPtr<HUDecoderState> Step(HUPtr<HUDecoderState> state, size_t* hypIndices, int selIdxSize, std::vector<size_t>& embIndices, int beamSize, int realDimBatch, uint8_t* isAllDone, const std::vector<size_t>& tgtEmbIdx, bool isFirstStep);

    HUPtr<HUTensor> GetOutputLayerBias() { return output_->GetBias(); }
    HUPtr<HUTensor> GetOutputLayerW() { return output_->GetW(); }
	
public:
	HUPtr<HUEncoder>     encoder_;
	HUPtr<HUDecoder>     decoder_;
	HUPtr<HUOutputLayer> output_;
};

}
