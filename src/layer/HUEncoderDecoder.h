
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
	HUPtr<HUDecoderState> Step(HUPtr<HUDecoderState> state, std::vector<size_t>& hypIndices, std::vector<size_t>& embIndices, int beamSize, const std::vector<uint8_t> &isAllDoneCopy, uint8_t* isAllDone);

    HUPtr<HUTensor> GetOutputLayerBias() { return output_->GetBias(); }; 
	
public:
	HUPtr<HUEncoder>     encoder_;
	HUPtr<HUDecoder>     decoder_;
	HUPtr<HUOutputLayer> output_;
};

}
