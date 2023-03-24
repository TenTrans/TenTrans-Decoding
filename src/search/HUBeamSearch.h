#pragma once
#include <iostream>
#include "HUEncoderDecoder.h"
#include "HUHistory.h"
#include "HUNthElement.h"
#include "HUBeamCell.h"
#include <cuda_runtime.h>
namespace TenTrans
{

class HUBeamSearch
{

private:
	HUPtr<HUConfig> options_;
	HUPtr<HUEncoderDecoder> encdec_;
	int beamSize_;
	int trgEosId_ = -1;
	int trgUnkId_ = -1; 
    bool earlyStop_ = false;
	HUPtr<HUMemPool> memPool_;
	HUPtr<HUDevice> device_;

public:
	HUBeamSearch(HUPtr<HUConfig> options, HUPtr<HUEncoderDecoder> encdec, size_t beamSize, bool earlyStop, int trgEosId, int trgUnkId, HUPtr<HUMemPool> memoryPool, HUPtr<HUDevice> device)
      : options_(options), encdec_(encdec), beamSize_(beamSize), earlyStop_(earlyStop), trgEosId_(trgEosId), trgUnkId_(trgUnkId), memPool_(memoryPool), device_(device) {}

	HUHistories Search(HUPtr<HUBatch> batch);
    // void TopK(HUPtr<HUTensor> logProbs, const int K, HUPtr<HUTensor> topKIds, HUPtr<HUTensor> topKValues, HUPtr<HUTensor> logProbsTmp);
};

} 

