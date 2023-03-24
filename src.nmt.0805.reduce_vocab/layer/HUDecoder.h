
#pragma once
#include "HUBaseLayer.h"
#include "HUEmbeddingLayer.h"
#include "HUDecoderLayer.h"
#include "HUEncoderState.h"
#include "HUDecoderState.h"
#include "HUData.h"
#include <iostream>

namespace TenTrans{

class HUDecoder : public HUBaseLayer{

public:
	HUDecoder(HUPtr<HUConfig> options, HUPtr<HUMemPool> memoryPool, HUPtr<HUDevice> device, cnpy::npz_t modelNpz, bool isEncoder=false);
	~HUDecoder();
	void Init();
    HUPtr<HUDecoderState> PrepareForDecoding(HUPtr<HUBatch> batch, HUPtr<HUEncoderState> encoderState);



	// HUPtr<HUTensor> Forward(int position, HUPtr<HUTensor> encoderOutput, std::vector<size_t> &embIdx);
    // HUPtr<HUDecoderState> PrepareForDecoding(HUPtr<HUBatch> batch, HUPtr<HUEncoderState> encState);
	HUPtr<HUDecoderState> StartDecode_v2(HUPtr<HUBatch> batch, HUPtr<HUEncoderState> encState);
	void EmbeddingsFromPrediction(HUPtr<HUDecoderState> state, std::vector<size_t>& embIdx, HUPtr<HUEncoderState> encState, int beamSize);
	HUPtr<HUTensor> Step(HUPtr<HUDecoderState> state, States& decoderStates, int realDimBatch, uint8_t* isAllDone);

public:
	HUPtr<HUTensor> encoderContext_;
	HUPtr<HUTensor> encoderMask_;
	HUPtr<HUEmbeddingLayer> embedding_;
    // std::vector<int> lengths_;
    HUPtr<HUTensor> lengths_;
	std::vector<HUPtr<HUDecoderLayer>> decoderLayers_;
	int layers_;
    int heads_;
};

}

