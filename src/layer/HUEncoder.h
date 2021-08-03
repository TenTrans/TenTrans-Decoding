#pragma once
#include "HUBaseLayer.h"
#include "HUEmbeddingLayer.h"
#include "HUEncoderLayer.h"
#include "HUEncoderState.h"
#include <iostream>

namespace TenTrans{

/*
struct EncoderSelfAttentionBuffer {
    HUPtr<HUTensor> q_tmp, k_tmp, v_tmp;
    HUPtr<HUTensor> q_buf, k_buf, v_buf;
    HUPtr<HUTensor> qk_buf;
    HUPtr<HUTensor> att_out_transpose_buf;
}; */

class HUEncoder : public HUBaseLayer{

public:
	HUEncoder(HUPtr<HUConfig> options, HUPtr<HUMemPool> memoryPool, HUPtr<HUDevice> device, cnpy::npz_t modelNpz, bool isEncoder=true);
	~HUEncoder();
	void Init();
	HUPtr<HUEncoderState> Forward(HUPtr<HUBatch> batch);
    // void Forward_test(HUPtr<HUBatch> batch);
    //

public:
	HUPtr<HUEmbeddingLayer> embedding_;
	std::vector<HUPtr<HUEncoderLayer>> encoderLayers_;
	int layers_;
    int heads_;

    void InitBuffer(EncoderSelfAttentionBuffer &params, int dimBatch, int dimSeqLen, int dimModel);
    void DestroyBuffer(EncoderSelfAttentionBuffer &params);
};

}

