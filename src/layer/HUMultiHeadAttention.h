
#pragma once

#include "HUBaseLayer.h"
#include "HUTensorUtil.h"
#include "HUDecoderState.h"

namespace TenTrans{

enum QKVEnum {Wk, bk, Wv, bv, Wq, bq, Wo, bo};

class HUMultiHeadAttention : public HUBaseLayer {

public:
	HUMultiHeadAttention(HUPtr<HUConfig> options, HUPtr<HUMemPool> memoryPool, HUPtr<HUDevice> device, cnpy::npz_t modelNpz, bool isEncoder, int layerId);
    ~HUMultiHeadAttention();

	void Init();
	HUPtr<HUTensor> MultiHead(HUPtr<HUTensor> q, const HUPtr<HUTensor> &keys, const HUPtr<HUTensor> &values, const HUPtr<HUTensor> &mask, bool isContext=false);
    // void AddQKVBiasTranspose(HUPtr<HUTensor> &q, HUPtr<HUTensor> &keys, HUPtr<HUTensor> &values);

	HUPtr<HUTensor> SplitHeads(HUPtr<HUTensor> input);
	HUPtr<HUTensor> Attention(HUPtr<HUTensor> q, HUPtr<HUTensor> k, HUPtr<HUTensor> v, HUPtr<HUTensor> mask);
	// HUPtr<HUTensor> JoinHeads(HUPtr<HUTensor> input, int dimBeam = 1);
    HUPtr<HUTensor> JoinHeads(HUPtr<HUTensor> input);
	HUPtr<HUTensor> Forward(HUPtr<HUTensor> batchEmbedding, HUPtr<HUTensor> batchMask);
    HUPtr<HUTensor> ForwardFusedEncoderSelfAttention(HUPtr<HUTensor> batchEmbedding, HUPtr<HUTensor> batchMask, EncoderSelfAttentionBuffer &params);

	HUPtr<HUTensor> DecoderLayerSelfAttention(State& decoderLayerState, const State& prevdecoderLayerState, HUPtr<HUTensor> input, HUPtr<HUTensor> selfMask, int startPos, const std::vector<uint8_t> &isAllDoneCopy, uint8_t* isAllDone);
    HUPtr<HUTensor> DecoderLayerSelfAttention_V2(State& decoderLayerState, const State& prevdecoderLayerState, HUPtr<HUTensor> input, HUPtr<HUTensor> selfMask, int startPos);

    // HUPtr<HUTensor> DecoderLayerCrossAttention(State& decoderLayerState, const State& prevdecoderLayerState, HUPtr<HUTensor> q, const HUPtr<HUTensor> &memory, const HUPtr<HUTensor> &mask, int startPos);
   HUPtr<HUTensor> DecoderLayerCrossAttention(State& decoderLayerState, const State& prevdecoderLayerState, HUPtr<HUTensor> q, const HUPtr<HUTensor> &memory, const HUPtr<HUTensor> &mask, HUPtr<HUTensor> &lengths, int startPos, const std::vector<uint8_t> &isAllDoneCopy, uint8_t* isAllDone);

   HUPtr<HUTensor> GetSelfAttentionOutputBias() { return this->self_bo; };
   HUPtr<HUTensor> GetCrossAttentionOutputBias() { return this->context_bo; };

private:
	void InitBySuffix(QKVEnum e, string prefix);
	void NewBySuffix(QKVEnum e, string prefix);

    /* for self attention */
	HUPtr<HUTensor> self_Wk;
	HUPtr<HUTensor> self_bk;
	HUPtr<HUTensor> self_Wv;
	HUPtr<HUTensor> self_bv;
	HUPtr<HUTensor> self_Wq;
	HUPtr<HUTensor> self_bq;
	HUPtr<HUTensor> self_Wo;
	HUPtr<HUTensor> self_bo;

#ifdef SELF_ATTENTION_FUSION
    /* for self attention */
    HUPtr<HUTensor> self_Wqkv_fusion;
    HUPtr<HUTensor> self_bqkv_fusion;
#endif

    /* for target-source attention */
	HUPtr<HUTensor> context_Wk;
	HUPtr<HUTensor> context_bk;
	HUPtr<HUTensor> context_Wv;
	HUPtr<HUTensor> context_bv;
	HUPtr<HUTensor> context_Wq;
	HUPtr<HUTensor> context_bq;
	HUPtr<HUTensor> context_Wo;
	HUPtr<HUTensor> context_bo;
	
    int heads_;
	int layerId_;
	std::string self_;
	std::string context_;
   
    /*
    bool isExistTmpBuf_=false;
    HUPtr<HUTensor> beamLengths_;
    */
};

}
