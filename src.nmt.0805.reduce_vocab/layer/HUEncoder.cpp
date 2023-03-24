
#include "HUEncoder.h"

namespace TenTrans{

HUEncoder::HUEncoder(HUPtr<HUConfig> options, HUPtr<HUMemPool> memoryPool, HUPtr<HUDevice> device, cnpy::npz_t modelNpz, bool isEncoder)
	: HUBaseLayer(options, memoryPool, device, modelNpz, isEncoder)
{
	LOG(debug, "[TenTrans][HUEncoder] Loading Source Embedding Layer ...");
	this->embedding_ = HUNew<HUEmbeddingLayer>(options, memoryPool, device, modelNpz, isEncoder);
	this->layers_ = options->get<int>("enc-depth");
    this->heads_ = options_->get<int>("transformer-heads");
	for(int i = 0; i < this->layers_; i++) {
        LOG(debug, "[TenTrans][HUEncoder] Loading Encoder Layer {}", i);
		HUPtr<HUEncoderLayer> layer = HUNew<HUEncoderLayer>(options, memoryPool, device,modelNpz, i);
		encoderLayers_.push_back(layer);
	}
}

HUEncoder::~HUEncoder()
{

}

void HUEncoder::Init()
{
	LOG(debug, "[TenTrans][HUEncoder] Initialize Source Embedding Layer ...");
	this->embedding_->Init();
	for(int i = 0; i < this->layers_; i++)
	{
		LOG(debug, "[TenTrans][HUEncoder] Initialize Encoder Layer {}", i);
		encoderLayers_[i]->Init();
	}
}

void HUEncoder::InitBuffer(EncoderSelfAttentionBuffer &params, int dimBatch, int dimSeqLen, int dimModel)
{
    int head_num = this->heads_;
    params.q_tmp = HUTensorUtil::Zeros({dimBatch, dimSeqLen, dimModel}, this->memPool_, this->device_);
    params.k_tmp = HUTensorUtil::Zeros({dimBatch, dimSeqLen, dimModel}, this->memPool_, this->device_);
    params.v_tmp = HUTensorUtil::Zeros({dimBatch, dimSeqLen, dimModel}, this->memPool_, this->device_);

    params.q_buf = HUTensorUtil::Zeros({dimBatch, head_num, dimModel/head_num, dimSeqLen}, this->memPool_, this->device_);
    params.k_buf = HUTensorUtil::Zeros({dimBatch, head_num, dimModel/head_num, dimSeqLen}, this->memPool_, this->device_);
    params.v_buf = HUTensorUtil::Zeros({dimBatch, head_num, dimSeqLen, dimModel/head_num}, this->memPool_, this->device_);

    params.qk_buf = HUTensorUtil::Zeros({dimBatch, head_num, dimSeqLen, dimSeqLen}, this->memPool_, this->device_);
    params.att_out_transpose_buf = HUTensorUtil::Zeros({dimBatch, head_num, dimSeqLen, dimModel/head_num},
            this->memPool_, this->device_);
}

void HUEncoder::DestroyBuffer(EncoderSelfAttentionBuffer &params)
{
    this->memPool_->free((params.q_tmp)->memory());
    this->memPool_->free((params.k_tmp)->memory());
    this->memPool_->free((params.v_tmp)->memory());

    this->memPool_->free((params.q_buf)->memory());
    this->memPool_->free((params.k_buf)->memory());
    this->memPool_->free((params.v_buf)->memory());

    this->memPool_->free((params.qk_buf)->memory());
    this->memPool_->free((params.att_out_transpose_buf)->memory());

}

HUPtr<HUEncoderState> HUEncoder::Forward(HUPtr<HUBatch> batch)
{
    /* batchEmbedding, [dimBatch, dimSteps, dimModel] */
    /* batchMask, [-4: dimBatch, -3: numHeads broadcast=1, -2: dimSteps broadcast=1, -1: dimSteps] */
    HUPtr<HUTensor> batchEmbedding, batchMask;
    this->embedding_->Forward(batch, batchEmbedding, batchMask);

    EncoderSelfAttentionBuffer params;
#ifdef ENCODER_SELF_ATTENTION_FUSION
    int dimBatch = batchEmbedding->shape()[-3];
    int dimSeqLen = batchEmbedding->shape()[-2];
    int dimModel = batchEmbedding->shape()[-1];
    int head_num = this->heads_;

    // EncoderSelfAttentionBuffer params;
    memset(&params, 0, sizeof(params));
    InitBuffer(params, dimBatch, dimSeqLen, dimModel);

    /*
    params.q_tmp = HUTensorUtil::Zeros({dimBatch, dimSeqLen, dimModel}, this->memPool_, this->device_);
    params.k_tmp = HUTensorUtil::Zeros({dimBatch, dimSeqLen, dimModel}, this->memPool_, this->device_);
    params.v_tmp = HUTensorUtil::Zeros({dimBatch, dimSeqLen, dimModel}, this->memPool_, this->device_);

    params.q_buf = HUTensorUtil::Zeros({dimBatch, head_num, dimModel/head_num, dimSeqLen}, this->memPool_, this->device_);
    params.k_buf = HUTensorUtil::Zeros({dimBatch, head_num, dimModel/head_num, dimSeqLen}, this->memPool_, this->device_);
    params.v_buf = HUTensorUtil::Zeros({dimBatch, head_num, dimSeqLen, dimModel/head_num}, this->memPool_, this->device);

    params.qk_buf = HUTensorUtil::Zeros({dimBatch, head_num, dimSeqLen, dimSeqLen}, this->memPool_, this->device_);
    params.att_out_transpose_buf = HUTensorUtil::Zeros({dimBatch, head_num, dimSeqLen, dimModel/head_num}, 
            this->memPool_, this->device_);
    */
#endif

    /* layer, [dimBatch, dimSteps, dimModel] */
    auto layer = batchEmbedding;
    for(int i = 0; i < this->layers_; ++i)
    {
        // LOG(trace, "[TenTrans][HUEncoder] Forward Encoder Layer {}, {}", i, layer->debug());
        layer = encoderLayers_[i]->Forward(layer, batchMask, params);
    }

#ifdef ENCODER_SELF_ATTENTION_FUSION
    DestroyBuffer(params);
#endif

    return HUNew<HUEncoderState>(layer, batchMask, batch);
}

}
