
#include "HULayerNorm.h"

namespace TenTrans{

HULayerNorm::HULayerNorm(HUPtr<HUConfig> options, HUPtr<HUMemPool> memoryPool, HUPtr<HUDevice> device, cnpy::npz_t modelNpz, bool isEncoder, int layerId, bool isFFN, bool isContext)
	: HUBaseLayer(options, memoryPool, device, modelNpz, isEncoder) {
	this->layerId_ = layerId;
	this->isFFN_ = isFFN;
	this->isContext_ = isContext;

	if(this->isEncoder_)
	{
        /* wheather FFN layer normalization or Self-Attention layer normalization */
		if(this->isFFN_) 
        { 
            this->prefix_ = "encoder.layers." + std::to_string(this->layerId_) + ".ffn_layer_norm.";
        }
		else 
        {
            this->prefix_ = "encoder.layers." + std::to_string(this->layerId_) + ".att_layer_norm.";
        }

        /* Embedding layer normalization */
        if (this->layerId_ < 0) {
            this->prefix_ = "encoder.embed_norm.";
        }
	}
	else
	{
        /* FFN layer normalization */
		if(this->isFFN_) 
        {
            this->prefix_ = "decoder.layers." + std::to_string(this->layerId_) + ".ffn_layer_norm.";
        }
		else
		{
            /* wheather Self-Attention layer normalization or Cross-Attention layer normalization */
			if(this->isContext_) 
            {
                this->prefix_ = "decoder.layers." + std::to_string(this->layerId_) + ".src_tgt_att_layer_norm.";
            }
			else 
            {
                this->prefix_ = "decoder.layers." + std::to_string(this->layerId_) + ".tgt_att_layer_norm.";
            }
		}

        /* Embedding layer normalization */
        if (this->layerId_ < 0) {
            this->prefix_ = "decoder.embed_norm.";
        }                                   
	}
	
}

HULayerNorm::~HULayerNorm()
{
    this->memPool_->free(this->ln_bias_->memory());
    this->memPool_->free(this->ln_scale_->memory());
}

void HULayerNorm::Init()
{
	string scale = this->prefix_ + "weight";
	string bias = this->prefix_ + "bias";
	auto scaleNp = modelNpz_[scale];
	auto biasNp = modelNpz_[bias];
	auto scaleShape = GetShapeByModel(scale, this->modelNpz_);
	auto biasShape = GetShapeByModel(bias, this->modelNpz_);
	auto scaleMem = this->memPool_->alloc<float>(scaleShape->elements());
	auto biasMem = this->memPool_->alloc<float>(biasShape->elements());

	this->ln_bias_ = HUNew<HUTensor>(biasMem, *biasShape, device_);
	this->ln_scale_ = HUNew<HUTensor>(scaleMem, *scaleShape, device_);

	size_t size = 1;
	for(size_t dim : scaleNp->shape) {
		size *= dim;
    }
	this->ln_scale_->set((float*)scaleNp->data(), (float*)scaleNp->data() + size);
#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][LayerNorm] Loading {} parameters, {}, {}", scale, scaleShape->toString(), this->ln_scale_->debug());
#endif

	size = 1;
	for(size_t dim : scaleNp->shape) {
		size *= dim;
    }
	this->ln_bias_->set((float*)biasNp->data(), (float*)biasNp->data() + size);
#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][LayerNorm] Loading {} parameters, {}, {}", bias, biasShape->toString(), this->ln_bias_->debug());
#endif    
}

HUPtr<HUTensor> HULayerNorm::Forward(HUPtr<HUTensor> in, float eps)
{
    HUPtr<HUTensor> out = HUTensorUtil::LayerNormalization(in, this->ln_scale_, this->memPool_, \ 
            this->device_, this->ln_bias_, eps);
	return out;
}

/*
 * (in + bias) + x -> in
 * >> layer_norm((in + bias) + x)
 *
 */
HUPtr<HUTensor> HULayerNorm::AddBiasInputForward(HUPtr<HUTensor> in, HUPtr<HUTensor> x, const HUPtr<HUTensor> previousBias, float eps)
{
    HUPtr<HUTensor> out = HUTensorUtil::AddBiasInputLayerNormalization(in, x, previousBias, this->ln_scale_, \ 
            this->memPool_, this->device_, this->ln_bias_, eps);
    return out;
}

}
