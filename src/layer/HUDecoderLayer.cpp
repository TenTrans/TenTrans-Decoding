
#include "HUDecoderLayer.h"

namespace TenTrans{

HUDecoderLayer::HUDecoderLayer(HUPtr<HUConfig> options, HUPtr<HUMemPool> memoryPool, HUPtr<HUDevice> device, cnpy::npz_t modelNpz, int layerId, bool isEncoder) 
    : HUBaseLayer(options, memoryPool, device, modelNpz, isEncoder)
{
#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][HUDecoderLayer] Loading AttentionLayer ...");
#endif
    this->attentionLayer_ = HUNew<HUMultiHeadAttention>(options, memoryPool, device, modelNpz, isEncoder, layerId);

#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][HUDecoderLayer] Loading FFNLayer ...");
#endif
    this->ffnLayer_ = HUNew<HUFFNLayer>(options, memoryPool, device, modelNpz, isEncoder, layerId);

#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][HUDecoderLayer] Loading LayerNorm1 ...");
#endif
    this->lnLayer1_ = HUNew<HULayerNorm>(options, memoryPool, device, modelNpz, isEncoder, layerId, false); 

#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][HUDecoderLayer] Loading LayerNorm2 ...");
#endif
    this->lnLayer2_ = HUNew<HULayerNorm>(options, memoryPool, device, modelNpz, isEncoder, layerId, false, true);

#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][HUDecoderLayer] Loading LayerNorm3 ...\n");
#endif
	this->lnLayer3_ = HUNew<HULayerNorm>(options, memoryPool, device, modelNpz, isEncoder, layerId, true);

    this->layerId_ = layerId;
    // this->isNormalizeBefore_= true;
    this->isNormalizeBefore_ = this->options_->get<bool>("normalize-before");
}

void HUDecoderLayer::Init()
{
	this->attentionLayer_->Init();
	this->ffnLayer_->Init();
	this->lnLayer1_->Init();
	this->lnLayer2_->Init();
	this->lnLayer3_->Init();
}

HUDecoderLayer::~HUDecoderLayer()
{

}

HUPtr<HUTensor> HUDecoderLayer::Forward(HUPtr<HUTensor> embedding, HUPtr<HUTensor> selfAttMask, State& decoderState, State prevDecoderState, int position, HUPtr<HUTensor> encoderContext, HUPtr<HUTensor> encoderMask, HUPtr<HUTensor> lengths, const std::vector<uint8_t> &isAllDoneCopy, uint8_t* isAllDone)
{
    HUPtr<HUTensor> output;
    if (this->isNormalizeBefore_)  // pre-norm
    {
#ifdef BASIC_KERNEL_FUSION
        // 1. self_attention(layer_norm(x))
        auto ln1Output = this->lnLayer1_->Forward(embedding, 1e-12);
        auto selfAttention = this->attentionLayer_->DecoderLayerSelfAttention(decoderState, prevDecoderState, 
                ln1Output, selfAttMask, position, isAllDoneCopy, isAllDone);
        this->memPool_->free(ln1Output->memory());

        // 2. add_bias_input_layernorm
        // resnet: selfAttention + bias + x -> selfAttention
        auto ln2Output = this->lnLayer2_->AddBiasInputForward(selfAttention, embedding, 
                this->attentionLayer_->GetSelfAttentionOutputBias(), 1e-12);
        this->memPool_->free(embedding->memory());

        // 3. cross_attention
        HUPtr<HUTensor> contextAttention = this->attentionLayer_->DecoderLayerCrossAttention(decoderState, prevDecoderState, 
                ln2Output, encoderContext, encoderMask, lengths, position, isAllDoneCopy, isAllDone);
        this->memPool_->free(ln2Output->memory());

        // 4. add_bias_input_layernorm
        // resnet: contextAttention + bias + x -> contextAttention 
        auto ln3Output = this->lnLayer3_->AddBiasInputForward(contextAttention, selfAttention, 
                this->attentionLayer_->GetCrossAttentionOutputBias(), 1e-12);
        this->memPool_->free(selfAttention->memory());

        // 5. ffn_layer
        output = this->ffnLayer_->Forward(ln3Output);
        this->memPool_->free(ln3Output->memory());

        // 6. add_bias_input
        HUTensorUtil::AddBiasInput(output, this->ffnLayer_->GetBias2(), contextAttention);
        this->memPool_->free(contextAttention->memory());
#else
        auto ln1Output = this->lnLayer1_->Forward(embedding, 1e-12);
        auto selfAttention = this->attentionLayer_->DecoderLayerSelfAttention(decoderState, prevDecoderState, 
                ln1Output, selfAttMask, position, isAllDoneCopy, isAllDone);
        this->memPool_->free(ln1Output->memory());
        auto resOutput = HUTensorUtil::Plus(selfAttention, embedding, this->memPool_, device_);
        this->memPool_->free(embedding->memory());
        this->memPool_->free(selfAttention->memory());

        auto ln2Output = this->lnLayer2_->Forward(resOutput, 1e-12);
        HUPtr<HUTensor> contextAttention = this->attentionLayer_->DecoderLayerCrossAttention(decoderState, prevDecoderState, 
                ln2Output, encoderContext, encoderMask, lengths, position, isAllDoneCopy, isAllDone);
        this->memPool_->free(ln2Output->memory());
        auto resOutput2 = HUTensorUtil::Plus(contextAttention, resOutput, this->memPool_, device_);
        this->memPool_->free(resOutput->memory());
        this->memPool_->free(contextAttention->memory());

        auto ln3Output = this->lnLayer3_->Forward(resOutput2, 1e-12);
        auto ffnOutput = this->ffnLayer_->Forward(ln3Output);
        this->memPool_->free(ln3Output->memory());
        output = HUTensorUtil::Plus(ffnOutput, resOutput2, this->memPool_, device_);
        this->memPool_->free(ffnOutput->memory());
        this->memPool_->free(resOutput2->memory());
#endif

    }
    else
    {
#ifdef BASIC_KERNEL_FUSION
        // 1. self_attention(x) -> add_bias_input_layernorm
        auto selfAttention = this->attentionLayer_->DecoderLayerSelfAttention(decoderState, prevDecoderState, 
                embedding, selfAttMask, position, isAllDoneCopy, isAllDone);
        auto ln1Output = this->lnLayer1_->AddBiasInputForward(selfAttention, embedding, 
                this->attentionLayer_->GetSelfAttentionOutputBias(), 1e-12);
        this->memPool_->free(embedding->memory());
        this->memPool_->free(selfAttention->memory());

        // 2. cross_attention(x) -> add_bias_input_layernorm
        HUPtr<HUTensor> contextAttention = this->attentionLayer_->DecoderLayerCrossAttention(decoderState, prevDecoderState, 
                ln1Output, encoderContext, encoderMask, lengths, position, isAllDoneCopy, isAllDone);
        auto ln2Output = this->lnLayer2_->AddBiasInputForward(contextAttention, ln1Output, 
                this->attentionLayer_->GetCrossAttentionOutputBias(), 1e-12);
        this->memPool_->free(contextAttention->memory());
        this->memPool_->free(ln1Output->memory());

        // 3. ffn_layer(x) -> add_bias_input
        auto ffnOutput = this->ffnLayer_->Forward(ln2Output);
        output = this->lnLayer3_->AddBiasInputForward(ffnOutput, ln2Output, this->ffnLayer_->GetBias2(), 1e-12);
        this->memPool_->free(ffnOutput->memory());
        this->memPool_->free(ln2Output->memory());
#else
        auto selfAttention = this->attentionLayer_->DecoderLayerSelfAttention(decoderState, prevDecoderState, 
                embedding, selfAttMask, position, isAllDoneCopy, isAllDone);
        auto resOutput = HUTensorUtil::Plus(selfAttention, embedding, this->memPool_, device_);
        this->memPool_->free(embedding->memory());
        this->memPool_->free(selfAttention->memory());

        auto ln1Output = this->lnLayer1_->Forward(resOutput, 1e-12);
        this->memPool_->free(resOutput->memory());

        HUPtr<HUTensor> contextAttention = this->attentionLayer_->DecoderLayerCrossAttention(decoderState, prevDecoderState, 
                ln1Output, encoderContext, encoderMask, lengths, position, isAllDoneCopy, isAllDone);
        auto resOutput2 = HUTensorUtil::Plus(ln1Output, contextAttention, this->memPool_, device_);
        this->memPool_->free(ln1Output->memory());
        this->memPool_->free(contextAttention->memory());

        auto ln2Output = this->lnLayer2_->Forward(resOutput2, 1e-12);
        this->memPool_->free(resOutput2->memory());

        auto ffnOutput = this->ffnLayer_->Forward(ln2Output);
        auto resOutput3 = HUTensorUtil::Plus(ffnOutput, ln2Output, this->memPool_, device_);
        output = this->lnLayer3_->Forward(resOutput3, 1e-12);
        this->memPool_->free(ln2Output->memory());
        this->memPool_->free(ffnOutput->memory());
        this->memPool_->free(resOutput3->memory());
#endif
    }
#ifdef DECODER_DEBUG
    LOG(trace, "[TenTrans][HUDecoderLayer][Forward]Layer {} output {}", layerId_, output->debug());
#endif

    return output;
}


HUPtr<HUTensor> HUDecoderLayer::Forward_V2(HUPtr<HUTensor> embedding, HUPtr<HUTensor> selfAttMask, State& decoderState, State prevDecoderState, int position, HUPtr<HUTensor> encoderContext, HUPtr<HUTensor> encoderMask, HUPtr<HUTensor> lengths, const std::vector<uint8_t> &isAllDoneCopy, uint8_t* isAllDone)
{

	//Step 1. self-attention  [batch, beam, 1, hidden]
#ifdef DECODER_DEBUG
    LOG(trace, "[TenTrans][HUDecoderLayer][Forward]Layer {} embedding {}", layerId_, embedding->debug());
    LOG(trace, "[TenTrans][HUDecoderLayer][Forward]Layer {} selfAttMask {}", layerId_, selfAttMask->debug());
#endif
	auto selfAttention = this->attentionLayer_->DecoderLayerSelfAttention(decoderState, prevDecoderState, embedding, selfAttMask, position, isAllDoneCopy, isAllDone);
	//std::cout << "decoderState.output " <<  decoderState.output->debug() << std::endl;
	//std::cout << "self-attention done " << selfAttention->debug() <<std::endl;
#ifdef DECODER_DEBUG
	LOG(trace, "[TenTrans][HUDecoderLayer][Forward]Layer {} selfAttention {}", layerId_, selfAttention->debug());
    LOG(trace, "[TenTrans][HUDecoderLayer][Forward]Layer {} embedding {}", layerId_, embedding->debug());
#endif
	
	//Step 2. Add
#ifdef BIAS_LAYERNORM_FUSION
    auto ln1Output = this->lnLayer1_->AddBiasInputForward(selfAttention, embedding, 1e-12);
#else
    auto resOutput = HUTensorUtil::Plus(selfAttention, embedding, this->memPool_, device_);
    // std::cout << "[HUDecoderLayer] embedding free ..." << std::endl;
#ifdef DECODER_DEBUG
	LOG(trace, "[TenTrans][HUDecoderLayer][Forward]Layer {} resnet {}", layerId_, resOutput->debug());
#endif

	//Step 3. Norm
	auto ln1Output = this->lnLayer1_->Forward(resOutput, 1e-12);
    this->memPool_->free(resOutput->memory());
#endif
    this->memPool_->free(selfAttention->memory());
    this->memPool_->free(embedding->memory());
    // this->memPool_->free(resOutput->memory());
	//std::cout << "layer nomr " << ln1Output->debug() << std::endl;
#ifdef DECODER_DEBUG
	LOG(trace, "[TenTrans][HUDecoderLayer][Forward]Layer {} layerNorm {}", layerId_, ln1Output->debug());
#endif

	//Step 4. source-target attention
    /*
	int dimOut = ln1Output->shape()[-1];
    int dimBeam = ln1Output->shape()[-3];
    int dimBatch = ln1Output->shape()[-4];
	int dimHeads = this->options_->get<int>("transformer-heads");
    */
	// auto contextAttention = this->attentionLayer_->MultiHead(dimOut, dimHeads, ln1Output, encoderContext, encoderContext, encoderMask, true);

#ifdef DECODER_DEBUG
    LOG(trace, "[TenTrans][HUDecoderLayer][Forward]Layer {}, encoderContext {}", layerId_, encoderContext->debug());
    LOG(trace, "[TenTrans][HUDecoderLayer][Forward]Layer {}, encoderMask {}", layerId_, encoderMask->debug());
#endif

    // HUPtr<HUTensor> contextAttention = this->attentionLayer_->DecoderLayerCrossAttention(decoderState, prevDecoderState, ln1Output, encoderContext, encoderMask, position);
    HUPtr<HUTensor> contextAttention = this->attentionLayer_->DecoderLayerCrossAttention(decoderState, prevDecoderState, ln1Output, encoderContext, encoderMask, lengths, position, isAllDoneCopy, isAllDone);
#ifdef DECODER_DEBUG
	LOG(trace, "[TenTrans][HUDecoderLayer][Forward]Layer {} source-target attention {}", layerId_, contextAttention->debug());
#endif

	//Step 5. Add
#ifdef BIAS_LAYERNORM_FUSION
    auto ln2Output = this->lnLayer2_->AddBiasInputForward(contextAttention, ln1Output, 1e-12);
#else
	auto resOutput2 = HUTensorUtil::Plus(ln1Output, contextAttention, this->memPool_, device_);
	//std::cout << "+ " << resOutput2->debug() << std::endl;
#ifdef DECODER_DEBUG
	LOG(trace, "[TenTrans][HUDecoderLayer][Forward]Layer {} resNet {}", layerId_, resOutput2->debug());
#endif

	//Step 6. Norm
	auto ln2Output = this->lnLayer2_->Forward(resOutput2, 1e-12);
	this->memPool_->free(resOutput2->memory());
#endif
    this->memPool_->free(contextAttention->memory());
    this->memPool_->free(ln1Output->memory());

	//std::cout << "layer norm " << ln2Output->debug() << std::endl;
#ifdef DECODER_DEBUG
	LOG(trace, "[TenTrans][HUDecoderLayer][Forward]Layer {} layerNorm {}", layerId_, ln2Output->debug());
#endif

	//Step 7. Feed Forward
	auto ffnOutput = this->ffnLayer_->Forward(ln2Output);
	//std::cout << "end ffn " << ffnOutput->debug() << std::endl;
#ifdef DECODER_DEBUG
	LOG(trace, "[TenTrans][HUDecoderLayer][Forward]Layer {} FFNLayer {}", layerId_, ffnOutput->debug());
#endif

	//Step 8. Add
#ifdef BIAS_LAYERNORM_FUSION
    auto ln3Output = this->lnLayer3_->AddBiasInputForward(ffnOutput, ln2Output, 1e-12);
#else
	auto resOutput3 = HUTensorUtil::Plus(ffnOutput, ln2Output, this->memPool_, device_);
	//std::cout << "+ " << resOutput3->debug() << std::endl;
#ifdef DECODER_DEBUG
	LOG(trace, "[TenTrans][HUDecoderLayer][Forward]Layer {} resNet {}", layerId_, resOutput3->debug());
#endif

	//Step 9. Norm
	auto ln3Output = this->lnLayer3_->Forward(resOutput3, 1e-12);
	this->memPool_->free(resOutput3->memory());
#endif
    this->memPool_->free(ffnOutput->memory());
    this->memPool_->free(ln2Output->memory());
	//std::cout << "layer norm " << ln3Output->debug() << std::endl;
#ifdef DECODER_DEBUG
	LOG(trace, "[TenTrans][HUDecoderLayer][Forward]Layer {} layerNorm {}", layerId_, ln3Output->debug());
#endif

	return ln3Output;
}

}
