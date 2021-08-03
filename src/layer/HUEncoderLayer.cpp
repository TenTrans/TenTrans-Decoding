
#include "HUEncoderLayer.h"

namespace TenTrans{

HUEncoderLayer::HUEncoderLayer(HUPtr<HUConfig> options, HUPtr<HUMemPool> memoryPool, HUPtr<HUDevice> device, cnpy::npz_t modelNpz, int layerId, bool isEncoder) 
	: HUBaseLayer(options, memoryPool, device, modelNpz, isEncoder)
{
#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][HUEncoderLayer] Loading AttentionLayer ...");
#endif
	this->attentionLayer_ = HUNew<HUMultiHeadAttention>(options, memoryPool, device, modelNpz, isEncoder, layerId);

#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][HUEncoderLayer] Loading FFNLayer ...");
#endif
	this->ffnLayer_ = HUNew<HUFFNLayer>(options, memoryPool, device, modelNpz, isEncoder, layerId);

#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][HUEncoderLayer] Loading LayerNorm1 ...");
#endif
    this->lnLayer1_ = HUNew<HULayerNorm>(options, memoryPool, device, modelNpz, isEncoder, layerId, false);

#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][HUEncoderLayer] Loading LayerNorm2 ...\n");
#endif
	this->lnLayer2_ = HUNew<HULayerNorm>(options, memoryPool, device, modelNpz, isEncoder, layerId, true);

	this->layerId_ = layerId;
    // this->isNormalizeBefore_= true;
    this->isNormalizeBefore_ = this->options_->get<bool>("normalize-before");
}

void HUEncoderLayer::Init()
{
	this->attentionLayer_->Init();
    this->ffnLayer_->Init();
    this->lnLayer1_->Init();
	this->lnLayer2_->Init();
}

HUEncoderLayer::~HUEncoderLayer()
{

}

HUPtr<HUTensor> HUEncoderLayer::Forward(HUPtr<HUTensor> x, HUPtr<HUTensor> batchMask, EncoderSelfAttentionBuffer &params)
{
    /*
     *  pre-norm:
     *  1) y = x + self_attention(layer_norm(x));
     *  2) outputs = y + ffn_layer(layer_norm(y))
     *  optimizations:
     *  a) x->layer_norm->self_attention(without add bias) => y
     *  b) add_bias_input_layernorm->ffn_layer->add_bias_input => outputs
     *
     *  post-norm:
     *  1) y = layer_norm(x + self_attention(x))
     *  2) outputs = layer_norm(y + ffn_layer(y))
     *  optimizations:
     *  a) x->self_attention(without add bias)->add_bias_input_layernorm => y
     *  b) ffn_layer(without add bias)->add_bias_input_layernorm => outputs
     *
     */
#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][HUEncoderLayer][Forward]Layer {}, input {}", layerId_, x->debug());
#endif

    HUPtr<HUTensor> output;
    if (this->isNormalizeBefore_)  // pre-norm
    {
#ifdef BASIC_KERNEL_FUSION
        // 1. self_attention(layer_norm(x))
        auto ln1Output = this->lnLayer1_->Forward(x, 1e-12);
#ifdef DEBUG_MOD
        LOG(trace, "[TenTrans][HUEncoderLayer][ForwardLayer] {}, [WKX layernorm1] {}", layerId_, ln1Output->debug());
#endif

#ifdef ENCODER_SELF_ATTENTION_FUSION
        auto attentionOutput = this->attentionLayer_->ForwardFusedEncoderSelfAttention(ln1Output, batchMask, params);
#else
        auto attentionOutput = this->attentionLayer_->Forward(ln1Output, batchMask);
#endif
        this->memPool_->free(ln1Output->memory());

        // LOG(trace, "[TenTrans][HUEncoderLayer][ForwardLayer] {}, attentionOutput {}", layerId_, attentionOutput->debug());
        // LOG(trace, "[TenTrans][HUEncoderLayer][ForwardLayer] {}, x {}", layerId_, x->debug());
        // 2. add_bias_input_layernorm
        // resnet: attentionOutput + bias + x -> attentionOutput
        auto ln2Output = this->lnLayer2_->AddBiasInputForward(attentionOutput, x, 
                this->attentionLayer_->GetSelfAttentionOutputBias(), 1e-12);
        this->memPool_->free(x->memory());
#ifdef DEBUG_MOD
        LOG(trace, "[TenTrans][HUEncoderLayer][Forward]Layer {}, [WKX layernorm2] {}", layerId_, ln2Output->debug());
#endif

        // 3. ffn_layer
        output = this->ffnLayer_->Forward(ln2Output);
        this->memPool_->free(ln2Output->memory());
#ifdef DEBUG_MOD
        LOG(trace, "[TenTrans][HUEncoderLayer][Forward]Layer {}, [WKX ffnlayer] {}", layerId_, output->debug());
#endif

        // 4. add_bias_input
        HUTensorUtil::AddBiasInput(output, this->ffnLayer_->GetBias2(), attentionOutput);
        this->memPool_->free(attentionOutput->memory());
#else   // BASIC_KERNEL_FUSION

        auto ln1Output = this->lnLayer1_->Forward(x, 1e-12);
#ifdef DEBUG_MOD
        LOG(trace, "[TenTrans][HUEncoderLayer][Forward]Layer {}, [WKX layernorm1] {}", layerId_, ln1Output->debug());
#endif

#ifdef ENCODER_SELF_ATTENTION_FUSION
        auto attentionOutput = this->attentionLayer_->ForwardFusedEncoderSelfAttention(ln1Output, batchMask, params);
#else
        auto attentionOutput = this->attentionLayer_->Forward(ln1Output, batchMask);
#endif

#ifdef DEBUG_MOD
        LOG(trace, "[TenTrans][HUEncoderLayer][Forward]Layer {}, [WKX selfattention] {}", layerId_, attentionOutput->debug());
#endif

        auto resOutput = HUTensorUtil::Plus(attentionOutput, x, this->memPool_, device_);
        this->memPool_->free(x->memory());
        this->memPool_->free(ln1Output->memory());
        this->memPool_->free(attentionOutput->memory());

        auto ln2Output = this->lnLayer2_->Forward(resOutput, 1e-12);
#ifdef DEBUG_MOD
        LOG(trace, "[TenTrans][HUEncoderLayer][Forward]Layer {}, [WKX layernorm2] {}", layerId_, ln2Output->debug());
#endif

        auto ffnOutput = this->ffnLayer_->Forward(ln2Output);
#ifdef DEBUG_MOD
        LOG(trace, "[TenTrans][HUEncoderLayer][Forward]Layer {}, [WKX ffnlayer] {}", layerId_, ffnOutput->debug());
#endif
        output = HUTensorUtil::Plus(ffnOutput, resOutput, this->memPool_, device_);
        this->memPool_->free(ln2Output->memory());
        this->memPool_->free(ffnOutput->memory());
        this->memPool_->free(resOutput->memory());
#endif
    }
    else // post-norm
    {
#ifdef BASIC_KERNEL_FUSION

        // 1. self_attention(x) -> add_bias_input_layernorm
#ifdef ENCODER_SELF_ATTENTION_FUSION
        auto attentionOutput = this->attentionLayer_->ForwardFusedEncoderSelfAttention(x, batchMask, params);
#else
        auto attentionOutput = this->attentionLayer_->Forward(x, batchMask);
#endif
        // LOG(trace, "[TenTrans][HUEncoderLayer][Forward]Layer {} x {}", layerId_, attentionOutput->debug());
        // attentionOutput + bias + x -> attentionOutput
        auto ln1Output = this->lnLayer1_->AddBiasInputForward(attentionOutput, x, 
                this->attentionLayer_->GetSelfAttentionOutputBias(), 1e-12);

        // LOG(trace, "[TenTrans][HUEncoderLayer][Forward]Layer {} x {}", layerId_, attentionOutput->debug());
        this->memPool_->free(x->memory());
        this->memPool_->free(attentionOutput->memory());

        // 2. ffn_layer -> add_bias_layernorm
        auto ffnOutput = this->ffnLayer_->Forward(ln1Output);
        output = this->lnLayer2_->AddBiasInputForward(ffnOutput, ln1Output, this->ffnLayer_->GetBias2(), 1e-12);
        this->memPool_->free(ln1Output->memory());
        this->memPool_->free(ffnOutput->memory());
#else

#ifdef ENCODER_SELF_ATTENTION_FUSION
        auto attentionOutput = this->attentionLayer_->ForwardFusedEncoderSelfAttention(x, batchMask, params);
#else
        auto attentionOutput = this->attentionLayer_->Forward(x, batchMask);
#endif
        auto resOutput = HUTensorUtil::Plus(attentionOutput, x, this->memPool_, device_);
        this->memPool_->free(x->memory());
        this->memPool_->free(attentionOutput->memory());

        auto ln1Output = this->lnLayer1_->Forward(resOutput, 1e-12);
        this->memPool_->free(resOutput->memory());

        auto ffnOutput = this->ffnLayer_->Forward(ln1Output);
        resOutput = HUTensorUtil::Plus(ln1Output, ffnOutput, this->memPool_, device_);
        this->memPool_->free(ln1Output->memory());
        this->memPool_->free(ffnOutput->memory());

        output = this->lnLayer2_->Forward(resOutput, 1e-12);
        this->memPool_->free(resOutput->memory());
#endif
    }
#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][HUEncoderLayer][Forward]Layer {}, output {}", layerId_, output->debug());
#endif
    return output;
}

HUPtr<HUTensor> HUEncoderLayer::Forward_V2(HUPtr<HUTensor> batchEmbedding, HUPtr<HUTensor> batchMask)
{

#ifdef TIME_CALCULATION
    cudaEvent_t start, stop;
    float elapsedTime = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
#endif

	/* Step 1. MultiHead Attention, [dimBatch, dimSteps, dimModel] */
	auto headsOutput = this->attentionLayer_->Forward(batchEmbedding, batchMask);
#ifdef DEBUG_MOD
	LOG(trace, "[TenTrans][HUEncoderLayer][Forward]Layer {} selfAttention {}", layerId_, headsOutput->debug());
#endif

#ifdef TIME_CALCULATION
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    LOG(info, "AttentionLayer Time Cost: {}", elapsedTime);
#endif
    
	/* Step 2. Residual Connection, [dimBatch, dimSteps, dimModel] */
#ifdef BIAS_LAYERNORM_FUSION
    auto ln1Output = this->lnLayer1_->AddBiasInputForward(headsOutput, batchEmbedding, 1e-12);
#else
	auto resOutput = HUTensorUtil::Plus(headsOutput, batchEmbedding, this->memPool_, device_);
#ifdef DEBUG_MOD
	LOG(trace, "[TenTrans][HUEncoderLayer][Forward]Layer {} resNet {}", layerId_, resOutput->debug());
#endif

#ifdef TIME_CALCULATION
    cudaEventRecord(start, 0);
#endif
	/* Step 3. Layer Normalization, [dimBatch, dimSteps, dimModel] */
	auto ln1Output = this->lnLayer1_->Forward(resOutput, 1e-12);
    this->memPool_->free(resOutput->memory());
#endif
    this->memPool_->free(headsOutput->memory());
    this->memPool_->free(batchEmbedding->memory());

#ifdef DEBUG_MOD
	LOG(trace, "[TenTrans][HUEncoderLayer][Forward]Layer {} layerNorm {}", layerId_, ln1Output->debug());
#endif

#ifdef TIME_CALCULATION
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    LOG(info, "LayerNorm Layer Time Cost: {}", elapsedTime);
#endif

	// this->memPool_->free(resOutput->memory());

    /*
    cudaEventRecord(start, 0);
    */

	/* Step 4. Feed Forward, [dimBatch, dimSteps, dimModel] */
	auto ffnOutput = this->ffnLayer_->Forward(ln1Output);
#ifdef DEBUG_MOD
	LOG(trace, "[TenTrans][HUEncoderLayer][Forward]Layer {} FFNLayer {}", layerId_, ffnOutput->debug());
#endif

    /*
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    LOG(info, "FFNLayer Time Cost: {}", elapsedTime);
    */

#ifdef TIME_CALCULATION
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif

	/* Step 5. Residual Connection, [dimBatch, dimSteps, dimModel] */
#ifdef BIAS_LAYERNORM_FUSION
    auto ln2Output = this->lnLayer2_->AddBiasInputForward(ffnOutput, ln1Output);
#else
	resOutput = HUTensorUtil::Plus(ln1Output, ffnOutput, this->memPool_, device_);
#ifdef DEBUG_MOD
	LOG(trace, "[TenTrans][HUEncoderLayer][Forward]Layer {} resNet {}", layerId_, resOutput->debug());
#endif

	/* Step 6. Layer Normalization, [dimBatch, dimSteps, dimModel] */
	auto ln2Output = this->lnLayer2_->Forward(resOutput);
	this->memPool_->free(resOutput->memory());
#endif
    this->memPool_->free(ffnOutput->memory());
    this->memPool_->free(ln1Output->memory());

#ifdef DEBUG_MOD
	LOG(trace, "[TenTrans][HUEncoderLayer][Forward]Layer {} layerNorm {}", layerId_, ln2Output->debug());
#endif

	return ln2Output;
}

}
