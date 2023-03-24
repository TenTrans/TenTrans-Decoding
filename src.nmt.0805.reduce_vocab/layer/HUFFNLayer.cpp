
#include "HUFFNLayer.h"
#include "HUTensorUtil.h"

namespace TenTrans
{

HUFFNLayer::HUFFNLayer(HUPtr<HUConfig> options, HUPtr<HUMemPool> memoryPool, HUPtr<HUDevice> device, cnpy::npz_t modelNpz, bool isEncoder, int layerId): 
    HUBaseLayer(options, memoryPool, device, modelNpz, isEncoder)
{
    this->layerId_ = layerId;
    if(this->isEncoder_) {
        this->prefix_ = "encoder.layers." + std::to_string(this->layerId_) + ".feed_forward.ffn_layer.";
    }
    else {
        this->prefix_ = "decoder.layers." + std::to_string(this->layerId_) + ".feed_forward.ffn_layer.";
    }
    
    NewBySuffix(W1, "0.weight");
    NewBySuffix(W2, "2.weight");
    NewBySuffix(b1, "0.bias");
    NewBySuffix(b2, "2.bias");
    
    string activationString = this->options_->get<string>("transformer-ffn-activation");
    switch(ResolveActivationOptions(activationString))
    {
        case Gelu:
            this->activationType_ = ActivationType::GELU;
            break;
        case Relu:
            this->activationType_ = ActivationType::RELU;
            break;
        case Swish:
            this->activationType_ = ActivationType::SWISH;
            break;
        default:
            ABORT("[TenTrans][ERROR][HUFFNLayer] Not found activation function {}", activationString);
            this->activationType_ = ActivationType::GELU;
            break;
    }
}

HUFFNLayer::~HUFFNLayer()
{
    this->memPool_->free(ffn_W1->memory());
    this->memPool_->free(ffn_W2->memory());
    
    this->memPool_->free(ffn_b1->memory());
    this->memPool_->free(ffn_b2->memory());
}

void HUFFNLayer::Init()
{
    InitBySuffix(W1, "0.weight");
    InitBySuffix(W2, "2.weight");
    InitBySuffix(b1, "0.bias");
    InitBySuffix(b2, "2.bias");
}

void HUFFNLayer::NewBySuffix(FFNEnum e, string param)
{
    HUPtr<HUShape> shape;
    shape = GetShapeByModel(this->prefix_ + param, this->modelNpz_);
    auto mem = this->memPool_->alloc<TT_DATA_TYPE>(shape->elements());
#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][FFNLayer] Loading {} parameters, {}", this->prefix_ + param, shape->toString());
#endif

    switch(e)
    {
        case W1:
            this->ffn_W1 = HUNew<HUTensor>(mem, *shape, this->device_);
            break;
        case W2:
            this->ffn_W2 = HUNew<HUTensor>(mem, *shape, this->device_);
            break;
        case b1:
            this->ffn_b1 = HUNew<HUTensor>(mem, *shape, this->device_);
            break;
        case b2:
            this->ffn_b2 = HUNew<HUTensor>(mem, *shape, this->device_);
            break;
        default:
            ABORT("[TenTrans] [Error] '{}' is not in our parameter lists", (this->prefix_ + param).c_str());
    }
}

void HUFFNLayer::InitBySuffix(FFNEnum e, string param)
{
    param = this->prefix_ + param;
    auto np = this->modelNpz_[param];
    size_t size = 1;
    for(size_t dim : np->shape) {
        size *= dim;
    }

    switch(e)
    {
        case W1:
            this->ffn_W1->set((TT_DATA_TYPE*)np->data(), (TT_DATA_TYPE*)np->data() + size);
            break;
        case W2:
            this->ffn_W2->set((TT_DATA_TYPE*)np->data(), (TT_DATA_TYPE*)np->data() + size);
            break;
        case b1:
            this->ffn_b1->set((TT_DATA_TYPE*)np->data(), (TT_DATA_TYPE*)np->data() + size);
            break;
        case b2:
            this->ffn_b2->set((TT_DATA_TYPE*)np->data(), (TT_DATA_TYPE*)np->data() + size);
            break;
        default:
            ABORT("[TenTrans] [Error] '{}' is not in our parameter lists", (param).c_str());
    }
}

HUPtr<HUTensor> HUFFNLayer::Forward(HUPtr<HUTensor> input)
{
#ifdef BASIC_KERNEL_FUSION
    /* step1: activationOutput = W1 * x */
    auto activationOutput = HUTensorUtil::Multiply(input, this->ffn_W1, this->memPool_, this->device_);
    /* step2: activationOutput = ActivationType(W1*x + b1) */
    HUTensorUtil::AddBiasActivation(activationOutput, this->ffn_b1, this->activationType_);
    /* step3: output = W2 * activationOutput */
    auto output = HUTensorUtil::Multiply(activationOutput, this->ffn_W2, this->memPool_, this->device_);
    this->memPool_->free(activationOutput->memory());
#else
    /* step1: ffn1 = W1*x + b1 */
    auto ffn1 = HUTensorUtil::Affine(input, this->ffn_W1, this->ffn_b1, this->memPool_, this->device_);
    /* step2: activationOutput = ActivationType(ffn1) */
    auto activationOutput = HUTensorUtil::Activation(ffn1, this->activationType_, this->memPool_, this->device_);
    this->memPool_->free(ffn1->memory());
    /* step3: ffn2 = W2*x + b2 */
    auto output = HUTensorUtil::Affine(activationOutput, this->ffn_W2, this->ffn_b2, this->memPool_, this->device_);
    this->memPool_->free(activationOutput->memory());
#endif  // BASIC_KERNEL_FUSION

#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][HUFFNLayer][Forward] output {}", output->debug());
#endif

    return output; 
}

}
