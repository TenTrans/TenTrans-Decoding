
#include "HUMultiHeadAttention.h"

namespace TenTrans{

	HUMultiHeadAttention::HUMultiHeadAttention(HUPtr<HUConfig> options, HUPtr<HUMemPool> memoryPool, HUPtr<HUDevice> device, cnpy::npz_t modelNpz, bool isEncoder, int layerId)
	: HUBaseLayer(options, memoryPool, device, modelNpz, isEncoder) {
		this->layerId_ = layerId;
        this->heads_ = this->options_->get<int>("transformer-heads");

		if(this->isEncoder_)
		{
            this->self_ = "encoder.layers." + std::to_string(this->layerId_) + ".src_src_att.";
			this->context_ = "";
		}
		else
		{
            // this->self_ = "decoder_l" + std::to_string(this->layerId_) + "_self_";
			// this->context_ = "decoder_l" + std::to_string(this->layerId_) + "_context_";
            this->self_ = "decoder.layers." + std::to_string(this->layerId_) + ".tgt_tgt_att.";
            this->context_ = "decoder.layers." + std::to_string(this->layerId_) + ".src_tgt_att.";
		}
	
        NewBySuffix(Wq, "q_layer.weight");
        NewBySuffix(bq, "q_layer.bias");
        NewBySuffix(Wk, "k_layer.weight");
        NewBySuffix(bk, "k_layer.bias");
        NewBySuffix(Wv, "v_layer.weight");
        NewBySuffix(bv, "v_layer.bias");
        NewBySuffix(Wo, "output_layer.weight");
        NewBySuffix(bo, "output_layer.bias");
	}

	//@TODO This part needs to be simplified
	void HUMultiHeadAttention::Init() {

		InitBySuffix(Wq, "q_layer.weight");
		InitBySuffix(bq, "q_layer.bias");
		InitBySuffix(Wk, "k_layer.weight");
		InitBySuffix(bk, "k_layer.bias");
		InitBySuffix(Wv, "v_layer.weight");
		InitBySuffix(bv, "v_layer.bias");
		InitBySuffix(Wo, "output_layer.weight");
		InitBySuffix(bo, "output_layer.bias");
    }

	void HUMultiHeadAttention::NewBySuffix(QKVEnum e, string param){

		HUPtr<HUShape> selfShape, contextShape;
		string selfParam = this->self_ + param;

        selfShape = GetShapeByModel(selfParam, this->modelNpz_);

		HUPtr<HUMemoryPiece> selfMem, contextMem;
	
		
		selfMem = this->memPool_->alloc<TT_DATA_TYPE>(selfShape->elements());
#ifdef DEBUG_MOD
		LOG(trace, "[TenTrans][MultiHeadAttention] Loading {} parameters, {}", selfParam, selfShape->toString());	
#endif

		if(!isEncoder_)
		{
			string contextParam = this->context_ + param;
			contextShape = GetShapeByModel(contextParam, this->modelNpz_);
			contextMem = this->memPool_->alloc<TT_DATA_TYPE>(contextShape->elements());
#ifdef DEBUG_MOD
			LOG(trace, "[TenTrans][MultiHeadAttention] Loading {} parameters, {}",contextParam, contextShape->toString());
#endif
		}

        switch(e){
            case Wq:
			{
                this->self_Wq = HUNew<HUTensor>(selfMem, *selfShape, this->device_);
				if(!isEncoder_)
					this->context_Wq = HUNew<HUTensor>(contextMem, *contextShape, this->device_);
                break;
			}
            case bq:
			{
                this->self_bq = HUNew<HUTensor>(selfMem, *selfShape, this->device_);
				if(!isEncoder_)
					this->context_bq = HUNew<HUTensor>(contextMem, *contextShape, this->device_);
                break;
			}
            case Wk:
			{
                this->self_Wk = HUNew<HUTensor>(selfMem, *selfShape, this->device_);
				if(!isEncoder_)
					this->context_Wk = HUNew<HUTensor>(contextMem, *contextShape, this->device_);
                break;
			}
            case bk:
			{
                this->self_bk = HUNew<HUTensor>(selfMem, *selfShape, this->device_);
				if(!isEncoder_)
					this->context_bk = HUNew<HUTensor>(contextMem, *contextShape, this->device_);
                break;
			}
			case Wv:
			{
				this->self_Wv = HUNew<HUTensor>(selfMem, *selfShape, this->device_);
				if(!isEncoder_)
					this->context_Wv = HUNew<HUTensor>(contextMem, *contextShape, this->device_);
				break;
			}
			case bv:
			{
				this->self_bv = HUNew<HUTensor>(selfMem, *selfShape, this->device_);
				if(!isEncoder_)
					this->context_bv = HUNew<HUTensor>(contextMem, *contextShape, this->device_);
				break;
			}
			case Wo:
			{
				this->self_Wo = HUNew<HUTensor>(selfMem, *selfShape, this->device_);
				if(!isEncoder_)
					this->context_Wo = HUNew<HUTensor>(contextMem, *contextShape, this->device_);
				break;
			}
			case bo:
			{
				this->self_bo = HUNew<HUTensor>(selfMem, *selfShape, this->device_);
				if(!isEncoder_)
					this->context_bo = HUNew<HUTensor>(contextMem, *contextShape, this->device_);
				break;
			}
            default:
                ABORT("[TenTrans] [Error] '{}' is not in our parameter lists", (this->self_ + param).c_str());
        }
	}

	void HUMultiHeadAttention::InitBySuffix(QKVEnum e, string param){
	
		string selfParam = this->self_ + param;
		auto np = this->modelNpz_[selfParam];
		string contextParam;
		if(!isEncoder_) {
			contextParam = this->context_ + param;
        }

		size_t size = 1;
		for(size_t dim : np->shape) {
			size *= dim;
        }
	
        // LOG(trace, "[TenTrans][MultiHeadAttention] Loading {} parameters, {}",contextParam, contextShape->toString());
		switch(e){
			case Wq:
			{
				this->self_Wq->set((TT_DATA_TYPE*)np->data(), (TT_DATA_TYPE*)np->data() + size);
#ifdef DEBUG_MOD
                LOG(trace, "[TenTrans][MultiHeadAttention] {} {}", selfParam, this->self_Wq->debug());
#endif

				if(!isEncoder_)
				{
					auto contextNp = this->modelNpz_[contextParam];
					this->context_Wq->set((TT_DATA_TYPE*)contextNp->data(), (TT_DATA_TYPE*)contextNp->data() + size);
				}

				break;
			}
			case bq: 
			{
				this->self_bq->set((TT_DATA_TYPE*)np->data(), (TT_DATA_TYPE*)np->data() + size);
#ifdef DEBUG_MOD
                LOG(trace, "[TenTrans][MultiHeadAttention] {} {}", selfParam, this->self_bq->debug());
#endif

				if(!isEncoder_)
				{
					auto contextNp = this->modelNpz_[contextParam];
					this->context_bq->set((TT_DATA_TYPE*)contextNp->data(), (TT_DATA_TYPE*)contextNp->data() + size);
				}
				break;
			}
			case Wk: 
			{
				this->self_Wk->set((TT_DATA_TYPE*)np->data(), (TT_DATA_TYPE*)np->data() + size);
#ifdef DEBUG_MOD
                LOG(trace, "[TenTrans][MultiHeadAttention] {} {}", selfParam, this->self_Wk->debug());
#endif

				if(!isEncoder_)
				{
					auto contextNp = this->modelNpz_[contextParam];
					this->context_Wk->set((TT_DATA_TYPE*)contextNp->data(), (TT_DATA_TYPE*)contextNp->data() + size);
				}
				break;
			}
			case bk: 
			{
				this->self_bk->set((TT_DATA_TYPE*)np->data(), (TT_DATA_TYPE*)np->data() + size);
#ifdef DEBUG_MOD
                LOG(trace, "[TenTrans][MultiHeadAttention] {} {}", selfParam, this->self_bk->debug());
#endif

				if(!isEncoder_)
				{
					auto contextNp = this->modelNpz_[contextParam];
					this->context_bk->set((TT_DATA_TYPE*)contextNp->data(), (TT_DATA_TYPE*)contextNp->data() + size);
				}
				break;
			}
			case Wv: 
			{
				this->self_Wv->set((TT_DATA_TYPE*)np->data(), (TT_DATA_TYPE*)np->data() + size);
#ifdef DEBUG_MOD
                LOG(trace, "[TenTrans][MultiHeadAttention] {} {}", selfParam, this->self_Wv->debug());
#endif

				if(!isEncoder_)
				{
					auto contextNp = this->modelNpz_[contextParam];
					this->context_Wv->set((TT_DATA_TYPE*)contextNp->data(), (TT_DATA_TYPE*)contextNp->data() + size);
				}
				break;
			}
			case bv: 
			{
				this->self_bv->set((TT_DATA_TYPE*)np->data(), (TT_DATA_TYPE*)np->data() + size);
#ifdef DEBUG_MOD
                LOG(trace, "[TenTrans][MultiHeadAttention] {} {}", selfParam, this->self_bv->debug());
#endif

				if(!isEncoder_)
				{
					auto contextNp = this->modelNpz_[contextParam];
					this->context_bv->set((TT_DATA_TYPE*)contextNp->data(), (TT_DATA_TYPE*)contextNp->data() + size);
				}
				break;
			}
			case Wo:
			{
				this->self_Wo->set((TT_DATA_TYPE*)np->data(), (TT_DATA_TYPE*)np->data() + size);
#ifdef DEBUG_MOD
                LOG(trace, "[TenTrans][MultiHeadAttention] {} {}", selfParam, this->self_Wo->debug());
#endif

				if(!isEncoder_)
				{
					auto contextNp = this->modelNpz_[contextParam];
					this->context_Wo->set((TT_DATA_TYPE*)contextNp->data(), (TT_DATA_TYPE*)contextNp->data() + size);
				}
				break;
			}
			case bo: 
			{
				this->self_bo->set((TT_DATA_TYPE*)np->data(), (TT_DATA_TYPE*)np->data() + size);
#ifdef DEBUG_MOD
                LOG(trace, "[TenTrans][MultiHeadAttention] {} {}", selfParam, this->self_bo->debug());
#endif

				if(!isEncoder_)
				{
					auto contextNp = this->modelNpz_[contextParam];
					this->context_bo->set((TT_DATA_TYPE*)contextNp->data(), (TT_DATA_TYPE*)contextNp->data() + size);
				}
				break;
			}
			default: 
				ABORT("[TenTrans] [Error] '{}' is not in our parameter lists", (this->self_ + param).c_str());
		}

#ifdef SELF_ATTENTION_FUSION
        this->self_Wqkv_fusion = HUTensorUtil::Concatenate({this->self_Wq, this->self_Wk, this->self_Wv}, -1, this->memPool_);
        this->self_bqkv_fusion = HUTensorUtil::Concatenate({this->self_bq, this->self_bk, this->self_bv}, -1, this->memPool_);
#endif
	}

HUMultiHeadAttention::~HUMultiHeadAttention()
{
    this->memPool_->free(this->self_Wk->memory());
    this->memPool_->free(this->self_bk->memory());
    this->memPool_->free(this->self_Wv->memory());
    this->memPool_->free(this->self_bv->memory());
    this->memPool_->free(this->self_Wq->memory());
    this->memPool_->free(this->self_bq->memory());
    this->memPool_->free(this->self_Wo->memory());
    this->memPool_->free(this->self_bo->memory());

    if (!this->isEncoder_)
    {
        this->memPool_->free(this->context_Wk->memory());
        this->memPool_->free(this->context_bk->memory());
        this->memPool_->free(this->context_Wv->memory());
        this->memPool_->free(this->context_bv->memory());
        this->memPool_->free(this->context_Wq->memory());
        this->memPool_->free(this->context_bq->memory());
        this->memPool_->free(this->context_Wo->memory());
        this->memPool_->free(this->context_bo->memory());
    }
   
    /*
    if (this->isExistTmpBuf_)
    {
        this->memPool_->free(this->beamLengths_->memory());
    } */

#ifdef SELF_ATTENTION_FUSION
    this->memPool_->free(this->self_Wqkv_fusion->memory());
    this->memPool_->free(this->self_bqkv_fusion->memory());
#endif

}

/*
void HUMultiHeadAttention::AddQKVBiasTranspose(HUPtr<HUTensor> &quries, HUPtr<HUTensor> &keys, HUPtr<HUTensor> &values)
{
    int dimModel = quries->shape()[-1];
    int dimSteps = quries->shape()[-2];
    int dimBatch = quries->shape()[-3];
    int dimPerHead = dimModel / this->heads_;
    
    HUTensorUtil::Add_QKV_Bias_Transpose(
            quries,
            keys,
            values,
            quries,
            this->self_bq,
            keys,
            this->self_bk,
            values,
            this->self_bv,
            dimBatch,
            dimSteps,
            this->heads_,
            dimPerHead);
}*/

HUPtr<HUTensor> HUMultiHeadAttention::MultiHead(HUPtr<HUTensor> q, const HUPtr<HUTensor> &keys, const HUPtr<HUTensor> &values, const HUPtr<HUTensor> &mask, bool isContext)
{
#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][HUMultiHeadAttention][MultiHead]Layer {}, q {}", layerId_, q->debug());
#endif
    /* Query Affine, [dimBatch, dimSteps, dimModel] */
	HUPtr<HUTensor> qh;
	if(!isContext) {
        // qh = HUTensorUtil::Multiply(q, this->self_Wq, this->memPool_, this->device_);
#ifdef DEBUG_MOD
        LOG(trace, "[TenTrans][HUMultiHeadAttention][MultiHead]Layer {}, Wq {}", layerId_, this->self_Wq->debug());
        LOG(trace, "[TenTrans][HUMultiHeadAttention][MultiHead]Layer {}, bq {}", layerId_, this->self_bq->debug());
#endif
        qh = HUTensorUtil::Affine(q, this->self_Wq, this->self_bq, this->memPool_, this->device_);
#ifdef DEBUG_MOD
        LOG(trace, "[TenTrans][HUMultiHeadAttention][MultiHead]Layer {}, qh {}", layerId_,  qh->debug());
#endif

        // std::cout << "is Self Attention..." << std::endl;
    }
	else {
        // qh = HUTensorUtil::Multiply(q, this->context_Wq, this->memPool_, this->device_);
		qh = HUTensorUtil::Affine(q, this->context_Wq, this->context_bq, this->memPool_, this->device_);
        // std::cout << "is Context Attention..." << std::endl;
    } 

    /*
    HUPtr<HUTensor> qh, kh, vh;
    HUTensorUtil::CopyFrom(qh, q, this->memPool_, this->device_);
    HUTensorUtil::CopyFrom(kh, q, this->memPool_, this->device_);
    HUTensorUtil::CopyFrom(vh, q, this->memPool_, this->device_);
    LOG(info, "[TenTrans][HUMultiHeadAttention][MultiHead]Layer {} q {}", layerId_, q->debug());
    LOG(info, "[TenTrans][HUMultiHeadAttention][MultiHead]Layer {} qh {}", layerId_, qh->debug());
    LOG(info, "[TenTrans][HUMultiHeadAttention][MultiHead]Layer {} kh {}", layerId_, kh->debug());
    LOG(info, "[TenTrans][HUMultiHeadAttention][MultiHead]Layer {} vh {}", layerId_, vh->debug());
    AddQKVBiasTranspose(qh, kh, vh);
    */

#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][HUMultiHeadAttention][MultiHead]Layer {}, q {}", layerId_, q->debug());
	LOG(trace, "[TenTrans][HUMultiHeadAttention][MultiHead]Layer {}, affine {}", layerId_, qh->debug());
#endif
    /* Query MultiHead Split, [dimBatch, dimSteps, dimModel] -> [dimBatch, dimHeads, dimSteps, dimDepth] */
    auto qhs = SplitHeads(qh);
	this->memPool_->free(qh->memory());
#ifdef DEBUG_MOD
	LOG(trace, "[TenTrans][HUMultiHeadAttention][MultiHead]Layer {}, qhs {}", layerId_, qhs->debug());
#endif

    // std::cout << "[KH] Affine start ..." << std::endl;
    /* Key Affine, [dimBatch, dimSteps, dimModel] */
	HUPtr<HUTensor> kh;
	if(!isContext) {
        // kh = HUTensorUtil::Multiply(keys, this->self_Wk, this->memPool_, this->device_);
		kh = HUTensorUtil::Affine(keys, this->self_Wk, this->self_bk, this->memPool_, this->device_);
    }
	else {
        // kh = HUTensorUtil::Multiply(keys, this->context_Wk, this->memPool_, this->device_);
		kh = HUTensorUtil::Affine(keys, this->context_Wk, this->context_bk, this->memPool_, this->device_);
    }
    // std::cout << "[KH] Affine ..." << std::endl;

#ifdef DEBUG_MOD
	LOG(trace, "[TenTrans][HUMultiHeadAttention][MultiHead]Layer {} affine {}", layerId_, kh->debug());
#endif

    // std::cout << "[KHS] Affine start ..." << std::endl;
    /* Key MultiHead Split, [dimBatch, dimSteps, dimModel] -> [dimBatch, dimHeads, dimSteps, dimModel] */
	auto khs = SplitHeads(kh);
	this->memPool_->free(kh->memory());
#ifdef DEBUG_MOD
	LOG(trace, "[TenTrans][HUMultiHeadAttention][MultiHead]Layer {}, khs {}", layerId_, khs->debug());
#endif
    // std::cout << "[KHS] Affine ..." << std::endl;

    /* Value Affine, [dimBatch, dimSteps, dimModel] */
	HUPtr<HUTensor> vh;
	if(!isContext) {
        // vh = HUTensorUtil::Multiply(values, this->self_Wv, this->memPool_, this->device_);
		vh = HUTensorUtil::Affine(values, this->self_Wv, this->self_bv, this->memPool_, this->device_);
    }
	else {
        // vh = HUTensorUtil::Multiply(values, this->context_Wv, this->memPool_, this->device_);
		vh = HUTensorUtil::Affine(values, this->context_Wv, this->context_bv, this->memPool_, this->device_);
    }
#ifdef DEBUG_MOD
	LOG(trace, "[TenTrans][HUMultiHeadAttention][MultiHead]Layer {} affine {}", layerId_, vh->debug());
#endif
    /* Value MultiHead Split, [dimBatch, dimSteps, dimModel] -> [dimBatch, dimHeads, dimSteps, dimDepth] */
	auto vhs = SplitHeads(vh);
	this->memPool_->free(vh->memory());
#ifdef DEBUG_MOD
	LOG(trace, "[TenTrans][HUMultiHeadAttention][MultiHead]Layer {}, vhs {}", layerId_, vhs->debug());
#endif

    /* kernel optimize, qh -> [batch, numhead*seqlen, per_head_size] */
    /*
    AddQKVBiasTranspose(qh, kh, vh);
    LOG(info, "[TenTrans][HUMultiHeadAttention][MultiHead]Layer {} qh {}", layerId_, qh->debug());
    LOG(info, "[TenTrans][HUMultiHeadAttention][MultiHead]Layer {} kh {}", layerId_, kh->debug());
    LOG(info, "[TenTrans][HUMultiHeadAttention][MultiHead]Layer {} vh {}", layerId_, vh->debug());
    */


    /*
    auto qhs = SplitHeads(qh);
    this->memPool_->free(qh->memory());
    auto khs = SplitHeads(kh);
    this->memPool_->free(kh->memory());
    auto vhs = SplitHeads(vh);
    this->memPool_->free(vh->memory());
    */


    /*
    int dk = k->shape()[-1];
    float scale = 1.0f / std::sqrt((float)dk);
    cublasGemmStridedBatchedEx(cublas_handle, 
            CUBLAS_OP_T, CUBLAS_OP_N, 
            seq_len, seq_len, size_per_head,
            &alpha, 
            k_buf_, AType_, size_per_head, seq_len * size_per_head, 
            q_buf_, BType_, size_per_head, seq_len * size_per_head,
            &beta,
            qk_buf_, CType_, seq_len, seq_len * seq_len,
            batch_size * head_num,
            computeType_, 
            static_cast<cublasGemmAlgo_t>(cublasBmmAlgo_[0]));


    auto z = HUTensorUtil::ProdBatched_v2(q, k, this->memPool_, device_, false, true, 0.f, scale);
    LOG(trace, "[TenTrans][HUMultiHeadAttention][Attention]Layer {} * {}", layerId_, z->debug());
    */


    /* MultiHead Attention, [dimBatch, numHeads, dimSteps, dimDepth] */
    auto multiHeadsAttention = Attention(qhs, khs, vhs, mask);
    this->memPool_->free(qhs->memory());
    this->memPool_->free(khs->memory());
    this->memPool_->free(vhs->memory());
#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][HUMultiHeadAttention][MultiHead]Layer {}, multiHeadsAttention {}", layerId_, multiHeadsAttention->debug());
#endif

    /* MultiHead Merge, [dimBatch, numHeads, dimWords, dimDepth] -> [dimBatch, dimWords, dimEmb] */
    auto joinHeads = JoinHeads(multiHeadsAttention);
    this->memPool_->free(multiHeadsAttention->memory());
#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][HUMultiHeadAttention][MultiHead]Layer {}, joinHeads {}", layerId_, joinHeads->debug());
#endif

    /* Outputs Affline, [dimBatch, dimWords, dimEmb] */
    HUPtr<HUTensor> output;
#ifdef BASIC_KERNEL_FUSION
    if(!isContext) {
        output = HUTensorUtil::Multiply(joinHeads, this->self_Wo, this->memPool_, this->device_);
    }
    else {
        output = HUTensorUtil::Multiply(joinHeads, this->context_Wo, this->memPool_, this->device_);
    }
#else
    if(!isContext) {
        output = HUTensorUtil::Affine(joinHeads, this->self_Wo, this->self_bo, this->memPool_, this->device_);
    }
    else {
        output = HUTensorUtil::Affine(joinHeads, this->context_Wo, this->context_bo, this->memPool_, this->device_);
    }
#endif
    this->memPool_->free(joinHeads->memory());

#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][HUMultiHeadAttention][MultiHead]Layer {} affine {}", layerId_, output->debug());
#endif

	return output;
}

//q: [-4: beam depth * batch size, -3: num heads, -2: max tgt length, -1: split vector dim]
HUPtr<HUTensor> HUMultiHeadAttention::Attention(HUPtr<HUTensor> q, HUPtr<HUTensor> k, HUPtr<HUTensor> v, HUPtr<HUTensor> mask)
{
	int dk = k->shape()[-1];
	
	// softmax over batched dot product of query and keys (applied over all
    // time steps and batch entries), also add mask for illegal connections
	
	float scale = 1.0f / std::sqrt((float)dk);     // scaling to avoid extreme values due to matrix multiplication
	
    // [dimBatch, numHeads, dimTgtWords, dimSrcWords]
	auto z = HUTensorUtil::ProdBatched(q, k, this->memPool_, device_, false, true, 0.f, scale);
#ifdef DEBUG_MOD
	LOG(trace, "[TenTrans][HUMultiHeadAttention][Attention]Layer {}, ProdBatched(q, k){}", layerId_, z->debug());
#endif

	auto maskedZ = HUTensorUtil::Plus(z, mask, memPool_, device_);
	this->memPool_->free(z->memory());
#ifdef DEBUG_MOD
	LOG(trace, "[TenTrans][HUMultiHeadAttention][Attention]Layer {}, Plus {}", layerId_, maskedZ->debug());
#endif

	// take softmax along src sequence axis (-1)
	// [-4: beam depth * batch size, -3: num heads, -2: max tgt length, -1: max src length]
	auto weights = HUTensorUtil::Softmax(maskedZ, memPool_, device_);
	this->memPool_->free(maskedZ->memory());
#ifdef DEBUG_MOD
	LOG(trace, "[TenTrans][HUMultiHeadAttention][Attention]Layer {}, softmax {}", layerId_, weights->debug());
#endif
	
	auto output = HUTensorUtil::ProdBatched(weights, v, this->memPool_, device_);
	this->memPool_->free(weights->memory());
#ifdef DEBUG_MOD
	LOG(trace, "[TenTrans][HUMultiHeadAttention][Attention]Layer {} * {}", layerId_, output->debug());
#endif

	return output;
}

/* input: [dimBatch, dimSteps, dimModel] */
HUPtr<HUTensor> HUMultiHeadAttention::SplitHeads(HUPtr<HUTensor> input)
{
    int dimModel = input->shape()[-1];
    int dimSteps = input->shape()[-2];
    int dimBatch = input->shape()[-3];
	int dimDepth = dimModel / this->heads_;

    auto output = HUTensorUtil::Reshape(input, {dimBatch, dimSteps, this->heads_, dimDepth});
#ifdef DEBUG_MOD
	LOG(trace, "[TenTrans][HUMultiHeadAttention][SplitHeads]Layer {} reshape {}", layerId_, output->debug());
#endif

    // [dimBatch, dimSteps, dimHeads, dimDepth] -> [dimBatch, dimHeads, dimSteps, dimDepth]
	return HUTensorUtil::TransposeTimeBatch(output, this->memPool_, this->device_);
}

HUPtr<HUTensor> HUMultiHeadAttention::JoinHeads(HUPtr<HUTensor> input)
{
	int dimDepth = input->shape()[-1];
    int dimSteps = input->shape()[-2];
    int dimHeads = input->shape()[-3];
    int dimBatch = input->shape()[-4];
    int dimModel = dimHeads * dimDepth;

    // [dimBatch, dimHeads, dimSteps, dimDepth] -> [dimBatch, dimSteps, dimHeads, dimDepth]
    auto output = HUTensorUtil::TransposeTimeBatch(input, this->memPool_, this->device_);
#ifdef DEBUG_MOD
	LOG(trace, "[TenTrans][HUMultiHeadAttention][JoinHeads]Layer {} transpose {}", layerId_, output->debug());
#endif

    // [dimBatch, dimSteps, dimHeads, dimDepth] -> [dimBatch, dimSteps, dimModel]
    return HUTensorUtil::Reshape(output, {dimBatch, dimSteps, dimModel});
}

HUPtr<HUTensor> HUMultiHeadAttention::Forward(HUPtr<HUTensor> batchEmbedding, HUPtr<HUTensor> batchMask)
{
    // LOG(trace, "[TenTrans][HUMultiHeadAttention][batchMask] {}", batchMask->debug());
	auto headsOutput = this->MultiHead(batchEmbedding, batchEmbedding, batchEmbedding, batchMask);

#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][HUMultiHeadAttention][HeadsOutput] {}",  headsOutput->debug());
#endif

	return headsOutput;
}

// only used for encoder self-attention
HUPtr<HUTensor> HUMultiHeadAttention::ForwardFusedEncoderSelfAttention(HUPtr<HUTensor> batchEmbedding, HUPtr<HUTensor> batchMask, EncoderSelfAttentionBuffer &params)
{
    // LOG(trace, "[TenTrans][HUMultiHeadAttention][batchMask] {}", batchMask->debug());

    // batchMask [-4: dimBatch, -3: numHeads broadcast=1, -2: dimWords broadcast=1, -1: dimWords] -> [0., 0., 0., -inf]
    int dimBatch = batchMask->shape()[-4];
    int dimSeqLen = batchMask->shape()[-1];
   
    /*
    auto newBatchMask = HUTensorUtil::Repeat(HUTensorUtil::Reshape(batchMask, {dimBatch, 1, dimSeqLen}),
            dimSeqLen, -2, this->memPool_); */
    // LOG(trace, "[TenTrans][HUMultiHeadAttention][newBatchMask] {}", newBatchMask->debug());
    auto joinHeads = HUTensorUtil::EncoderUnFusedSelfAttention(batchEmbedding, batchMask,
            this->self_Wq, this->self_bq,
            this->self_Wk, this->self_bk,
            this->self_Wv, this->self_bv,
            this->heads_, params, this->memPool_, this->device_);
    // this->memPool_->free(newBatchMask->memory());

#ifdef BASIC_KERNEL_FUSION
    auto attentionOutput = HUTensorUtil::Multiply(joinHeads, this->self_Wo, this->memPool_, this->device_);
#else
    auto attentionOutput = HUTensorUtil::Affine(joinHeads, this->self_Wo, this->self_bo, this->memPool_, this->device_);
#endif
    this->memPool_->free(joinHeads->memory());

#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][HUMultiHeadAttention][HeadsOutput] {}", attentionOutput->debug());
#endif

    return attentionOutput;
}

HUPtr<HUTensor> HUMultiHeadAttention::DecoderLayerSelfAttention(State& decoderLayerState, const State& prevdecoderLayerState, HUPtr<HUTensor> input, HUPtr<HUTensor> selfMask, int startPos, int realDimBatch, uint8_t* isAllDone)
{
#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][HUMutiHeadAttention][DecoderLayerSelfAttention] input {}", input->debug());
#endif
    int dimBatch = input->shape()[-3];
    int dimModel = input->shape()[-1];

#ifdef SELF_ATTENTION_FUSION
    HUPtr<HUTensor> curCacheKeys, curCacheValues, curCacheKeysTmp, curCacheValuesTmp;
#else
    HUPtr<HUTensor> qh = HUTensorUtil::Affine(input, this->self_Wq, this->self_bq, this->memPool_, this->device_);
    HUPtr<HUTensor> kh = HUTensorUtil::Affine(input, this->self_Wk, this->self_bk, this->memPool_, this->device_);
    HUPtr<HUTensor> vh = HUTensorUtil::Affine(input, this->self_Wv, this->self_bv, this->memPool_, this->device_);
    HUPtr<HUTensor> curCacheKeys, curCacheValues;
#endif

    if (startPos == 0)
    {
#ifdef SELF_ATTENTION_FUSION   // MAX_DECODER_STEPS=256
        curCacheKeys = HUTensorUtil::Zeros({dimBatch, MAX_DECODER_STEPS, dimModel}, this->memPool_, this->device_);
        curCacheValues = HUTensorUtil::Zeros({dimBatch, MAX_DECODER_STEPS, dimModel}, this->memPool_, this->device_);

        curCacheKeysTmp = HUTensorUtil::Zeros({dimBatch, MAX_DECODER_STEPS, dimModel}, this->memPool_, this->device_);
        curCacheValuesTmp = HUTensorUtil::Zeros({dimBatch, MAX_DECODER_STEPS, dimModel}, this->memPool_, this->device_);
#else
        HUTensorUtil::CopyFrom(curCacheKeys, kh, this->memPool_, this->device_);
        HUTensorUtil::CopyFrom(curCacheValues, vh, this->memPool_, this->device_);
#endif
    }
    else
    {
#ifdef SELF_ATTENTION_FUSION
        curCacheKeys = prevdecoderLayerState.cacheKeys;
        curCacheValues = prevdecoderLayerState.cacheValues;

        curCacheKeysTmp = prevdecoderLayerState.cacheKeysTmp;
        curCacheValuesTmp = prevdecoderLayerState.cacheValuesTmp;
#else
        auto prevCacheKeys = prevdecoderLayerState.cacheKeys;
        auto prevCacheValues = prevdecoderLayerState.cacheValues;

        curCacheKeys = HUTensorUtil::Concatenate({prevCacheKeys, kh}, -2, this->memPool_);
        curCacheValues = HUTensorUtil::Concatenate({prevCacheValues, vh}, -2, this->memPool_);

        this->memPool_->free(prevCacheKeys->memory());
        this->memPool_->free(prevCacheValues->memory());
#endif
    }

#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][HUMutiHeadAttention][DecoderLayerSelfAttention] curCacheKeys {}", curCacheKeys->debug());
    LOG(trace, "[TenTrans][HUMutiHeadAttention][DecoderLayerSelfAttention] curCacheValues {}", curCacheValues->debug());
#endif

#ifdef SELF_ATTENTION_FUSION
    auto qkvh = HUTensorUtil::Multiply(input, this->self_Wqkv_fusion, this->memPool_, this->device_);
    auto attentionOut = HUTensorUtil::FusedQKVSelfAttention(qkvh, this->self_bqkv_fusion, curCacheKeys, curCacheValues, 
            realDimBatch, isAllDone, this->heads_, startPos, this->memPool_, this->device_);
    this->memPool_->free(qkvh->memory());

    decoderLayerState.cacheKeys = curCacheKeys;
    decoderLayerState.cacheValues = curCacheValues;
    decoderLayerState.cacheKeysTmp = curCacheKeysTmp;
    decoderLayerState.cacheValuesTmp = curCacheValuesTmp;
#else
    auto qhs = SplitHeads(qh);
    auto khs = SplitHeads(curCacheKeys);
    auto vhs = SplitHeads(curCacheValues);
    
    this->memPool_->free(qh->memory());
    this->memPool_->free(kh->memory());
    this->memPool_->free(vh->memory());

    /* [dimBatch, dimHead, 1, dimPerHead] */
    auto attentionOut = Attention(qhs, khs, vhs, selfMask);
    this->memPool_->free(qhs->memory());
    this->memPool_->free(khs->memory());
    this->memPool_->free(vhs->memory());

    /* MultiHead Merge, [dimBatch, dimHead, 1, dimPerHead] -> [dimBatch, 1, dimModel] */
    attentionOut = HUTensorUtil::Reshape(attentionOut, {dimBatch, 1, dimModel});

    decoderLayerState.cacheKeys = curCacheKeys;
    decoderLayerState.cacheValues = curCacheValues;
#endif

#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][HUMutiHeadAttention][DecoderLayerSelfAttention] curCacheKeys {}", curCacheKeys->debug());
    LOG(trace, "[TenTrans][HUMutiHeadAttention][DecoderLayerSelfAttention] curCacheValues {}", curCacheValues->debug());
    LOG(trace, "[TenTrans][HUMutiHeadAttention][DecoderLayerSelfAttention] attentionOut {}", attentionOut->debug());
#endif

    /* Output, [dimBatch, 1, dimModel] */
    HUPtr<HUTensor> output;
#ifdef BASIC_KERNEL_FUSION
    output = HUTensorUtil::Multiply(attentionOut, this->self_Wo, this->memPool_, this->device_);
#else
    output = HUTensorUtil::Affine(attentionOut, this->self_Wo, this->self_bo, this->memPool_, this->device_);
#endif
    this->memPool_->free(attentionOut->memory());

    return output;
}

HUPtr<HUTensor> HUMultiHeadAttention::DecoderLayerSelfAttention_V2(State& decoderLayerState, const State& prevdecoderLayerState, HUPtr<HUTensor> input, HUPtr<HUTensor> selfMask, int startPos)
{

   HUPtr<HUTensor> qh, kh, vh;
#ifdef SELF_ATTENTION_FUSION
    std::vector<HUPtr<HUTensor>> nodes(3);
    auto qkvh = HUTensorUtil::Affine(input, this->self_Wqkv_fusion, this->self_bqkv_fusion, this->memPool_, this->device_);
    // LOG(trace, "[TenTrans][HUMutiHeadAttention][DecoderLayerSelfAttention] qkvh {}", qkvh->debug());
    HUTensorUtil::Split(qkvh, 3, nodes, this->memPool_);
    qh = nodes[0];
    // LOG(trace, "[TenTrans][HUMutiHeadAttention][DecoderLayerSelfAttention] qh {}", qh->debug());
    kh = nodes[1];
    vh = nodes[2];
    this->memPool_->free(qkvh->memory());
#else
    // HUPtr<HUTensor> qh;
    qh = HUTensorUtil::Affine(input, this->self_Wq, this->self_bq, this->memPool_, this->device_);
    // LOG(trace, "[TenTrans][HUMutiHeadAttention][DecoderLayerSelfAttention] qh {}", qh->debug());
#ifdef DECODER_DEBUG
    LOG(trace, "[TenTrans][HUMutiHeadAttention][DecoderLayerSelfAttention] qh {}", qh->debug());
#endif

    // HUPtr<HUTensor> kh;
    kh = HUTensorUtil::Affine(input, this->self_Wk, this->self_bk, this->memPool_, this->device_);
#ifdef DECODER_DEBUG
    LOG(trace, "[TenTrans][HUMutiHeadAttention][DecoderLayerSelfAttention] kh {}", kh->debug());
#endif

    // HUPtr<HUTensor> vh;
    vh = HUTensorUtil::Affine(input, this->self_Wv, this->self_bv, this->memPool_, this->device_);
#ifdef DECODER_DEBUG
    LOG(trace, "[TenTrans][HUMutiHeadAttention][DecoderLayerSelfAttention] vh {}", vh->debug());
#endif
#endif

    HUPtr<HUTensor> curCacheKeys;
    HUPtr<HUTensor> curCacheValues;
    if (startPos == 0) 
    {
        HUTensorUtil::CopyFrom(curCacheKeys, kh, this->memPool_, this->device_);
        HUTensorUtil::CopyFrom(curCacheValues, vh, this->memPool_, this->device_);
    }
    else
    {
        auto prevCacheKeys = prevdecoderLayerState.cacheKeys;
        auto prevCacheValues = prevdecoderLayerState.cacheValues;
        curCacheKeys = HUTensorUtil::Concatenate({prevCacheKeys, kh}, -2, this->memPool_);
        curCacheValues = HUTensorUtil::Concatenate({prevCacheValues, vh}, -2, this->memPool_);

        this->memPool_->free(prevCacheKeys->memory());
        this->memPool_->free(prevCacheValues->memory());
    }
#ifdef DECODER_DEBUG
    LOG(trace, "[TenTrans][HUMutiHeadAttention][DecoderLayerSelfAttention] curCacheKeys {}", curCacheKeys->debug());
    LOG(trace, "[TenTrans][HUMutiHeadAttention][DecoderLayerSelfAttention] curCacheValues {}", curCacheValues->debug());
#endif
    // std::cout << "is ok1 ..." << std::endl;
    decoderLayerState.cacheKeys = curCacheKeys;
    // std::cout << "is ok2 ..." << std::endl;
    decoderLayerState.cacheValues = curCacheValues;
    // std::cout << "is ok3 ..." << std::endl;

    // LOG(trace, "[TenTrans][HUMutiHeadAttention][DecoderLayerSelfAttention] curCacheKeys {}", curCacheKeys->debug());
    // LOG(trace, "[TenTrans][HUMutiHeadAttention][DecoderLayerSelfAttention] curCacheValues {}", curCacheValues->debug());

    auto qhs = SplitHeads(qh);
    this->memPool_->free(qh->memory());

    auto khs = SplitHeads(curCacheKeys);
    this->memPool_->free(kh->memory());

    auto vhs = SplitHeads(curCacheValues);
    this->memPool_->free(vh->memory());

    /* MultiHead Attention, [dimBatch, numHeads, dimSteps, dimDepth] */
    auto multiHeadsAttention = Attention(qhs, khs, vhs, selfMask);
    this->memPool_->free(qhs->memory());
    this->memPool_->free(khs->memory());
    this->memPool_->free(vhs->memory());

    /* MultiHead Merge, [dimBatch, numHeads, dimWords, dimDepth] -> [dimBatch, dimWords, dimEmb] */
    auto joinHeads = JoinHeads(multiHeadsAttention);
    this->memPool_->free(multiHeadsAttention->memory());
    // LOG(trace, "[TenTrans][HUMutiHeadAttention][DecoderLayerSelfAttention] attentionOut {}", joinHeads->debug());

    /* Outputs Affline, [dimBatch, dimWords, dimEmb] */
    HUPtr<HUTensor> output;
#ifdef BASIC_KERNEL_FUSION
    output = HUTensorUtil::Multiply(joinHeads, this->self_Wo, this->memPool_, this->device_);
#else
    output = HUTensorUtil::Affine(joinHeads, this->self_Wo, this->self_bo, this->memPool_, this->device_);
#endif
    this->memPool_->free(joinHeads->memory());

    return output;
}


HUPtr<HUTensor> HUMultiHeadAttention::DecoderLayerCrossAttention(State& decoderLayerState, const State& prevdecoderLayerState, HUPtr<HUTensor> q, const HUPtr<HUTensor> &memory, const HUPtr<HUTensor> &mask, HUPtr<HUTensor> &lengths, int startPos, int realDimBatch, uint8_t* isAllDone)
{
#ifdef CROSS_ATTENTION_FUSION
    HUPtr<HUTensor> qh = HUTensorUtil::Multiply(q, this->context_Wq, this->memPool_, this->device_);
#else
    HUPtr<HUTensor> qh = HUTensorUtil::Affine(q, this->context_Wq, this->context_bq, this->memPool_, this->device_);
#endif

    HUPtr<HUTensor> kh, vh;
    if(startPos == 0)         // the first step 
    {
#ifdef CROSS_ATTENTION_FUSION
        kh = HUTensorUtil::Multiply(memory, this->context_Wk, this->memPool_, this->device_);
        decoderLayerState.memoryKeys = kh;

        vh = HUTensorUtil::Multiply(memory, this->context_Wv, this->memPool_, this->device_);
        decoderLayerState.memoryValues = vh;
#else
        kh = HUTensorUtil::Affine(memory, this->context_Wk, this->context_bk, this->memPool_, this->device_);
        decoderLayerState.memoryKeys = kh;

        vh = HUTensorUtil::Affine(memory, this->context_Wv, this->context_bv, this->memPool_, this->device_);
        decoderLayerState.memoryValues = vh;
#endif
    }
    else                      // repeat beam_size times
    {
        auto tmpMemoryKeys = prevdecoderLayerState.memoryKeys;
        auto tmpMemoryValues = prevdecoderLayerState.memoryValues;
        if (q->shape()[-3] != tmpMemoryKeys->shape()[-3])
        {
            int dimBeam = q->shape()[-3] / tmpMemoryKeys->shape()[-3];
            int dimBatch = tmpMemoryKeys->shape()[-3];
            int dimSrcWords = tmpMemoryKeys->shape()[-2];
            int dimEmb = tmpMemoryKeys->shape()[-1];
            auto repeatKh = HUTensorUtil::Repeat(HUTensorUtil::Reshape(tmpMemoryKeys, {dimBatch, 1, dimSrcWords, dimEmb}),
                    dimBeam, -3, this->memPool_);
            repeatKh = HUTensorUtil::Reshape(repeatKh, {dimBatch*dimBeam, dimSrcWords, dimEmb});

            auto repeatVh = HUTensorUtil::Repeat(HUTensorUtil::Reshape(tmpMemoryValues, {dimBatch, 1, dimSrcWords, dimEmb}),
                    dimBeam, -3, this->memPool_);
            repeatVh = HUTensorUtil::Reshape(repeatVh, {dimBatch*dimBeam, dimSrcWords, dimEmb});

            kh = repeatKh;
            decoderLayerState.memoryKeys = repeatKh;
            vh = repeatVh;
            decoderLayerState.memoryValues = repeatVh;

            this->memPool_->free(tmpMemoryKeys->memory());
            this->memPool_->free(tmpMemoryValues->memory());
        }
        else
        {
            kh = tmpMemoryKeys;
            decoderLayerState.memoryKeys = kh;

            vh = tmpMemoryValues;
            decoderLayerState.memoryValues = vh;
        }
    }

    HUPtr<HUTensor> attentionOut;
#ifdef CROSS_ATTENTION_FUSION
    if (startPos == 0) {
        attentionOut = HUTensorUtil::CrossAttention(qh, this->context_bq, kh, this->context_bk, vh, this->context_bv, 
                lengths, realDimBatch, isAllDone, this->heads_, startPos, this->memPool_, this->device_);
    }
    else
    {
        /*
        if (!this->isExistTmpBuf_) 
        {
            int dimBeam = qh->shape()[-3] / lengths->size();
            this->beamLengths_ = HUTensorUtil::Repeat(HUTensorUtil::Reshape(lengths, {lengths->size(), 1}), dimBeam, -1, this->memPool_);
            this->beamLengths_ = HUTensorUtil::Reshape(this->beamLengths_, {qh->shape()[-3]});
            this->isExistTmpBuf_ = true;
         } */
        
        int dimBeam = qh->shape()[-3] / lengths->size();
        auto beamLengths = HUTensorUtil::Repeat(HUTensorUtil::Reshape(lengths, {lengths->size(), 1}), dimBeam, -1, this->memPool_);
        beamLengths = HUTensorUtil::Reshape(beamLengths, {qh->shape()[-3]});

        attentionOut = HUTensorUtil::CrossAttention(qh, this->context_bq, kh, this->context_bk, vh, this->context_bv, 
                beamLengths, realDimBatch, isAllDone, this->heads_, startPos, this->memPool_, this->device_);
        this->memPool_->free(beamLengths->memory());
    }

#else
    auto qhs = SplitHeads(qh);
    auto khs = SplitHeads(kh);
    auto vhs = SplitHeads(vh);

    auto multiHeadsAttention = Attention(qhs, khs, vhs, mask);
    attentionOut = JoinHeads(multiHeadsAttention);

    this->memPool_->free(qhs->memory());
    this->memPool_->free(khs->memory());
    this->memPool_->free(vhs->memory());
    this->memPool_->free(multiHeadsAttention->memory());
#endif
    this->memPool_->free(qh->memory());

#ifdef BASIC_KERNEL_FUSION
    auto output = HUTensorUtil::Multiply(attentionOut, this->context_Wo, this->memPool_, this->device_);
#else
    auto output = HUTensorUtil::Affine(attentionOut, this->context_Wo, this->context_bo, this->memPool_, this->device_);
#endif
    this->memPool_->free(attentionOut->memory());

    return output;
}

/*
HUPtr<HUTensor> HUMultiHeadAttention::DecoderLayerCrossAttention(State& decoderLayerState, const State& prevdecoderLayerState, HUPtr<HUTensor> q, const HUPtr<HUTensor> &memory, const HUPtr<HUTensor> &mask, int startPos)
{
    HUPtr<HUTensor> qh = HUTensorUtil::Multiply(q, this->context_Wq, this->memPool_, this->device_);
    HUPtr<HUTensor> kh, vh;
    if(startPos == 0)    // the first step 
    {
        kh = HUTensorUtil::Multiply(memory, this->context_Wk, this->memPool_, this->device_);
        decoderLayerState.memoryKeys = kh;

        vh = HUTensorUtil::Multiply(memory, this->context_Wv, this->memPool_, this->device_);
        decoderLayerState.memoryValues = vh;
    }
    else                 // repeat beam_size times
    {
        auto tmpMemoryKeys = prevdecoderLayerState.memoryKeys;
        auto tmpMemoryValues = prevdecoderLayerState.memoryValues;
        if (q->shape()[-3] != tmpMemoryKeys->shape()[-3]) // repeat beams
        {
            int dimBeam = q->shape()[-3] / tmpMemoryKeys->shape()[-3];
            int dimBatch = tmpMemoryKeys->shape()[-3];
            int dimSrcWords = tmpMemoryKeys->shape()[-2];
            int dimEmb = tmpMemoryKeys->shape()[-1];
            auto repeatKh = HUTensorUtil::Repeat(HUTensorUtil::Reshape(tmpMemoryKeys,{dimBatch, 1, dimSrcWords, dimEmb}), 
                    dimBeam, -3, this->memPool_);
            repeatKh = HUTensorUtil::Reshape(repeatKh, {dimBatch*dimBeam, dimSrcWords, dimEmb});

            auto repeatVh = HUTensorUtil::Repeat(HUTensorUtil::Reshape(tmpMemoryValues, {dimBatch, 1, dimSrcWords, dimEmb}), 
                    dimBeam, -3, this->memPool_);
            repeatVh = HUTensorUtil::Reshape(repeatVh, {dimBatch*dimBeam, dimSrcWords, dimEmb});

            kh = repeatKh;
            decoderLayerState.memoryKeys = repeatKh;
            vh = repeatVh;
            decoderLayerState.memoryValues = repeatVh; 

            this->memPool_->free(tmpMemoryKeys->memory());
            this->memPool_->free(tmpMemoryValues->memory());
        }
        else
        {
            kh = tmpMemoryKeys;
            decoderLayerState.memoryKeys = kh;

            vh = tmpMemoryValues;
            decoderLayerState.memoryValues = vh;
        }
    }
    
    std::vector<float> lengths(kh->shape()[-3], kh->shape()[-2]);
    auto lengthMem = this->memPool_->alloc<float>(lengths.size());
    HUShape lengthShape = HUShape({lengths.size()});
    HUPtr<HUTensor> lengthsT = HUNew<HUTensor>(lengthMem, lengthShape, this->device_);
    lengthsT->set(lengths);

    
    // LOG(trace, "[TenTrans][HUMutiHeadAttention][DecoderLayerCrossAttention] q_bias {}", this->context_bq->debug());
    // LOG(trace, "[TenTrans][HUMutiHeadAttention][DecoderLayerCrossAttention] qh1 {}", qh->debug());
    // LOG(trace, "[TenTrans][HUMutiHeadAttention][DecoderLayerCrossAttention] kh1 {}", kh->debug());
    // LOG(trace, "[TenTrans][HUMutiHeadAttention][DecoderLayerCrossAttention] vh1 {}", vh->debug());
    

    auto attentionOut = HUTensorUtil::CrossAttention(qh, this->context_bq, kh, this->context_bk, vh, this->context_bv, 
            lengthsT, this->heads_, startPos, this->memPool_, this->device_);

    // LOG(trace, "[TenTrans][HUMutiHeadAttention][DecoderLayerCrossAttention] qh2 {}", qh->debug());
    // LOG(trace, "[TenTrans][HUMutiHeadAttention][DecoderLayerCrossAttention] kh2 {}", kh->debug());
    // LOG(trace, "[TenTrans][HUMutiHeadAttention][DecoderLayerCrossAttention] vh2 {}", vh->debug());
    // LOG(trace, "[TenTrans][HUMutiHeadAttention][DecoderLayerCrossAttention] attentionOut {}", attentionOut->debug());
    
    
    this->memPool_->free(qh->memory());
    this->memPool_->free(lengthsT->memory());

#ifndef BIAS_LAYERNORM_FUSION
    auto output = HUTensorUtil::Affine(attentionOut, this->context_Wo, this->context_bo, this->memPool_, this->device_);
#else
    auto output = HUTensorUtil::Multiply(attentionOut, this->context_Wo, this->memPool_, this->device_);
#endif
    this->memPool_->free(attentionOut->memory());

    return output;
}
*/

/*
HUPtr<HUTensor> HUMultiHeadAttention::DecoderLayerCrossAttention(State& decoderLayerState, const State& prevdecoderLayerState, HUPtr<HUTensor> q, const HUPtr<HUTensor> &memory, const HUPtr<HUTensor> &mask, int startPos)
{
    HUPtr<HUTensor> qh = HUTensorUtil::Affine(q, this->context_Wq, this->context_bq, this->memPool_, this->device_);
    LOG(trace, "[TenTrans][HUMutiHeadAttention][DecoderLayerCrossAttention] qh {}", qh->debug());
    auto qhs = SplitHeads(qh);
    this->memPool_->free(qh->memory());

    HUPtr<HUTensor> kh;
    if(startPos == 0) {
        // std::cout << "is ok1 ..." << std::endl;
        kh = HUTensorUtil::Affine(memory, this->context_Wk, this->context_bk, this->memPool_, this->device_);
        // std::cout << "is ok2 ..." << std::endl;
        decoderLayerState.memoryKeys = kh;
        // HUTensorUtil::CopyFrom(decoderLayerState.memoryKeys, kh, this->memPool_, this->device_);
    }
    else {
        // std::cout << "is ok3 ..." << std::endl;
        // kh = decoderLayerState.memoryKeys;
        kh = prevdecoderLayerState.memoryKeys;
        decoderLayerState.memoryKeys = kh;

    }
    LOG(trace, "[TenTrans][HUMutiHeadAttention][DecoderLayerCrossAttention] kh {}", kh->debug());
    // std::cout << "is ok4 ..." << std::endl;
    // auto khs = SplitHeads(kh);
    // this->memPool_->free(kh->memory());

    HUPtr<HUTensor> vh;
    if(startPos == 0) {
        vh = HUTensorUtil::Affine(memory, this->context_Wv, this->context_bv, this->memPool_, this->device_);
        decoderLayerState.memoryValues = vh;
        // HUTensorUtil::CopyFrom(decoderLayerState.memoryValues, vh, this->memPool_, this->device_);
    }
    else {
        // vh = decoderLayerState.memoryValues;
        vh = prevdecoderLayerState.memoryValues;
        decoderLayerState.memoryValues = vh;
        // std::cout << "is ok41 ..." << std::endl;
    }
    LOG(trace, "[TenTrans][HUMutiHeadAttention][DecoderLayerCrossAttention] vh {}", vh->debug());
    // std::cout << "is ok5 ..." << std::endl;
#ifdef DECODER_DEBUG
    LOG(trace, "[TenTrans][HUMutiHeadAttention][DecoderLayerCrossAttention] vh {}", vh->debug());
#endif

    HUPtr<HUTensor> khs, vhs;
    if (q->shape()[-3] == memory->shape()[-3]) 
    {
        // std::cout << "is ok11 ..." << std::endl;
        khs = SplitHeads(kh);
        // this->memPool_->free(kh->memory());
        // std::cout << "is ok22 ..." << std::endl;
        vhs = SplitHeads(vh);
        // this->memPool_->free(vh->memory());
        // std::cout << "is ok33 ..." << std::endl;
    }
    else
    {
        // std::cout << "is ok51 ..." << std::endl;
        int dimBeam = q->shape()[-3] / memory->shape()[-3];
        int dimBatch = memory->shape()[-3];
        int dimSrcWords = memory->shape()[-2];
        int dimEmb = memory->shape()[-1];

        // std::cout << "is ok6 ..." << std::endl;
        auto repeatKh = HUTensorUtil::Repeat(HUTensorUtil::Reshape(kh, {dimBatch, 1, dimSrcWords, dimEmb}), dimBeam, -3, this->memPool_);
        repeatKh = HUTensorUtil::Reshape(repeatKh, {dimBatch*dimBeam, dimSrcWords, dimEmb});
        khs = SplitHeads(repeatKh);
        this->memPool_->free(repeatKh->memory());
        // std::cout << "is ok7 ..." << std::endl;

        auto repeatVh = HUTensorUtil::Repeat(HUTensorUtil::Reshape(vh, {dimBatch, 1, dimSrcWords, dimEmb}), dimBeam, -3, this->memPool_);
        repeatVh = HUTensorUtil::Reshape(repeatVh, {dimBatch*dimBeam, dimSrcWords, dimEmb});
        vhs = SplitHeads(repeatVh);
        this->memPool_->free(repeatVh->memory());
        // std::cout << "is ok8 ..." << std::endl;

        // auto repeatMask = HUTensorUtil::Reshape(HUTensorUtil::Repeat(mask, dimBeam, -3, this->memPool_), {dimBatch*dimBeam, 1, 1, mask->shape()[-1]});
        // contextAttention = this->attentionLayer_->MultiHead(ln1Output, reshapeEncoderContext, reshapeEncoderContext, encoderMask, true);
        // this->memPool_->free(reshapeEncoderContext->memory());
        // this->memPool_->free(reshapeEncoderMask->memory());
    }
    
    // std::cout << "is ok9 ..." << std::endl;
    auto multiHeadsAttention = Attention(qhs, khs, vhs, mask);
    this->memPool_->free(qhs->memory());
    this->memPool_->free(khs->memory());
    this->memPool_->free(vhs->memory());

    // std::cout << "is ok10 ..." << std::endl;
    auto joinHeads = JoinHeads(multiHeadsAttention);
    this->memPool_->free(multiHeadsAttention->memory());
    // LOG(trace, "[TenTrans][HUMutiHeadAttention][DecoderLayerCrossAttention] qh {}", qh->debug());
    // LOG(trace, "[TenTrans][HUMutiHeadAttention][DecoderLayerCrossAttention] kh {}", kh->debug());
    // LOG(trace, "[TenTrans][HUMutiHeadAttention][DecoderLayerCrossAttention] vh {}", vh->debug());
    LOG(trace, "[TenTrans][HUMutiHeadAttention][DecoderLayerCrossAttention] joinHeads {}", joinHeads->debug());

    // std::cout << "is ok11 ..." << std::endl;
#ifndef BIAS_LAYERNORM_FUSION
    auto output = HUTensorUtil::Affine(joinHeads, this->context_Wo, this->context_bo, this->memPool_, this->device_);
#else
    auto output = HUTensorUtil::Multiply(joinHeads, this->context_Wo, this->memPool_, this->device_);
#endif
    this->memPool_->free(joinHeads->memory());
    // std::cout << "is ok12 ..." << std::endl;

    return output;
} */

}
