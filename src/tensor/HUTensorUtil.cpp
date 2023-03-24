#include "HUTensorUtil.h"

namespace TenTrans{

HUShape HUTensorUtil::GetCopyRowsShape(HUPtr<HUTensor> a, const std::vector<size_t>& indices) {
    int num = indices.size();
    HUShape shape = a->shape();
    ABORT_IF(shape.size() != 2, "[TenTrans] rows operator can only be used with 2-dimensional tensors");
    shape.set(0, num);
    return shape;
}

HUShape HUTensorUtil::GetTransposeShape(HUPtr<HUTensor> a, const std::vector<int>& axes)
{
	HUShape shape = a->shape();
	ABORT_IF(shape.size() != axes.size(), "[TenTrans] Shape and transpose axes have different number of dimensions");
	for(size_t i = 0; i < shape.size(); ++i)
		shape.set(i, a->shape()[axes[i]]);
	return shape;
}

void HUTensorUtil::Save(const std::string name, HUPtr<HUTensor> a, const std::string tensorName)
{
	LOG(info, "Saving tensor {} to {}", tensorName, name);

	std::vector<TT_DATA_TYPE> v;
	a->get(v);
	auto& pShape = a->shape();
	unsigned dim = pShape.size();
	unsigned* shape = new unsigned[dim];
	for(int i = 0; i < dim; ++i)
		shape[i] = pShape[i];
	cnpy::npz_save(name, tensorName, v.data(), shape, dim, "w");
	delete[] shape;
}

HUPtr<HUTensor> HUTensorUtil::AtLeastNd(HUPtr<HUTensor> a, size_t dims)
{
	if(a->shape().size() >= dims)
		return a;
	HUShape nShape;
	nShape.resize(dims);
	for(int i = 1; i <= (int)a->shape().size(); ++i)
		nShape.set(-i, a->shape()[-i]);
	return Reshape(a, nShape);
}

HUPtr<HUTensor> HUTensorUtil::TransposeTimeBatch(HUPtr<HUTensor> input, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device)
{
	auto transposeShape = HUTensorUtil::GetTransposeShape(input, {0,2,1,3});
	auto transposeMem = mem->alloc<float>(transposeShape.elements());
	auto output = HUNew<HUTensor>(transposeMem, transposeShape, device);
	TransposeND(output, input, {0,2,1,3});
	return output;
}

HUPtr<HUTensor> HUTensorUtil::Transpose(HUPtr<HUTensor> input, const std::vector<int>& axes, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device) 
{
    auto transposeShape = HUTensorUtil::GetTransposeShape(input, axes);
    auto transposeMem = mem->alloc<float>(transposeShape.elements());
    auto output = HUNew<HUTensor>(transposeMem, transposeShape, device);
    TransposeND(output, input, axes);
    return output;
}

HUPtr<HUTensor> HUTensorUtil::Neg(HUPtr<HUTensor> input, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device)
{
	auto newMem = mem->alloc<TT_DATA_TYPE>(input->size());
	auto output = HUNew<HUTensor>(newMem, input->shape(), device);
	NegOP(output, input);
	return output;
}

HUPtr<HUTensor> HUTensorUtil::Plus(HUPtr<HUTensor> a, HUPtr<HUTensor> b, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device)
{
	HUShape newShape = HUShape::broadcast({a, b});
	auto newMem = mem->alloc<TT_DATA_TYPE>(newShape.elements());
	auto output = HUNew<HUTensor>(newMem, newShape, device);
	PlusBroadcast(output, a, b);
	return output;
}

HUPtr<HUTensor> HUTensorUtil::ScaleAndShift(HUPtr<HUTensor> &a, float scale, float shift)
{
	ScaleAndShiftOP(a, scale, shift);
	return a;
}

HUPtr<HUTensor> HUTensorUtil::TransposedLogMask(HUPtr<HUTensor> mask, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device)
{
    // [-3: dimBatch, -2: dimWords broadcast=1, -1: dimWords]
	auto ms = mask->shape();
	auto negMask = Neg(mask, mem, device);

    // [-3: dimBatch, -2: dimWords broadcast=1, -1: dimWords]
	ScaleAndShift(negMask, 1, 1);
	/// ScaleAndShift(negMask, -99999999.f, 0);
    ScaleAndShift(negMask, -10000.f, 0);  // consider half && float

    // [-4: dimBatch, -3: numHeads broadcast=1, -2: dimWords broadcast=1, -1: dimWords]
	auto output = Reshape(negMask, {ms[-3], 1, ms[-2], ms[-1]});
	return output;
}

HUShape HUTensorUtil::GetAffineShape(HUPtr<HUTensor> a, HUPtr<HUTensor> b, bool transA, bool transB)
{
	auto shapeA = a->shape();
	auto shapeB = b->shape();
	HUShape outShape;
    if (!transA) 
    {
        outShape = shapeA;
        if (!transB)
        {
            outShape.set(outShape.size() - 1, shapeB[shapeB.size() - 1]);
            ABORT_IF(shapeA[shapeA.size() - 1] != shapeB[shapeB.size() - 2], "[TenTrans] matrix product requires dimensions to match");
        }
        else
        {
            outShape.set(outShape.size() - 1, shapeB[shapeB.size() - 2]);
            ABORT_IF(shapeA[shapeA.size() - 1] != shapeB[shapeB.size() - 1], "[TenTrans] matrix product requires dimensions to match");
        }
    }
    else
    {
         ABORT_IF(true, "[TenTrans] Not support the operation.");
    }
	return outShape;


}

HUShape HUTensorUtil::GetDotBatchedShape(HUPtr<HUTensor> a, HUPtr<HUTensor> b, bool transA, bool transB, float scalar)
{
	auto shapeA = a->shape();
	if(transA) {
      shapeA.set(-2, a->shape()[-1]);
      shapeA.set(-1, a->shape()[-2]);
    }

	auto shapeB = b->shape();
    if(transB) {
      shapeB.set(-2, b->shape()[-1]);
      shapeB.set(-1, b->shape()[-2]);
    }

	HUShape outShape = shapeA;
    outShape.set(-1, shapeB[-1]);
    ABORT_IF(shapeA[-1] != shapeB[-2], "[TenTrans] matrix product requires dimensions to match");
    return outShape;
}

void HUTensorUtil::Affine(HUPtr<HUTensor> &y, HUPtr<HUTensor> x, HUPtr<HUTensor> w, HUPtr<HUTensor> b, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device, bool transA, bool transB, float beta, float alpha)
{
    /*
    cudaEvent_t start, stop;
    // clock_t c_start, c_end;
    float elapsedTime = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    // c_start = clock();
    */

    auto yShape = HUTensorUtil::GetAffineShape(x, w, transA, transB);
    auto yMem = mem->alloc<TT_DATA_TYPE>(yShape.elements());
    y = HUNew<HUTensor>(yMem, yShape, device);

    /*
    cudaThreadSynchronize();
    c_end = clock();
    LOG(info, "[item] memcpy Time C Cost: {}", ((c_end-c_start)/CLOCKS_PER_SEC)*1000);
    */
    // std::cout << "c Cost1: " << (c_end-c_start)/1000.0 << std::endl;
    //

    /*
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    LOG(info, "[item] Alloc Time Cost: {}", elapsedTime);

    cudaEventRecord(start, 0);
    // c_start = clock();
    */

    // ProdWithBias(y, x, w, b, false, false, 0, 1);
	Prod(y, x, w, false, false, 0, 1);
	AddBias(y, b);


    /*
    cudaThreadSynchronize();
    c_end = clock();
    LOG(info, "[item] memcpy Time C Cost: {}", ((double)(c_end-c_start)/CLOCKS_PER_SEC)*1000);
    */

    /*
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    LOG(info, "[item] ProdWithBias Time Cost: {}", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    */
}

HUPtr<HUTensor> HUTensorUtil::Affine(HUPtr<HUTensor> x, HUPtr<HUTensor> w, HUPtr<HUTensor> b, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device, bool transA, bool transB, float beta, float alpha)
{
    /*
	cudaEvent_t start, stop;
	// clock_t c_start, c_end;
	float elapsedTime = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	// c_start = clock();
    */
	
	auto yShape = HUTensorUtil::GetAffineShape(x, w, transA, transB);
	auto yMem = mem->alloc<TT_DATA_TYPE>(yShape.elements());
	auto y = HUNew<HUTensor>(yMem, yShape, device);

	/*
	cudaThreadSynchronize();
	c_end = clock();
	LOG(info, "[item] memcpy Time C Cost: {}", ((c_end-c_start)/CLOCKS_PER_SEC)*1000);
	*/
	// std::cout << "c Cost1: " << (c_end-c_start)/1000.0 << std::endl;


    /*
    cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	LOG(info, "[item] Alloc Time Cost: {}", elapsedTime);

	cudaEventRecord(start, 0);
	// c_start = clock();
    */

	ProdWithBias(y, x, w, b, transA, transB, beta, alpha);

	
	/*
	cudaThreadSynchronize();
	c_end = clock();
	LOG(info, "[item] memcpy Time C Cost: {}", ((double)(c_end-c_start)/CLOCKS_PER_SEC)*1000);
	*/

    /*
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	LOG(info, "[item] ProdWithBias Time Cost: {}", elapsedTime);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
    */
	
	//std::cout << y->debug() << std::endl;
	return y;
}

HUPtr<HUTensor> HUTensorUtil::Multiply(HUPtr<HUTensor> x, HUPtr<HUTensor> w, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device,  bool transA, bool transB, float beta, float alpha)
{
    auto yShape = HUTensorUtil::GetAffineShape(x, w, transA, transB);
    auto yMem = mem->alloc<TT_DATA_TYPE>(yShape.elements());
    auto y = HUNew<HUTensor>(yMem, yShape, device);

    Prod(y, x, w, transA, transB, beta, alpha);
    return y;
}

void HUTensorUtil::Multiply_v2(HUPtr<HUTensor> &y, HUPtr<HUTensor> x, HUPtr<HUTensor> w, bool transA, bool transB, float beta, float alpha)
{
    Prod(y, x, w, transA, transB, beta, alpha);
}

/*
HUPtr<HUTensor> HUTensorUtil::ProdBatched_v2(HUPtr<HUTensor> A, HUPtr<HUTensor> B, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device, bool transA, bool transB, float beta, float scalar)
{
    auto cShape = HUTensorUtil::GetDotBatchedShape(A, B, transA, transB, scalar);
    auto cMem = mem->alloc<float>(cShape.elements());
    auto C = HUNew<HUTensor>(cMem, cShape, device);

    ProdBatchedOP_v2(C, A, B, mem, transA, transB, beta, scalar);
    return C;
} */

HUPtr<HUTensor> HUTensorUtil::ProdBatched(HUPtr<HUTensor> A, HUPtr<HUTensor> B, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device, bool transA, bool transB, float beta, float scalar)
{
	auto cShape = HUTensorUtil::GetDotBatchedShape(A, B, transA, transB, scalar);
	auto cMem = mem->alloc<TT_DATA_TYPE>(cShape.elements());
	auto C = HUNew<HUTensor>(cMem, cShape, device);

	ProdBatchedOP(C, A, B, mem, transA, transB, beta, scalar);
	return C;
}

HUPtr<HUTensor> HUTensorUtil::Softmax(HUPtr<HUTensor> in, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device, HUPtr<HUTensor> mask)
{
	auto outMem = mem->alloc<TT_DATA_TYPE>(in->size());
	auto out = HUNew<HUTensor>(outMem, in->shape(), device);
	SoftmaxOP(out, in, mask);
	return out;
}

HUPtr<HUTensor> HUTensorUtil::LogSoftmax(HUPtr<HUTensor> in, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device)
{
	auto outMem = mem->alloc<float>(in->size());
    auto out = HUNew<HUTensor>(outMem, in->shape(), device);
    LogSoftmaxOP(out, in);
    return out;
}

HUPtr<HUTensor> HUTensorUtil::AddBiasLogSoftmax(HUPtr<HUTensor> in, const HUPtr<HUTensor> bias, const int realDimBatch, uint8_t* isAllDone, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device)
{
    auto outMem = mem->alloc<TT_DATA_TYPE>(in->size());
    auto out = HUNew<HUTensor>(outMem, in->shape(), device);
    // AddBiasLogSoftmaxOP(out, in, bias);
    AddBiasLogSoftmaxOP_V2(out, in, bias, realDimBatch, isAllDone);
    return out;
}

HUPtr<HUTensor> HUTensorUtil::AddBiasInputLayerNormalization(HUPtr<HUTensor> in, HUPtr<HUTensor> x, const HUPtr<HUTensor> bias, HUPtr<HUTensor> gamma, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device, HUPtr<HUTensor> beta, float eps)
{
    auto outMem = mem->alloc<TT_DATA_TYPE>(in->size());
    auto out = HUNew<HUTensor>(outMem, in->shape(), device);
    AddBiasInputLayerNormalOP(out, in, x, bias, gamma, beta, eps);
    return out;
}

HUPtr<HUTensor> HUTensorUtil::LayerNormalization(HUPtr<HUTensor> in, HUPtr<HUTensor> gamma, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device, HUPtr<HUTensor> beta, float eps)
{
	auto outMem = mem->alloc<TT_DATA_TYPE>(in->size());
	auto out = HUNew<HUTensor>(outMem, in->shape(), device);
#ifdef FAST_LAYERNORM
    LayerNormalOP_V2(out, in, gamma, beta, eps);
#else
	LayerNormalOP(out, in, gamma, beta, eps);
#endif
	return out;
}

void HUTensorUtil::AddBiasActivation(HUPtr<HUTensor> &C, const HUPtr<HUTensor> bias, ActivationType type)
{
	if (type == ActivationType::GELU) {
	    AddBiasGeluOP(C, bias);
	}
	else if (type == ActivationType::RELU) {
		AddBiasReluOP(C, bias);
	}
	else if (type == ActivationType::SWISH) {
		AddBiasSwishOP(C, bias);
	}
	else {
		AddBiasGeluOP(C, bias);
	}
}

HUPtr<HUTensor> HUTensorUtil::Activation(HUPtr<HUTensor> in, ActivationType type, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device)
{
	auto outMem = mem->alloc<float>(in->size());
	auto out = HUNew<HUTensor>(outMem, in->shape(), device);

    if (type == ActivationType::GELU) {
        GeluOP(out, in);
    }
    else if (type == ActivationType::RELU) {
        ReluOP(out, in);
    }
    else if (type == ActivationType::SWISH) {
        SwishOP(out, in);
    }
    else {
        GeluOP(out, in);
    }

    return out;
}

HUPtr<HUTensor> HUTensorUtil::Gelu(HUPtr<HUTensor> in, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device)
{
    auto outMem = mem->alloc<float>(in->size());
    auto out = HUNew<HUTensor>(outMem, in->shape(), device);
    GeluOP(out, in);
    return out;
}

HUPtr<HUTensor> HUTensorUtil::Relu(HUPtr<HUTensor> in, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device)
{
	auto outMem = mem->alloc<float>(in->size());
	auto out = HUNew<HUTensor>(outMem, in->shape(), device);
	ReluOP(out, in);
	return out;
}

HUPtr<HUTensor> HUTensorUtil::Swish(HUPtr<HUTensor> in, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device)
{
	auto outMem = mem->alloc<float>(in->size());
    auto out = HUNew<HUTensor>(outMem, in->shape(), device);
    SwishOP(out, in);
    return out;
}

HUPtr<HUTensor> HUTensorUtil::Zeros(HUShape inShape, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device)
{
	auto outMem = mem->alloc<TT_DATA_TYPE>(inShape.elements());
	auto out = HUNew<HUTensor>(outMem, inShape, device);
	out->set((TT_DATA_TYPE)0.f);
	return out;
}

HUPtr<HUTensor> HUTensorUtil::Ones(HUShape inShape, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device)
{
    auto outMem = mem->alloc<TT_DATA_TYPE>(inShape.elements());
    auto out = HUNew<HUTensor>(outMem, inShape, device);
    out->set((TT_DATA_TYPE)1.0); 
    return out;
}

HUPtr<HUTensor> HUTensorUtil::Set(HUShape inShape, const float num, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device)
{
    auto outMem = mem->alloc<float>(inShape.elements());
    auto out = HUNew<HUTensor>(outMem, inShape, device);
    out->set(num);
    return out;
}

HUPtr<HUTensor> HUTensorUtil::CopyRows(HUPtr<HUTensor> in, const std::vector<size_t>& indices, HUPtr<HUMemPool> mem)
{
	auto outShape = GetCopyRowsShape(in, indices);
	auto outMem = mem->alloc<TT_DATA_TYPE>(outShape.elements());
	auto out = HUNew<HUTensor>(outMem, outShape, in->getDevice());
	CopyRowsOP(out, in, indices);
	return out;
}

HUPtr<HUTensor> HUTensorUtil::CopyRows_V2(HUPtr<HUTensor> in, size_t* indices, int num, HUPtr<HUMemPool> mem)
{
    // auto outShape = GetCopyRowsShape(in, indices);
    auto outShape = in->shape();
    ABORT_IF(outShape.size() != 2, "[TenTrans] rows operator can only be used with 2-dimensional tensors");
    outShape.set(0, num);
    auto outMem = mem->alloc<TT_DATA_TYPE>(outShape.elements());
    auto out = HUNew<HUTensor>(outMem, outShape, in->getDevice());
    CopyRowsOP_V2(out, in, indices, num);
    return out;
}

HUPtr<HUTensor> HUTensorUtil::Reshape(HUPtr<HUTensor> in, HUShape shape)
{
	return ReshapeOP(in, shape);
}

HUShape HUTensorUtil::GetConcatenateShape(std::vector<HUPtr<HUTensor> >& nodes, int &ax)
{
	HUShape shape = nodes.back()->shape();
    ax = shape.axis(ax);

    int sum = 0;
    for(auto child : nodes)
      sum += child->shape()[ax];
    shape.set(ax, sum);

    return shape;
}

HUPtr<HUTensor> HUTensorUtil::Concatenate(std::vector<HUPtr<HUTensor> > nodes, int ax, HUPtr<HUMemPool> mem)
{
	auto outShape = GetConcatenateShape(nodes, ax);
	auto outMem = mem->alloc<TT_DATA_TYPE>(outShape.elements());
	auto out = HUNew<HUTensor>(outMem, outShape, nodes.back()->getDevice());
	ConcatenateOP(out, nodes, ax);
	return out;
}

void HUTensorUtil::Split(HUPtr<HUTensor> in, int num, std::vector<HUPtr<HUTensor> > &nodes, HUPtr<HUMemPool> mem, int ax)
{
    HUShape outShape = in->shape();
    outShape.set(ax, in->shape()[ax]/num);
    auto newIn = Transpose(Reshape(in, {outShape[0], outShape[1], num, outShape[2]}), {2, 0, 1, 3}, mem, in->getDevice()); 
    for (int i = 0; i < num; i++)
    {
        // HUShape outShape = in->shape();
        // outShape.set(ax, in->shape()[ax]/num);
        auto outMem = mem->alloc<float>(outShape.elements());
        auto out = HUNew<HUTensor>(outMem, outShape, in->getDevice());
        // auto reShapeIn = Reshape(in, {outShape[0], outShape[1], num, outShape[2]}); 
        // out->set(reShapeIn->data() + (i * outShape.elements()), reShapeIn->data() + ((i+1) * outShape.elements()));
        out->set(newIn->data() + (i * outShape.elements()), newIn->data() + ((i+1) * outShape.elements()));
        nodes[i] = out;
    }
    mem->free(newIn->memory());
}

HUPtr<HUTensor> HUTensorUtil::Repeat(HUPtr<HUTensor> a, size_t repeats, int ax, HUPtr<HUMemPool> mem)
{
	if(repeats == 1)
    {
        HUPtr<HUTensor> out;
        CopyFrom(out, a, mem, a->getDevice());
		return out;
    }
	return Concatenate(std::vector<HUPtr<HUTensor> >(repeats, a), ax, mem);
}

HUPtr<HUTensor> HUTensorUtil::ConstantFloat(HUShape inShape, std::vector<TT_DATA_TYPE> data, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device)
{
	auto outMem = mem->alloc<TT_DATA_TYPE>(inShape.elements());
	auto out = HUNew<HUTensor>(outMem, inShape, device);
	out->set((TT_DATA_TYPE*)data.data(), (TT_DATA_TYPE*)data.data() + data.size());
	return out;
}

void HUTensorUtil::CopyFrom(HUPtr<HUTensor> &out, HUPtr<HUTensor> in, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device)
{
    auto outMem = mem->alloc<TT_DATA_TYPE>(in->size());
    out = HUNew<HUTensor>(outMem, in->shape(), device);
    out->set(in->data(), in->data() + in->size());
}

/*
void HUTensorUtil::Add_QKV_Bias_Transpose(
        HUPtr<HUTensor> buf_q,
        HUPtr<HUTensor> buf_k,
        HUPtr<HUTensor> buf_v,
        HUPtr<HUTensor> Q,
        HUPtr<HUTensor> b_Q,
        HUPtr<HUTensor> K,
        HUPtr<HUTensor> b_K,
        HUPtr<HUTensor> V,
        HUPtr<HUTensor> b_V,
        const int batch_size,
        const int seq_len,
        const int head_num,
        const int size_per_head)
{
    add_QKV_bias_transpose_kernelLauncher(
            buf_q->data(),
            buf_k->data(),
            buf_v->data(),
            Q->data(),
            b_Q->data(),
            K->data(),
            b_K->data(),
            V->data(),
            b_V->data(),
            batch_size,
            seq_len,
            head_num,
            size_per_head);
} */

/*
HUPtr<HUTensor> HUTensorUtil::ReduceSum(HUPtr<HUTensor> in, int ax, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device)
{
    HUShape inShape = in->shape();
    HUShape outShape.resize(inShape.size()-1);
    for (int i = outShape.size(); i >= 1 ; i--)
    {
        if (-i <= ax) {
            outShape.set(-i, inShape[-i-1]);
        }
        else {
            outShape.set(-i, inShape[-i]);
        }
    }
    
    auto outMem = mem->alloc<float>(outShape.elements());
    auto out = HUNew<HUTensor>(outMem, outShape, device);

    ReduceSumOP(out, in);
} */

HUPtr<HUTensor> HUTensorUtil::FusedQKVSelfAttention(
        const HUPtr<HUTensor> qkv_buf, const HUPtr<HUTensor> QKV_bias,
        HUPtr<HUTensor> key_cache, HUPtr<HUTensor> value_cache, 
        const int realDimBatch, uint8_t* isAllDone, 
        const int head_num, const int step, 
        HUPtr<HUMemPool> mem, HUPtr<HUDevice> device)
{
    auto outMem = mem->alloc<TT_DATA_TYPE>((qkv_buf->size()) / 3);
    int qkvHiddenSize = qkv_buf->shape()[-1];
    auto outShape = qkv_buf->shape();
    outShape.set(-1, qkvHiddenSize / 3);
    auto context_buf = HUNew<HUTensor>(outMem, outShape, device);

    MaskedMultiHeadAttentionOP(
            qkv_buf, QKV_bias,
            key_cache, value_cache,
            context_buf, 
            realDimBatch, isAllDone, 
            head_num, step);

    return context_buf;
}

HUPtr<HUTensor> HUTensorUtil::CrossAttention(
        HUPtr<HUTensor> query_buf, const HUPtr<HUTensor> Q_bias, 
        HUPtr<HUTensor> key_cache, const HUPtr<HUTensor> K_bias, 
        HUPtr<HUTensor> value_cache, const HUPtr<HUTensor> V_bias, 
        HUPtr<HUTensor> lengths, 
        const int realDimBatch, const uint8_t* isAllDone, 
        const int head_num, const int step, 
        HUPtr<HUMemPool> mem, HUPtr<HUDevice> device)
{
    auto outMem = mem->alloc<TT_DATA_TYPE>(query_buf->size());
    auto context_buf = HUNew<HUTensor>(outMem, query_buf->shape(), device);
    // LOG(trace, "[TenTrans][HUTensorUtil][CrossAttention] context_buf {}", context_buf->debug());

    CrossAttentionOP(
            query_buf, Q_bias, 
            key_cache, K_bias, 
            value_cache, V_bias, 
            lengths, context_buf, 
            realDimBatch, isAllDone, 
            head_num, step);

    return context_buf;
}

HUPtr<HUTensor> HUTensorUtil::EncoderUnFusedSelfAttention(
        HUPtr<HUTensor> input, HUPtr<HUTensor> att_mask, 
        const HUPtr<HUTensor> Q, const HUPtr<HUTensor> Q_bias, 
        const HUPtr<HUTensor> K, const HUPtr<HUTensor> K_bias, 
        const HUPtr<HUTensor> V, const HUPtr<HUTensor> V_bias, 
        const int head_num, EncoderSelfAttentionBuffer &params, 
        HUPtr<HUMemPool> mem, HUPtr<HUDevice> device)
{
    cudaSetDevice(input->getDeviceId().no);

    auto outMem = mem->alloc<TT_DATA_TYPE>(input->size());
    auto att_out = HUNew<HUTensor>(outMem, input->shape(), device);

    /* only multiply weight matrix */
    /// auto q_tmp = HUTensorUtil::Multiply(input, Q, mem, device);
    /// auto k_tmp = HUTensorUtil::Multiply(input, K, mem, device);
    /// auto v_tmp = HUTensorUtil::Multiply(input, V, mem, device);

    /*
    HUTensorUtil::Multiply_v2(params.q_tmp, input, Q);
    HUTensorUtil::Multiply_v2(params.k_tmp, input, K);
    HUTensorUtil::Multiply_v2(params.v_tmp, input, V);
    */

    TT_DATA_TYPE** qkv_kernel;
    cudaMalloc(&qkv_kernel, sizeof(TT_DATA_TYPE *) * 9);
    TT_DATA_TYPE** qkv_input;
    TT_DATA_TYPE** qkv_tmp;
    qkv_input = qkv_kernel + 3;
    qkv_tmp = qkv_input + 3;

    const TT_DATA_TYPE* hArray[] { Q->data(), K->data(), V->data(), 
                            input->data(), input->data(), input->data(),
                            (params.q_tmp)->data(), (params.k_tmp)->data(), (params.v_tmp)->data() };
    cudaMemcpyAsync((void*)qkv_kernel, hArray, sizeof(TT_DATA_TYPE *) * 9, cudaMemcpyHostToDevice);

    auto cublas_handle = input->getDevice()->getCublasHandle();
    TT_DATA_TYPE alpha = (TT_DATA_TYPE)1.0f, beta = (TT_DATA_TYPE)0.0f;
    
    const int batch_size = input->shape()[-3];
    const int seq_len = input->shape()[-2];
    const int hidden_units = input->shape()[-1];

    const int m = batch_size * seq_len;
    const int k = hidden_units;
    const int n = k;

    cublasGemmBatchedEx(cublas_handle, 
                        CUBLAS_OP_N, CUBLAS_OP_N, 
                        n, m, k, 
                        &alpha, 
                        (const void* const*) qkv_kernel, CUDA_R_32F, n, 
                        (const void* const*) qkv_input, CUDA_R_32F, k, 
                        &beta, 
                        (void* const*)qkv_tmp, CUDA_R_32F, n, 
                        3,
                        CUDA_R_32F, 
                        static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));

    cudaFree(qkv_kernel);

    /// int dimBatch = input->shape()[-3];
    /// int dimSeqLen = input->shape()[-2];
    /// int dimModel = input->shape()[-1];

    /* save the transpose result for attention */
    /// HUPtr<HUTensor> q_buf = HUTensorUtil::Zeros({dimBatch, head_num, dimModel/head_num, dimSeqLen}, mem, device);
    /// HUPtr<HUTensor> k_buf = HUTensorUtil::Zeros({dimBatch, head_num, dimModel/head_num, dimSeqLen}, mem, device);
    /// HUPtr<HUTensor> v_buf = HUTensorUtil::Zeros({dimBatch, head_num, dimSeqLen, dimModel/head_num}, mem, device);
    /* save the attention weights */
    /// HUPtr<HUTensor> qk_buf = HUTensorUtil::Zeros({dimBatch, head_num, dimSeqLen, dimSeqLen}, mem, device);
    /* save the transpose result of attention output */
    /// HUPtr<HUTensor> att_out_transpose_buf = HUTensorUtil::Zeros({dimBatch, head_num, dimSeqLen, dimModel/head_num}, mem, device);

    EncoderUnFusedSelfAttentionOP(
            params.q_tmp, Q_bias, 
            params.k_tmp, K_bias, 
            params.v_tmp, V_bias, 
            att_mask, att_out, 
            params.q_buf, params.k_buf, params.v_buf, params.qk_buf, 
            params.att_out_transpose_buf, head_num, mem);

    /*
    mem->free(q_tmp->memory());
    mem->free(k_tmp->memory());
    mem->free(v_tmp->memory());

    mem->free(q_buf->memory());
    mem->free(k_buf->memory());
    mem->free(v_buf->memory());
    mem->free(qk_buf->memory());
    mem->free(att_out_transpose_buf->memory());
    */

    return att_out;
}

// put k/v_buf from shape [B, H, L, Dh]
// to cache [B*H, Dh/x, L, x]  and [B, H, L, Dh]
void HUTensorUtil::Transpose4DBatchMajor(
        HUPtr<HUTensor> &k_dst, /*HUPtr<HUTensor> &v_dst,*/
        const HUPtr<HUTensor> k_src, /* const HUPtr<HUTensor> v_src, */
        HUPtr<HUMemPool> mem, HUPtr<HUDevice> device)
{
    int dimPerHead = k_src->shape()[-1];
    int dimSeqLen = k_src->shape()[-2];
    int dimHead = k_src->shape()[-3];
    int dimBatch = k_src->shape()[-4];
    int dimX = 4;

    auto k_dst_mem = mem->alloc<float>(k_src->size());
    auto k_dst_shape = HUShape({dimBatch*dimHead, dimPerHead/dimX, dimSeqLen, dimX}); 
    k_dst = HUNew<HUTensor>(k_dst_mem, k_dst_shape, device);

    /*
    auto v_dst_mem = mem->alloc<float>(v_src->size());
    auto v_dst_shape = HUShape({dimBatch, dimHead, dimSeqLen, dimPerHead});
    v_dst = HUNew<HUTensor>(v_dst_mem, v_dst_shape, device);
    */

    Transpose4DBatchMajorOP(
            k_dst, /*v_dst,*/
            k_src, /*v_src, */
            dimBatch, dimSeqLen,
            dimSeqLen, dimPerHead, dimHead);
}

void HUTensorUtil::UpdateKVBatchMajorCache(
        HUPtr<HUTensor> key_src_cache, HUPtr<HUTensor> key_tgt_cache,
        HUPtr<HUTensor> value_src_cache, HUPtr<HUTensor> value_tgt_cache,
        size_t* beams_ids, uint8_t* isAllDone, 
        const int batch_size, const int beam_width, const int head_num, const int step)
{

    UpdateKVBatchMajorCacheOP(
            key_src_cache, key_tgt_cache,
            value_src_cache, value_tgt_cache, 
            beams_ids, isAllDone, batch_size, 
            beam_width, head_num, step);
}

void HUTensorUtil::AddBiasInput(HUPtr<HUTensor> output, const HUPtr<HUTensor> bias, const HUPtr<HUTensor> input)
{
    AddBiasInputOP(output, bias, input);
}

HUPtr<HUTensor> HUTensorUtil::EmbeddingLookUpPositionEncoding(const HUPtr<HUTensor> word_emb, const HUPtr<HUTensor> pos_emb, const std::vector<size_t> &word_ids, const size_t startPos, bool isScale, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device)
{
    // std::cout << "[1]" << std::endl;
    int dimWords = word_ids.size();
    int dimModel = word_emb->shape()[-1];
    auto outMem = mem->alloc<TT_DATA_TYPE>(dimWords * dimModel);
    auto outShape = HUShape({dimWords, 1, dimModel});
    auto out = HUNew<HUTensor>(outMem, outShape, device);

    // std::cout << "[2]" << std::endl;
    EmbeddingLookUpPositionEncodingOP(out, word_emb, pos_emb, word_ids, startPos, isScale);
    // std::cout << "[3]" << std::endl;

    return out;
}

HUPtr<HUTensor> HUTensorUtil::StartIdEmbeddingLookUpPositionEncoding(const HUPtr<HUTensor> word_emb, const HUPtr<HUTensor> pos_emb, const std::vector<size_t> &word_ids, const int batch_size, bool isScale, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device)
{
    int dimSeqLen = word_ids.size() / batch_size;
    int dimModel = word_emb->shape()[-1];
    auto outMem = mem->alloc<TT_DATA_TYPE>(batch_size * dimSeqLen * dimModel);
    auto outShape = HUShape({batch_size, dimSeqLen, dimModel});
    auto out = HUNew<HUTensor>(outMem, outShape, device);

    StartIdEmbeddingLookUpPositionEncodingOP(out, word_emb, pos_emb, word_ids, batch_size, isScale);

    return out;
}

HUPtr<HUTensor> HUTensorUtil::BroadCastPlus(HUPtr<HUTensor> log_probs, HUPtr<HUTensor> cum_log_probs, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device)
{
    auto outMem = mem->alloc<TT_DATA_TYPE>(log_probs->size());
    auto out = HUNew<HUTensor>(outMem, log_probs->shape(), device);

    BroadCastPlusOP(out, log_probs, cum_log_probs);

    return out;
}

HUPtr<HUTensor> HUTensorUtil::BroadCastPlusWithBias(HUPtr<HUTensor> log_probs, HUPtr<HUTensor> cum_log_probs, const HUPtr<HUTensor> bias, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device)
{
    auto outMem = mem->alloc<TT_DATA_TYPE>(log_probs->size());
    auto out = HUNew<HUTensor>(outMem, log_probs->shape(), device);

    BroadCastPlusWithBiasOP(out, log_probs, cum_log_probs, bias);
    
    return out;
}

void HUTensorUtil::TopK(HUPtr<HUTensor> log_probs, std::vector<int> &topKIds, std::vector<float> &topKValues, const int vocab_size)
{
    TopKOP(log_probs, topKIds, topKValues, vocab_size);
}

/*
void HUTensorUtil::TopK_V2(HUPtr<HUTensor> log_probs, std::vector<int> &topKIds, std::vector<float> &topKValues, const int K, const int vocab_size)
{
    TopKOP_V2(log_probs, topKIds, topKValues, K, vocab_size);
}
*/

void HUTensorUtil::TopK_V2(HUPtr<HUTensor> log_probs, std::vector<int> &topKIds, std::vector<float> &topKValues, const int K, const int vocab_size, void* tmp_storage)
{
    // log_probs -> [batch, beam*vocab_size]
    TopKOP_V2(log_probs, topKIds, topKValues, K, vocab_size, tmp_storage);
}

void HUTensorUtil::TopKSoftmax(HUPtr<HUTensor> log_probs,
                               const HUPtr<HUTensor> bias,
                               std::vector<float> &cum_log_probs,
                               std::vector<int> &topKIds,
                               const int K,
                               void* temp_storage,
                               const int temp_storage_size,
                               uint8_t* isAllDone)
{
    TopKSoftmaxOP(log_probs, bias, cum_log_probs, topKIds, K, temp_storage, temp_storage_size, isAllDone);
}
/*
void HUTensorUtil::TopK(HUPtr<HUTensor> logProbs, const int K, HUPtr<HUTensor> topKIds, HUPtr<HUTensor> topKValues)
{

} */

} // namespace TenTrans
