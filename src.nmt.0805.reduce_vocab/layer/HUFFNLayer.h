
#pragma once
#include <cuda_runtime.h>
#include <map>
#include "HUBaseLayer.h"
using namespace std;

namespace TenTrans{

enum FFNEnum {W1, W2, b1, b2};

/* activation function for FFN layer */
enum ActivationFuncCode {Gelu, Relu, Swish, InvalidOption};
static ActivationFuncCode ResolveActivationOptions(const std::string input)
{
    static const std::map<const std::string, ActivationFuncCode> optionStrings { 
        {"gelu", Gelu},
        {"relu", Relu},
        {"swish", Swish}
    };

    auto itr = optionStrings.find(input);
    if(itr != optionStrings.end()) { 
        return itr->second;
    }

    return InvalidOption;
};


class HUFFNLayer : public HUBaseLayer{

public:
	HUFFNLayer(HUPtr<HUConfig> options, HUPtr<HUMemPool> memoryPool, HUPtr<HUDevice> device, cnpy::npz_t modelNpz, bool isEncoder, int layerId);
    ~HUFFNLayer();

	void Init();
	HUPtr<HUTensor> Forward(HUPtr<HUTensor> in);
    HUPtr<HUTensor> GetBias2() { return this->ffn_b2; };  // for basic_kernel_fusion

private:
	void NewBySuffix(FFNEnum e, string param);
	void InitBySuffix(FFNEnum e, string param);

	int layerId_;
	HUPtr<HUTensor> ffn_W1;
	HUPtr<HUTensor> ffn_W2;

	ActivationType  activationType_;

	HUPtr<HUTensor> ffn_b1;
    HUPtr<HUTensor> ffn_b2;

	string prefix_;
};

}
