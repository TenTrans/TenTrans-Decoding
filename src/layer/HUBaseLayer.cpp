#include "HUBaseLayer.h"

namespace TenTrans{

HUPtr<HUShape> HUBaseLayer::GetShapeByModel(std::string pname, cnpy::npz_t modelNpz){
    HUPtr<HUShape> shape = HUNew<HUShape>();
    if(modelNpz[pname]->shape.size() == 1)
    {   
        shape->resize(2);
        shape->set(0,1);
        shape->set(1, modelNpz[pname]->shape[0]);
    }   
    else
    {   
        shape->resize(modelNpz[pname]->shape.size());
        for(int i = 0; i < modelNpz[pname]->shape.size(); ++i)
            shape->set(i, modelNpz[pname]->shape[i]);
    }   

    return shape;
}

}


