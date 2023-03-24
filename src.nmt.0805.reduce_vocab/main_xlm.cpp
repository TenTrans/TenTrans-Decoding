#include <iostream>
#include <fstream>
#include "HUConfig.h"
#include "HUConfigParser.h"
#include "HUVocab.h"
#include "HUDevice.h"
#include "HUGlobal.h"
#include "HUMemory.h"
#include "cnpy.h"

#include "HUEmbeddingLayer.h"
#include "HUMultiHeadAttention.h"
#include "HULayerNorm.h"
#include "HUFFNLayer.h"
#include "HUOutputLayer.h"
#include "HUUtil.h"
#include "HUData.h"
#include "HUTensorUtil.h"
#include "HUEncoderLayer.h"
#include "HUEncoder.h"

#include "Classifier.h"
#include <cuda_runtime.h>

/*
#include <time.h>
typedef long clock_t;
#define CLOCKS_PER_SEC ((clock_t)1000)
*/

/*
#include "HUShape.h"
#include "HUTensor.h"
#include "HUBaseLayer.h"
#include "HUFFNLayer.h"
*/

using namespace TenTrans;

int main(int argc, char** argv) 
{

    HUPtr<HUConfig> config = HUNew<HUConfig>();
    config->Init(argc, argv, HUConfigMode::translating);

    auto vocabs = config->get<std::vector<std::string>>("vocabs");
    HUPtr<HUVocab> srcVocab = HUNew<HUVocab>();
    srcVocab->Load(vocabs[0]);

    std::vector<DeviceId> deviceIDs = config->getDevices();
    HUPtr<HUDevice> device = HUNew<HUDevice>(deviceIDs[0]);

    HUPtr<HUMemPool> memPool = HUNew<HUMemPool>(device, 0, GROW, ALIGN);
    auto workspace = config->get<int>("workspace");
    std::cout << "[Workspace(MB)]: " << workspace <<std::endl;
    std::cout << "[Memory Pool(Byte)]: " << (workspace * MBYTE) << std::endl;
    memPool->Reserve(workspace * MBYTE);

    auto models = config->get<std::vector<std::string>>("models");
    auto modelNpz = cnpy::npz_load(models[0]);

    HUPtr<Classifier> classifier = HUNew<Classifier>(config, memPool, device, modelNpz);
    classifier->Init();

    ifstream fin1(argv[3]);
    if(!fin1.is_open())
    {
        printf("[Error] Load test file failed: %s\n", argv[3]);
        return 0;
    }

    /*
    clock_t start,ends;
    start=clock();
    */

    // cudaEvent_t A, B;
    int totalTokenNum = 0;
    int batch_id = 0;
    float totalElapsedTime = 0.0f;
    // cudaEventCreate(&A);
    // cudaEventCreate(&B);
    // cudaEventRecord(A, 0);

    /*
    cudaEvent_t start, stop;
    float elapsedTime = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    */
    // cudaEventRecord(start, 0);

    int batch_size = config->get<int>("mini-batch");
    std::vector<string> inputs;
    string source = "";
    int count = 0;

    while(getline(fin1, source))
    {
        count++;
        if (count % batch_size == 0)
        {
            batch_id++;
            inputs.push_back(source);
            // std::cout << "[batch_size]: " << inputs.size() << std::endl;
            HUPtr<HUTextInput> in = HUNew<HUTextInput>(inputs, srcVocab);
            HUPtr<HUBatch> batch = in->ToBatch(in->ToSents());

            /*
            auto batches = batch->data();
            for(auto& item: batches) {
                std::cout << item << " ";
            }
            std::cout << std::endl;
            */
            if (batch_id <= 50)
            {
                classifier->Forward(batch);
            }
            else
            {
                totalTokenNum += (batch->data()).size();

            // encoder->Forward_test(batch);

                cudaEvent_t start, stop;
                float elapsedTime = 0.0f;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);

                cudaEventRecord(start, 0);
                classifier->Forward(batch);

                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);

                cudaEventElapsedTime(&elapsedTime, start, stop);
                std::cout << "Time Cost(ms): " << elapsedTime << std::endl;
                totalElapsedTime += elapsedTime;

                cudaEventDestroy(start);
                cudaEventDestroy(stop);
            }

            inputs.clear();
            if (batch_id == 150)
                break;
        }
        else
        {
            inputs.push_back(source);
        }


    }

    if (inputs.size() > 0)
    {
        // std::cout << "[batch_size]: " << inputs.size() << std::endl;
        HUPtr<HUTextInput> in = HUNew<HUTextInput>(inputs, srcVocab);
        HUPtr<HUBatch> batch = in->ToBatch(in->ToSents());

        totalTokenNum += (batch->data()).size();

        /*
        auto batches = batch->data();
        for(auto& item: batches) {
            std::cout << item << " ";
        }
        std::cout << std::endl;
        */

        // encoder->Forward_test(batch);
        cudaEvent_t start, stop;
        float elapsedTime = 0.0f;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);

        classifier->Forward(batch);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&elapsedTime, start, stop);
        std::cout << "Time Cost(ms): " << elapsedTime << std::endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // cudaEventRecord(B, 0);
    // cudaEventSynchronize(B);

    // cudaEventElapsedTime(&totalElapsedTime, A, B);
    std::cout << "Total Time Cost(s): " << totalElapsedTime/1000.0 << std::endl;
    std::cout << "Speed(token/s): " << totalTokenNum*1000.0/ totalElapsedTime << std::endl;
    // std::cout << "Average Time Cost(ms): " << totalElapsedTime/(count/batch_size) << std::endl;

    // cudaEventDestroy(A);
    // cudaEventDestroy(B);

    return 0;
}
