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

#include "HUDecoder.h"
//#include "HUDecoderState.h"
#include "HUEncoderDecoder.h"
#include "HUBeamCell.h"
#include "HUHistory.h"
#include "HUNthElement.h"
#include "HUBeamSearch.h"
#include "HUResult.h"
// #include "NMT.h"

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

#include <map>

using namespace TenTrans;

int main(int argc, char** argv) {

    cudaEvent_t allStart, allStop;
    float allElapsedTime = 0.0f;
    cudaEventCreate(&allStart);
    cudaEventCreate(&allStop);
    cudaEventRecord(allStart, 0);

	/* load configuration file */
    HUPtr<HUConfig> config = HUNew<HUConfig>();
    config->Init(argc, argv, HUConfigMode::translating);
        
	/* device */
    std::vector<DeviceId> deviceIDs = config->getDevices();
    HUPtr<HUDevice> device = HUNew<HUDevice>(deviceIDs[0]);

	/* allocate device's memory */
    HUPtr<HUMemPool> memPool = HUNew<HUMemPool>(device, 0, GROW, ALIGN);
    auto workspace = config->get<int>("workspace");
    std::cout << "[Workspace(MB)]: " << workspace <<std::endl;
    std::cout << "[Memory Pool(Byte)]: " << (workspace * MBYTE) << std::endl;
    memPool->Reserve(workspace * MBYTE);

	/* load model */
    auto models = config->get<std::vector<std::string>>("models");
    auto modelNpz = cnpy::npz_load(models[0]);

    /* load source & target vocabulary */
    auto vocabs = config->get<std::vector<std::string>>("vocabs");
    HUPtr<HUVocab> srcVocab = HUNew<HUVocab>();
    srcVocab->Load(vocabs[0]);
    HUPtr<HUVocab> tgtVocab = HUNew<HUVocab>();
    tgtVocab->Load(vocabs[1]);

    std::map<size_t, std::vector<size_t>> shortList;
    auto shortListPath = config->get<std::string>("shortlist-path");
    ifstream vocabMapping(shortListPath);

    // std::cout << "ok0" << std::endl;
    string mappingLine = "";
    while(getline(vocabMapping, mappingLine))
    {
        // std::cout << "1: " << mappingLine << std::endl;
        vector<string> ret, tgtStrIndices;
        vector<size_t> tgtIndices;
        TrimLine(mappingLine);
        // std::cout << "2: " << mappingLine << std::endl;
        // std::cout << "ok1" << std::endl;
        StringUtil::split_v2(mappingLine, "||||", ret);
        // std::cout << "ok2" << std::endl;
        // std::cout << "3: " << ret[0] << "\t" << ret[1] << std::endl;

        int srcIndex = atoi(ret[0].c_str());
        // std::cout << "srcIndex: " << srcIndex << std::endl;
        // std::cout << ret[1] << std::endl;
        StringUtil::split_v2(ret[1], " ", tgtStrIndices);
        // std::cout << "ok3" << std::endl;
        for(int i = 0; i < tgtStrIndices.size(); i++) {
            // int aa = atoi(tgtStrIndices[i].c_str());
            // std::cout << aa << " ";
            tgtIndices.push_back(atoi(tgtStrIndices[i].c_str()));
        }
        // std::cout << std::endl;
        shortList[srcIndex] = tgtIndices;
        // std::cout << "ok4" << std::endl;
    }

    /*
    std::cout << "shortlist test ..." << std::endl;
    for (int i = 0; i < shortList[100].size(); i++) {
        std::cout << shortList[100][i] << "\t";
    }
    std::cout << std::endl;
    */

    /* load shortlist for vocabulary selection */
    // cnpy::npz_t vocabShortListNpz = NULL;

    /*
    auto useShortList = config->get<bool>("use-shortlist");
    if (useShortList) 
    {
        auto vocabShortList = config->get<std::string>("shortlist-path");
        auto vocabShortListNpz = cnpy::npz_load(vocabShortList);
    }
    else
    {
        auto vocabShortList = config->get<std::string>("shortlist-path");
        auto vocabShortListNpz = cnpy::npz_load(vocabShortList);
    } */

    /*
    auto vocabShortList = config->get<std::string>("shortlist-path");
    auto vocabShortListNpz = cnpy::npz_load(vocabShortList);
    */

    // HUPtr<NMT> translator = HUNew<NMT>(config, memPool, device, modelNpz);
    HUPtr<HUEncoderDecoder> encdec = HUNew<HUEncoderDecoder>(config, memPool, device, modelNpz);
    encdec->Init();
   
    ifstream fin1(argv[3]);
    ofstream fout1(argv[4]);
    // std::vector<std::string> bestResults;
    if(!fin1.is_open())
    {
        printf("[Error] Load test file failed: %s\n", argv[3]);
        return 0;
    }

    if(!fout1.is_open())
    {
        printf("[Error] Load test file failed: %s\n", argv[4]);
        return 0;
    } 

    float totalCost = 0.0f;
    int totalTokenNum = 0;
    std::vector<string> inputs;
    string source = "";
    int count=0;
    HUPtr<HUResult> result = HUNew<HUResult>(config, tgtVocab);
    size_t beamSize = config->get<size_t>("beam-size");
    size_t miniBatch = config->get<size_t>("mini-batch");
    std::cout << "[beam size]: " << beamSize << std::endl;
    bool earlyStop = config->get<bool>("early-stop");
    bool useShortList = config->get<bool>("use-shortlist");
    HUPtr<HUBeamSearch> search = HUNew<HUBeamSearch>(config, encdec, beamSize, earlyStop, useShortList, EOS_ID, UNK_ID, memPool, device);
    while(getline(fin1, source))
    {
        count++; 
        if(count % miniBatch == 0)
        {
            cudaEvent_t start, stop;
            float elapsedTime = 0.0f;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);

            inputs.push_back(source);
            HUPtr<HUTextInput> in = HUNew<HUTextInput>(inputs, srcVocab/*, vocabShortListNpz*/);
            HUPtr<HUBatch> batch = in->ToBatch(in->ToSents(), shortList);
            //// std::vector<HUSentence> sents = in->ToSents();
            //// HUPtr<HUBatch> batch = in->ToBatch(sents);
            auto batches = batch->data();
        
            size_t batchWidth = batch->batchWidth();
            for (size_t i = 0; i < batchWidth*miniBatch; i++) { 
                // curMaxLen += (size_t)batch->mask()[mskId];
                totalTokenNum += (size_t)batch->mask()[i];
            }

            auto histories = search->Search(batch);

            for(auto history : histories)
            {
                string best1="";
                string bestn="";
                int token_num = 0;
                result->GetTransResult(history, batch->newVocab(), best1, bestn, token_num);
                std::cout << "[translation]: " << best1 << std::endl;
                // result->Add((long)history->GetLineNum(), best1, bestn);
                // std::cout << best1 << std::endl;
                // totalTokenNum += token_num;

                fout1 << best1 << "\n";
                // bestResults.push_back(best1+"\n");
            }

            /*
            for(int i = 0; i < sents.size(); i++) {
                HUSentence* ptr = &sents[i];
                if (ptr != NULL) {
                    delete ptr;
                    ptr = NULL;
                }
            */

                /*
                if (sents[i] != NULL) {
                    delete sents[i];
                    sents[i] = NULL;
                }

            }
            */

            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);
            std::cout << "Time Cost(ms): " << elapsedTime << std::endl;
            totalCost += elapsedTime;

            // auto translations = result->Collect(config->get<bool>("n-best"));
            // string resultStr = StringUtil::Join(translations, "\n");
            // std::cout << resultStr << std::endl;
            inputs.clear();
        }
        else
        {   
            inputs.push_back(source);
        }
    }

    if (inputs.size() > 0)
    {
        cudaEvent_t start, stop;
        float elapsedTime = 0.0f;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        HUPtr<HUTextInput> in = HUNew<HUTextInput>(inputs, srcVocab/*, vocabShortListNpz*/);
        HUPtr<HUBatch> batch = in->ToBatch(in->ToSents(), shortList);
        //// std::vector<HUSentence> sents = in->ToSents();
        //// HUPtr<HUBatch> batch = in->ToBatch(sents);
        auto batches = batch->data();

        size_t batchWidth = batch->batchWidth();
        for (size_t i = 0; i < batchWidth*inputs.size(); i++) {
            // curMaxLen += (size_t)batch->mask()[mskId];
            totalTokenNum += (size_t)batch->mask()[i];
        }

        auto histories = search->Search(batch);
        for(auto history : histories)
        {
            string best1="";
            string bestn="";
            int token_num = 0;
            result->GetTransResult(history, batch->newVocab(), best1, bestn, token_num);
            std::cout << "[translation]: " << best1 << std::endl;
            // result->Add((long)history->GetLineNum(), best1, bestn);
            // std::cout << best1 << std::endl;
            // totalTokenNum += token_num;
            fout1 << best1 << "\n";
            // bestResults.push_back(best1+"\n");
        }

        /*
        for(int i = 0; i < sents.size(); i++) {
            HUSentence* ptr = &sents[i];
            if (ptr != NULL) {
                delete ptr;
                ptr = NULL;
            }
        */
            /*
            if (sents[i] != NULL) {
                delete sents[i];
                sents[i] = NULL;
            }
        }
        */

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        std::cout << "Time Cost(ms): " << elapsedTime << "\tcur_batch: " << inputs.size() << std::endl;
        totalCost += elapsedTime;
    }

    cudaEventRecord(allStop, 0);
    cudaEventSynchronize(allStop);
    cudaEventElapsedTime(&allElapsedTime, allStart, allStop);

    std::cout << "full process time(s): " << allElapsedTime/1000.0f << std::endl;
    std::cout << "total time(s): " << totalCost/1000.0f << std::endl;
    std::cout << "total tokens(source): " << totalTokenNum << std::endl;
    std::cout << "avg time(ms): " << totalCost * 1.0 / count * miniBatch << std::endl;
    std::cout << "speed(token/s): " << totalTokenNum * 1000.0 / totalCost << std::endl;

    return 0;
}
