#include "online_decoder.h"

void OnlineDecoder::Init(std::string configPath)
{
    /* loading configuration file  */
    // std::vector<string> argv{"-c", configPath};
    char** argv = new char*[3];
	const char* tmp1 = "TenTrans";
    const char* tmp = "-c";
    argv[0] = const_cast<char*>(tmp1);
	argv[1] = const_cast<char*>(tmp);
    argv[2] = const_cast<char*>(configPath.c_str());
	printf("[AM_TMP_DEBUG] the config path in the interface is %s\n", argv[2]);
    this->config_ = HUNew<HUConfig>();
    this->config_->Init(3, argv, HUConfigMode::translating);
	printf("[AM_TMP_DEBUG] the config is done\n");
    // delete argv;

    /* setting device */
    std::vector<DeviceId> deviceIDs = this->config_->getDevices();
    this->device_ = HUNew<HUDevice>(deviceIDs[0]);

    /* allocate device's memory */
    this->memPool_ = HUNew<HUMemPool>(this->device_, 0, GROW, ALIGN);
    auto workspace = this->config_->get<int>("workspace");
    this->memPool_->Reserve(workspace * MBYTE);

    /* loading model */
    auto models = this->config_->get<std::vector<std::string>>("models");
    this->modelNpz_ = cnpy::npz_load(models[0]);

    /* load source & target vocabulary */
    auto vocabs = this->config_->get<std::vector<std::string>>("vocabs");
    this->srcVocab_ = HUNew<HUVocab>();
    this->srcVocab_->Load(vocabs[0]);
    this->tgtVocab_ = HUNew<HUVocab>();
    this->tgtVocab_->Load(vocabs[1]);

    /* encoder-decoder framework */
    this->encDec_ = HUNew<HUEncoderDecoder>(this->config_, this->memPool_, this->device_, this->modelNpz_);
    this->encDec_->Init();

    /* beam search */
    size_t beamSize = this->config_->get<size_t>("beam-size");
    bool earlyStop = this->config_->get<bool>("early-stop");
    this->search_ = HUNew<HUBeamSearch>(this->config_, this->encDec_, beamSize, earlyStop, EOS_ID, UNK_ID, this->memPool_, this->device_);
}

void OnlineDecoder::DoJob(std::vector<string> &inputs, std::vector<string> &outputs)
{
    HUPtr<HUResult> result = HUNew<HUResult>(this->config_, this->tgtVocab_);
    size_t beamSize = this->config_->get<size_t>("beam-size");
    size_t miniBatch = inputs.size();

    int totalTokenNum = 0;
    HUPtr<HUTextInput> in = HUNew<HUTextInput>(inputs, this->srcVocab_);
    HUPtr<HUBatch> batch = in->ToBatch(in->ToSents());

    size_t batchWidth = batch->batchWidth();
    for (size_t i = 0; i < batchWidth*miniBatch; i++) {
        totalTokenNum += (size_t)batch->mask()[i];
    }

    auto histories = this->search_->Search(batch);
    for(auto history: histories)
    {
        string best1 = "";
        string bestn = "";

        int token_num = 0;
        result->GetTransResult(history, best1, bestn, token_num);
        std::cout << "[translation]: " << best1 << std::endl;
        outputs.push_back(best1);
    }
}