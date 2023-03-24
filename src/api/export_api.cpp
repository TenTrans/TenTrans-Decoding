#include "export_api.h"
#include "online_decoder.h"
#define MAX_BATCH_SIZE 16

void* __init(std::string configPath) {
  OnlineDecoder* online_decoder = new OnlineDecoder();
  online_decoder->Init(configPath);

  if (online_decoder != nullptr) {
    std::cout << "[INFO] Init Decoder Sucessfully !" << std::endl;
    return reinterpret_cast<void*>(online_decoder);
    // return (void*)online_decoder;
  }

  std::cout << "[ERROR] Init Decoder Failed !" << std::endl;
  return nullptr;
}

std::vector<std::string> __translate(void* handle, std::vector<std::string> sources) {
  if (handle == nullptr) {
    std::cout << "[ERROR] Init Decoder Processing Handle First !" << std::endl;
    return sources;
  }

  if (sources.size() == 0) {
    std::cout << "[WARNING] The Input is NULL !" << std::endl;
    return sources;
  }

  std::vector<std::string> results;
  OnlineDecoder*           online_decoder = reinterpret_cast<OnlineDecoder*>(handle);
  if (sources.size() > MAX_BATCH_SIZE) {
      std::cout << "[WARNING] Batch split !" << std::endl;
      std::vector<std::string> tmpSources, tmpResults;
      for (size_t i = 0; i < sources.size(); i++) {
        tmpSources.push_back(sources[i]);
        if ((i+1) %  MAX_BATCH_SIZE == 0) {
          online_decoder->DoJob(tmpSources, tmpResults);
          results.insert(results.end(), tmpResults.begin(), tmpResults.end());
          tmpSources.clear();
          tmpResults.clear();
        }
      }

      if (tmpSources.size() > 0) {
        online_decoder->DoJob(tmpSources, tmpResults);
        results.insert(results.end(), tmpResults.begin(), tmpResults.end());
        tmpSources.clear();
        tmpResults.clear();
      }
  }
  else {
    online_decoder->DoJob(sources, results);
  }

  return results;
}

void __destroy(void* handle) {
  if (handle == nullptr) {
    std::cout << "[WARNING] The Decoder Processing Handle is NULL !" << std::endl;
    return;
  }

  OnlineDecoder* online_decoder = (OnlineDecoder*)handle;
  delete online_decoder;
  online_decoder = nullptr;
}