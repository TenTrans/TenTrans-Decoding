#include "export_api.h"
#include <iostream>
using namespace std;

int main()
{
    string configPath = "../../tools/config.b16.yml";
    std::vector<string> sources;
    sources.push_back("Yesterday , Gut@@ acht &apos;s Mayor gave a clear answer to this question .");

    void *handle = __init(configPath);
    std::vector<std::string> results = __translate(handle, sources);
    for(int i = 0; i < results.size(); i++)
    {
        std::cout << "[translation]: " << results[i] << std::endl;
    }

    /*
    for(auto result: results) {
        std::cout << "[translation]: " << result << std::endl; 
    }
    */

    __destroy(handle);

    return 0;
}
