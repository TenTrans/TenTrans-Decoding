#pragma once

#include <iostream>
#include <boost/program_options.hpp>
#include "yaml-cpp/yaml.h"
#include "HUUtil.h"
#include "HUGlobal.h"

namespace TenTrans{

enum struct HUConfigMode {
  classifying,
  translating,
  rescoring,
};

void OutputYaml(const YAML::Node node, YAML::Emitter& out);

class HUConfigParser{
public:
	HUConfigParser(int argc, char** argv, HUConfigMode mode)
      : mode_(mode),
    	cmdline_options_("Allowed options", 150) {
    	parseOptions(argc, argv);
  	}

	void parseOptions(int argc, char** argv);
	YAML::Node getConfig() const;
	std::vector<DeviceId> getDevices();
	
private:
	HUConfigMode mode_;
	boost::program_options::options_description cmdline_options_;
	YAML::Node config_;
	
	void addOptionsCommon(boost::program_options::options_description&);
	void addOptionsModel(boost::program_options::options_description&);
	void addOptionsTranslate(boost::program_options::options_description&);
    void addOptionsClassify(boost::program_options::options_description&);
}; 

}

