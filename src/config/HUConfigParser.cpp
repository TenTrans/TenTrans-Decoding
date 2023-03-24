#include <iostream>
#include "HUConfigParser.h"
#include "HUConfig.h"
#include "Logging.h"
//using namespace std;
namespace po = boost::program_options;

#define SET_OPTION(key, type)                    \
  do {                                           \
    if(!vm_[key].defaulted() || !config_[key]) { \
      config_[key] = vm_[key].as<type>();        \
    }                                            \
  } while(0)

#define SET_OPTION_NONDEFAULT(key, type)  \
  do {                                    \
    if(vm_.count(key) > 0) {              \
      config_[key] = vm_[key].as<type>(); \
    }                                     \
  } while(0)


namespace TenTrans{

void OutputYaml(const YAML::Node node, YAML::Emitter& out) {
  std::set<std::string> sorter;
  switch(node.Type()) {
    case YAML::NodeType::Null: out << node; break;
    case YAML::NodeType::Scalar: out << node; break;
    case YAML::NodeType::Sequence:
      out << YAML::BeginSeq;
      for(auto&& n : node)
        OutputYaml(n, out);
      out << YAML::EndSeq;
      break;
    case YAML::NodeType::Map:
      for(auto& n : node)
        sorter.insert(n.first.as<std::string>());
      out << YAML::BeginMap;
      for(auto& key : sorter) {
        out << YAML::Key;
        out << key; 
        out << YAML::Value;
        OutputYaml(node[key], out);
      }    
      out << YAML::EndMap;
      break;
    case YAML::NodeType::Undefined: out << node; break;
  }
}

void HUConfigParser::addOptionsCommon(po::options_description& desc) {
  int defaultWorkspace = (mode_ == HUConfigMode::translating) ? 4096*1.5: 4096*1.5;

  po::options_description general("General options", 150);
  // clang-format off
  general.add_options()
    ("config,c", po::value<std::string>(),
     "Configuration file")
    ("workspace,w", po::value<size_t>()->default_value(defaultWorkspace),
      "Preallocate  arg  MB of work space")
    ("log", po::value<std::string>(),
     "Log training process information to file given by  arg")
    ("logger", po::value<std::string>(), "logger name")
    ("log-level", po::value<std::string>()->default_value("info"),
     "Set verbosity level of logging "
     "(trace - debug - info - warn - err(or) - critical - off)")
    ("quiet", po::value<bool>()->zero_tokens()->default_value(false),
     "Suppress all logging to stderr. Logging to files still works")
    ("quiet-translation", po::value<bool>()->zero_tokens()->default_value(false),
     "Suppress logging for translation")
    ("seed", po::value<size_t>()->default_value(0),
     "Seed for all random number generators. 0 means initialize randomly")
    ("relative-paths", po::value<bool>()->zero_tokens()->default_value(false),
     "All paths are relative to the config file location")
    ("dump-config", po::value<bool>()->zero_tokens()->default_value(false),
     "Dump current (modified) configuration to stdout and exit")
    ("version", po::value<bool>()->zero_tokens()->default_value(false),
      "Print version number and exit")
    ("help,h", po::value<bool>()->zero_tokens()->default_value(false),
      "Print this help message and exit")
  ;
  // clang-format on
  desc.add(general);
}

void HUConfigParser::addOptionsModel(po::options_description& desc) {

	po::options_description model("Model options", 150);
  	if(mode_ == HUConfigMode::translating) {
    	model.add_options()
    	("models,m", po::value<std::vector<std::string>>()->multitoken(),
     	"Paths to model(s) to be loaded");
  	}
	
	model.add_options()
    ("ignore-model-config", po::value<bool>()->zero_tokens()->default_value(false),
     "Ignore the model configuration saved in npz file")
    ("type", po::value<std::string>()->default_value("transformer"),
      "Model type (possible values: amun, nematus, s2s, multi-s2s, transformer)")
    ("dim-vocabs", po::value<std::vector<int>>()
      ->multitoken()->default_value(std::vector<int>({0, 0}), "0 0"), 
      "Maximum items in vocabulary ordered by rank, 0 uses all items in the "
      "provided/created vocabulary file")
    // ("dim-emb", po::value<int>()->default_value(512),
    //  "Size of embedding vector")
    ("dim-emb", po::value<int>()->default_value(2048), 
     "Size of embedding vector")

    ("dim-rnn", po::value<int>()->default_value(1024),
     "Size of rnn hidden state")
    ("enc-type", po::value<std::string>()->default_value("bidirectional"),
     "Type of encoder RNN : bidirectional, bi-unidirectional, alternating (s2s)")
    ("enc-cell", po::value<std::string>()->default_value("gru"),
     "Type of RNN cell: gru, lstm, tanh (s2s)")
    ("enc-cell-depth", po::value<int>()->default_value(1),
     "Number of transitional cells in encoder layers (s2s)")
    ("enc-depth", po::value<int>()->default_value(12),
     "Number of encoder layers (s2s)")
    ("dec-cell", po::value<std::string>()->default_value("gru"),
     "Type of RNN cell: gru, lstm, tanh (s2s)")
    ("dec-cell-base-depth", po::value<int>()->default_value(2),
     "Number of transitional cells in first decoder layer (s2s)")
    ("dec-cell-high-depth", po::value<int>()->default_value(1),
     "Number of transitional cells in next decoder layers (s2s)")
    ("dec-depth", po::value<int>()->default_value(1),
     "Number of decoder layers (s2s)")
    //("dec-high-context", po::value<std::string>()->default_value("none"),
    // "Repeat attended context: none, repeat, conditional, conditional-repeat (s2s)")
    ("skip", po::value<bool>()->zero_tokens()->default_value(false),
     "Use skip connections (s2s)")
    ("layer-normalization", po::value<bool>()->zero_tokens()->default_value(false),
     "Enable layer normalization")
    ("right-left", po::value<bool>()->zero_tokens()->default_value(false),
     "Train right-to-left model")
    ("best-deep", po::value<bool>()->zero_tokens()->default_value(false),
     "Use Edinburgh deep RNN configuration (s2s)")
    ("special-vocab", po::value<std::vector<size_t>>()->multitoken(),
     "Model-specific special vocabulary ids")
    ("tied-embeddings", po::value<bool>()->zero_tokens()->default_value(false),
     "Tie target embeddings and output embeddings in output layer")
    ("tied-embeddings-src", po::value<bool>()->zero_tokens()->default_value(false),
     "Tie source and target embeddings")
    ("tied-embeddings-all", po::value<bool>()->zero_tokens()->default_value(false),
     "Tie all embedding layers and output layer")
    ("use-emb-scale", po::value<bool>()->zero_tokens()->default_value(true), 
     "Wheather scale input word embedding")
    ("early-stop", po::value<bool>()->zero_tokens()->default_value(false), 
     "use early-stop strategy for beam search.")
    // wheather use fp16
    ("use-fp16", po::value<bool>()->zero_tokens()->default_value(false), 
     "Wheather use fp16")
    ("transformer-heads", po::value<int>()->default_value(8),
     "Number of head in multi-head attention (transformer)")
    ("transformer-dim-ffn", po::value<int>()->default_value(8192),
     "Size of position-wise feed-forward network (transformer)")
    ("transformer-ffn-activation", po::value<std::string>()->default_value("gelu"),
     "Activation between filters: swish or relu (transformer)")
    ("transformer-preprocess", po::value<std::string>()->default_value(""),
     "Operation before each transformer layer: d = dropout, a = add, n = normalize")
    ("transformer-postprocess-emb", po::value<std::string>()->default_value("d"),
     "Operation after transformer embedding layer: d = dropout, a = add, n = normalize")
    ("transformer-postprocess", po::value<std::string>()->default_value("dan"),
     "Operation after each transformer layer: d = dropout, a = add, n = normalize")
#ifdef CUDNN
    ("char-stride", po::value<int>()->default_value(5),
     "Width of max-pooling layer after convolution layer in char-s2s model")
    ("char-highway", po::value<int>()->default_value(4),
     "Number of highway network layers after max-pooling in char-s2s model")
    ("char-conv-filters-num", po::value<std::vector<int>>()
      ->default_value(std::vector<int>({200, 200, 250, 250, 300, 300, 300, 300}),
                                      "200 200 250 250 300 300 300 300")
      ->multitoken(),
     "Numbers of convolution filters of correspoding width in char-s2s model")
    ("char-conv-filters-widths", po::value<std::vector<int>>()
     ->default_value(std::vector<int>({1, 2, 3, 4, 5, 6, 7, 8}), "1 2 3 4 5 6 7 8")
      ->multitoken(),
     "Convolution window widths in char-s2s model")
#endif
    ;

	desc.add(model);
}


void HUConfigParser::addOptionsClassify(po::options_description& desc) {
    po::options_description classifier("Classifier options", 150);
    
    classifier.add_options()
        ("models,m", po::value<std::vector<std::string>>()->multitoken(), 
         "Paths to model(s) to be loaded")

        ("ignore-model-config", po::value<bool>()->zero_tokens()->default_value(false), 
         "Ignore the model configuration saved in npz file")
        
        ("type", po::value<std::string>()->default_value("transformer"), 
         "Model type (possible values: amun, nematus, s2s, multi-s2s, transformer)")

        ("dim-emb", po::value<int>()->default_value(2048), 
         "Size of embedding vector")

        ("transformer-dim-ffn", po::value<int>()->default_value(8192), 
         "Size of position-wise feed-forward network (transformer)")

        ("transformer-heads", po::value<int>()->default_value(8), 
         "Number of head in multi-head attention (transformer)")

        ("enc-depth", po::value<int>()->default_value(12), 
         "Number of encoder layers (s2s)")

        ("transformer-ffn-activation", po::value<std::string>()->default_value("gelu"), 
         "Activation between filters: swish, relu (transformer) or gelu(XLM)")

        ("enc-type", po::value<std::string>()->default_value("bidirectional"), 
         "Type of encoder RNN : bidirectional, bi-unidirectional, alternating (s2s)")

        ("input,i", po::value<std::vector<std::string>>()
         ->multitoken()->default_value(std::vector<std::string>({"stdin"}), "stdin"), 
         "Paths to input file(s), stdin by default")

        ("vocabs,v", po::value<std::vector<std::string>>()->multitoken(), 
         "Paths to vocabulary files have to correspond to --input")

        ("max-length", po::value<size_t>()->default_value(1000), 
         "Maximum length of a sentence in a training sentence pair")

        ("max-length-crop", po::value<bool>()->zero_tokens()->default_value(false), 
         "Crop a sentence to max-length instead of ommitting it if longer than max-length")

        ("devices,d", po::value<std::vector<std::string>>()
         ->multitoken()->default_value(std::vector<std::string>({"0"}), "0"), 
         "GPUs to use for translating")

#ifdef CUDA_FOUND
        ("cpu-threads", po::value<size_t>()->default_value(0)->implicit_value(1), 
         "Use CPU-based computation with this many independent threads, 0 means GPU-based computation")
#else
        ("cpu-threads", po::value<size_t>()->default_value(1), 
         "Use CPU-based computation with this many independent threads, 0 means GPU-based computation")
#endif
        ("mini-batch", po::value<int>()->default_value(1), 
         "Size of mini-batch used during update") 
        
        ("maxi-batch", po::value<int>()->default_value(1), 
         "Number of batches to preload for length-based sorting") 
        
        ("maxi-batch-sort", po::value<std::string>()->default_value("none"), 
         "Sorting strategy for maxi-batch: none (default) src") 

        ("n-best", po::value<bool>()->zero_tokens()->default_value(false), 
         "Display n-best list")
    ;

    desc.add(classifier);

}


void HUConfigParser::addOptionsTranslate(po::options_description& desc) {
  po::options_description translate("Translator options", 150);
  // clang-format off
  translate.add_options()
    ("input,i", po::value<std::vector<std::string>>()
      ->multitoken()
      ->default_value(std::vector<std::string>({"stdin"}), "stdin"),
      "Paths to input file(s), stdin by default")
    ("vocabs,v", po::value<std::vector<std::string>>()->multitoken(),
      "Paths to vocabulary files have to correspond to --input")
    ("beam-size,b", po::value<size_t>()->default_value(12),
      "Beam size used during search")
    ("normalize,n", po::value<float>()->default_value(0.f)->implicit_value(1.f),
      "Divide translation score by pow(translation length, arg) ")
    ("allow-unk", po::value<bool>()->zero_tokens()->default_value(false),
      "Allow unknown words to appear in output")
    ("max-length", po::value<size_t>()->default_value(1000),
      "Maximum length of a sentence in a training sentence pair")
    ("max-length-crop", po::value<bool>()->zero_tokens()->default_value(false),
      "Crop a sentence to max-length instead of ommitting it if longer than max-length")
    ("devices,d", po::value<std::vector<std::string>>()
      ->multitoken()
      ->default_value(std::vector<std::string>({"0"}), "0"),
      "GPUs to use for translating")
#ifdef CUDA_FOUND
    ("cpu-threads", po::value<size_t>()->default_value(0)->implicit_value(1),
      "Use CPU-based computation with this many independent threads, 0 means GPU-based computation")
    //("omp-threads", po::value<size_t>()->default_value(1),
    //  "Set number of OpenMP threads for each CPU-based thread")
#else
    ("cpu-threads", po::value<size_t>()->default_value(1),
      "Use CPU-based computation with this many independent threads, 0 means GPU-based computation")
#endif
    ("mini-batch", po::value<int>()->default_value(1),
      "Size of mini-batch used during update")
    ("maxi-batch", po::value<int>()->default_value(1),
      "Number of batches to preload for length-based sorting")
    ("maxi-batch-sort", po::value<std::string>()->default_value("none"),
      "Sorting strategy for maxi-batch: none (default) src")
    ("n-best", po::value<bool>()->zero_tokens()->default_value(false),
      "Display n-best list")
    //("lexical-table", po::value<std::string>(),
    // "Path to lexical table")
    ("weights", po::value<std::vector<float>>()
      ->multitoken(),
      "Scorer weights")
    // TODO: the options should be available only in server
    ("port,p", po::value<size_t>()->default_value(8080),
      "Port number for web socket server")
  ;
  // clang-format on
  desc.add(translate);
}

/*
void HUConfigParser::parseOptions_v2(int argc, char** argv)
{

} */

void HUConfigParser::parseOptions(int argc, char** argv)
{
	addOptionsCommon(cmdline_options_);
	addOptionsModel(cmdline_options_);
    // addOptionsModel(cmdline_options_);

	switch(mode_) {
    case HUConfigMode::translating:
      // addOptionsModel(cmdline_options_);
      addOptionsTranslate(cmdline_options_);
      break;
    // case HUConfigMode::classifying:
    //   addOptionsClassify(cmdline_options_);
    //   break;
  	}
    // addOptionsClassify(cmdline_options_);

	boost::program_options::variables_map vm_;
  	try {
    	po::store(
        	po::command_line_parser(argc, argv).options(cmdline_options_).run(),
        	vm_);
    	po::notify(vm_);
  	} catch(std::exception& e) {
    	std::cerr << "Error: " << e.what() << std::endl << std::endl;
   		std::cerr << "Usage: " + std::string(argv[0]) + " [options]" << std::endl;
    	std::cerr << cmdline_options_ << std::endl;
    	exit(1);
  	}

  	if(vm_["help"].as<bool>()) {
    	std::cerr << "Usage: " + std::string(argv[0]) + " [options]" << std::endl;
    	std::cerr << cmdline_options_ << std::endl;
    	exit(0);
  	}

  	if((mode_ == HUConfigMode::translating) || (mode_ == HUConfigMode::classifying)) {
    	if(vm_.count("models") == 0 && vm_.count("config") == 0) {
      	std::cerr << "Error: you need to provide at least one model file or a "
        	           "config file"
            	    << std::endl
                	<< std::endl;

      	std::cerr << "Usage: " + std::string(argv[0]) + " [options]" << std::endl;
      	std::cerr << cmdline_options_ << std::endl;
      	exit(0);
    	}
  	}

	bool loadConfig = vm_.count("config");
	std::string configPath;
	if(loadConfig) {
   		configPath = vm_["config"].as<std::string>();
		try{
			//LOG(info, "[TenTrans][config] Load confiuration file {}", configPath);
	    	config_ = YAML::LoadFile(configPath);
		}
		catch(std::exception& e){
			std::cerr << "Error: " << e.what() << std::endl << std::endl;
		}
  	}

	if((mode_ == HUConfigMode::translating) || (mode_ == HUConfigMode::classifying)) {
		SET_OPTION_NONDEFAULT("models", std::vector<std::string>);
  	}

	if(!vm_["vocabs"].empty()) {
   		config_["vocabs"] = vm_["vocabs"].as<std::vector<std::string>>();
  	}

	SET_OPTION("ignore-model-config", bool);
  	SET_OPTION("type", std::string);
  	SET_OPTION("dim-vocabs", std::vector<int>);
  	SET_OPTION("dim-emb", int);
  	SET_OPTION("dim-rnn", int);

  	SET_OPTION("enc-type", std::string);
  	SET_OPTION("enc-cell", std::string);
  	SET_OPTION("enc-cell-depth", int);
  	SET_OPTION("enc-depth", int);

  	SET_OPTION("dec-cell", std::string);
  	SET_OPTION("dec-cell-base-depth", int);
  	SET_OPTION("dec-cell-high-depth", int);
  	SET_OPTION("dec-depth", int);

  	SET_OPTION("skip", bool);
  	SET_OPTION("tied-embeddings", bool);
  	SET_OPTION("tied-embeddings-src", bool);
  	SET_OPTION("tied-embeddings-all", bool);
  	SET_OPTION("layer-normalization", bool);
  	SET_OPTION("right-left", bool);
  	SET_OPTION("transformer-heads", int);
  	SET_OPTION("transformer-preprocess", std::string);
  	SET_OPTION("transformer-postprocess", std::string);
  	SET_OPTION("transformer-postprocess-emb", std::string);
  	SET_OPTION("transformer-dim-ffn", int);
  	SET_OPTION("transformer-ffn-activation", std::string);

#ifdef CUDNN
  	SET_OPTION("char-stride", int);
  	SET_OPTION("char-highway", int);
  	SET_OPTION("char-conv-filters-num", std::vector<int>);
  	SET_OPTION("char-conv-filters-widths", std::vector<int>);
#endif

  	SET_OPTION("best-deep", bool);
  	SET_OPTION_NONDEFAULT("special-vocab", std::vector<size_t>);	

	if((mode_ == HUConfigMode::translating) || (mode_ == HUConfigMode::classifying)) {
    	SET_OPTION("input", std::vector<std::string>);
    	SET_OPTION("beam-size", size_t);
    	SET_OPTION("normalize", float);
    	SET_OPTION("allow-unk", bool);
    	SET_OPTION("n-best", bool);
    	SET_OPTION_NONDEFAULT("weights", std::vector<float>);
    	SET_OPTION("port", size_t);
  	}

	SET_OPTION("workspace", size_t);
  	SET_OPTION("log-level", std::string);
  	SET_OPTION_NONDEFAULT("logger", std::string);
  	SET_OPTION("quiet", bool);
  	SET_OPTION("quiet-translation", bool);
  	SET_OPTION_NONDEFAULT("log", std::string);
  	SET_OPTION("seed", size_t);
  	SET_OPTION("relative-paths", bool);
  	SET_OPTION("devices", std::vector<std::string>);
  	SET_OPTION("cpu-threads", size_t);

  	SET_OPTION("mini-batch", int);
  	SET_OPTION("maxi-batch", int);

  	SET_OPTION("maxi-batch-sort", std::string);
  	SET_OPTION("max-length", size_t);
  	SET_OPTION("max-length-crop", bool);

	if(vm_["dump-config"].as<bool>()) {
		YAML::Emitter emit;
    	OutputYaml(config_, emit);
    	std::cout << emit.c_str() << std::endl;
	}
}

YAML::Node HUConfigParser::getConfig() const {
	return config_;
}

std::vector<DeviceId> HUConfigParser::getDevices()
{
	std::vector<DeviceId> devices;

	try {
		std::string devicesStr = StringUtil::Join(config_["devices"].as<std::vector<std::string>>());
		if(mode_ == HUConfigMode::translating) {
			for(auto d : StringUtil::split(devicesStr, " "))
				devices.push_back({std::stoull(d), DeviceType::gpu});
		}

		/*if(config_["cpu-threads"].as<size_t>() > 0) {
			devices.clear();
			for(size_t i = 0; i < config_["cpu-threads"].as<size_t>(); ++i)
				devices.push_back({i, DeviceType::cpu});
		}*/

  } catch(...) {
    ABORT("[TenTrans][Error] Problem parsing devices");
  }

  return devices;
}

}
