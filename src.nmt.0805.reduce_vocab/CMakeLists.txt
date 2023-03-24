cmake_minimum_required(VERSION 3.11)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS " -std=c++11 -fPIC -msse4.2 -w -march=native -Wno-enum-compare -Wno-sign-compare -Wno-format -Wno-dev -O3 -DUSE_FP16 -DCUDA_DEBUG -rdynamic ")
## -DCUDA_DEBUG
## -DNDEBUG

# set(EXT_LIBS ${EXT_LIBS} ${CUDA_cublas_LIBRARY})

set(CUDA_NVCC_FLAGS " -O3 -Xcompiler -fPIC -maxrregcount=0 --disable-warnings -use_fast_math -Wno-deprecated-gpu-targets -std=c++11 ")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_60 
	-gencode=arch=compute_60,code=sm_60 
	-gencode=arch=compute_61,code=sm_61 
	-gencode=arch=compute_62,code=sm_62 
	-gencode=arch=compute_70,code=sm_70
	-gencode=arch=compute_70,code=compute_70 
)
# -gencode=arch=compute_75,code=sm_75
# -gencode=arch=compute_80,code=sm_80
# -gencode=arch=compute_80,code=compute_80
list(REMOVE_DUPLICATES CUDA_NVCC_FLAGS)

find_package(ZLIB)
if(ZLIB_FOUND)
  include_directories(${ZLIB_INCLUDE_DIRS})
  set(EXT_LIBS ${EXT_LIBS} ${ZLIB_LIBRARIES})
else(ZLIB_FOUND)
  message(SEND_ERROR "Cannot find zlib.")
endif(ZLIB_FOUND)

find_package(CUDA "8.0")
include_directories(${CUDA_INCLUDE_DIRS})
ADD_DEFINITIONS(-D CUDA_FOUND)
set(EXT_LIBS ${EXT_LIBS} ${CUDA_cublas_LIBRARY})

set(BOOST_INCLUDE "/dockerdata/danielkxwu/envs/boost_1_66_0")
set(BOOST_LIBRARY_PATH "/dockerdata/danielkxwu/envs/boost_1_66_0/lib/")

find_package(Boost)
if(Boost_FOUND)
	include_directories(${Boost_INCLUDE_DIRS})
	set(EXT_LIBS ${EXT_LIBS} ${Boost_LIBRARIES})
	set(EXT_LIBS ${EXT_LIBS} ${ZLIB_LIBRARIES}) # hack for static compilation
else(Boost_FOUND)
	message(SEND_ERROR "Cannot find Boost libraries. Terminating.")
endif(Boost_FOUND)

set(FASTER_TENSOR_INCLUDE "/dockerdata/danielkxwu/Decoder/src.nmt/nvidia_export/")

include_directories(.)
include_directories(FASTER_TENSOR_INCLUDE)
include_directories(BOOST_INCLUDE)
include_directories(./3rd_party/yaml-cpp/include/)
include_directories(./3rd_party/cnpy/)
include_directories(./3rd_party/spdlog/include/)

include_directories(./config/)
include_directories(./data/)
include_directories(./device/)
include_directories(./global/)
include_directories(./layer/)
include_directories(./utils/)
include_directories(./mem/)
include_directories(./search/)
# include_directories(./cuda/)
include_directories(./tensor/cuda/)
include_directories(./tensor/)
include_directories(./task/)

include_directories(./cub/)
include_directories(./cub/block/)
include_directories(./cub/device/)
include_directories(./cub/grid/)
include_directories(./cub/warp/)
include_directories(./cub/iterator/)

include_directories(./api/)

file(GLOB CURRENT_SOURCES *)
foreach(file IN LISTS CURRENT_SOURCES)
	message(STATUS ${file})
endforeach(file)


# Find all the .cpp .h .cu .chu files in source folder
# file(GLOB_RECURSE CPP_FILES source/*.cpp)
# file(GLOB_RECURSE H_FILES source/*.h)
# file(GLOB_RECURSE CU_FILES source/*.cu)
# file(GLOB_RECURSE CUH_FILES source/*.cuh)

# set(CUDA_TOOLKIT_ROOT_DIR "/dockerdata/danielkxwu/envs/cuda-9.2")
# link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
# include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
# set(CUDA_LIB_DIR "${CUDA_TOOLKIT_ROOT_DIR}/lib64/")
# set(CUDA_LIB_PATH ${CUDA_LIB_PATH} "${CUDA_LIB_DIR}libcublas_static.a")
# set(CUDA_LIB_PATH ${CUDA_LIB_PATH} "${CUDA_LIB_DIR}libculibos.a")
# set(CUDA_LIB_PATH ${CUDA_LIB_PATH} "${CUDA_LIB_DIR}libnpps_static.a")
# set(CUDA_LIB_PATH ${CUDA_LIB_PATH} "${CUDA_LIB_DIR}libnppc_static.a")
# set(CUDA_LIB_PATH ${CUDA_LIB_PATH} "${CUDA_LIB_DIR}libcudadevrt.a")
# set(CUDA_LIB_PATH ${CUDA_LIB_PATH} "${CUDA_LIB_DIR}libcurand_static.a")
# set(CUDA_LIB_PATH ${CUDA_LIB_PATH} "${CUDA_LIB_DIR}libcudart_static.a")
# set(CUDA_LIB_PATH ${CUDA_LIB_PATH} "/usr/lib64/libdl.so.2")

message("Building TenTrans, welcome to the world of nmt translator.")
add_library(TenTrans STATIC
    ./api/export_api.cpp
    ./api/online_decoder.cpp 
    ./task/Classifier.cpp 
    ./layer/HUBaseLayer.cpp
	./layer/HUEncoderDecoder.cpp 
	./layer/HUEncoder.cpp
	./layer/HUEncoderLayer.cpp
    ./layer/HUDecoder.cpp
    ./layer/HUDecoderLayer.cpp
	./layer/HUFFNLayer.cpp
	./layer/HULayerNorm.cpp
	./layer/HUEmbeddingLayer.cpp
	./layer/HUMultiHeadAttention.cpp 
	./layer/HUOutputLayer.cpp
    ./search/HUBeamSearch.cpp
	./utils/Logging.cpp
	./config/HUConfig.cpp
	./config/HUConfigParser.cpp
	./utils/HUUtil.cpp
	./data/HUData.cpp
	./data/HUVocab.cpp
	./tensor/HUTensor.cpp
	./tensor/HUTensorUtil.cpp
    ./search/HUResult.cpp
	./3rd_party/cnpy/cnpy.cpp)

cuda_add_library(TenTrans_cuda 
	./device/HUDevice.cu 
	./utils/HUNthElement.cu
    ./tensor/HUTensor.cu
	./tensor/HUTensorOP.cu 
    ./tensor/cuda/masked_multihead_attention.cu
    STATIC)
# ./tensor/cuda/online_softmax_beamsearch_kernels.cu
# ./tensor/cuda/cuda_kernels.cu
link_directories("../lib")

target_link_libraries(TenTrans_cuda TenTrans)
add_executable(TenTrans_test main.cpp)
set_target_properties(TenTrans_test PROPERTIES OUTPUT_NAME TenTrans_test)

## add_executable(TenTrans_xlm main_xlm.cpp)
## set_target_properties(TenTrans_xlm PROPERTIES OUTPUT_NAME TenTrans_xlm)

target_link_libraries(TenTrans_test TenTrans TenTrans_cuda libboost_filesystem.a libboost_program_options.a libboost_regex.a libboost_system.a libyaml-cpp.a ${EXT_LIBS} ${CUDA_LIB_PATH} )
#### cuda_add_cublas_to_target(TenTrans_test)

## target_link_libraries(TenTrans_xlm TenTrans TenTrans_cuda libboost_filesystem.a libboost_program_options.a libboost_regex.a libboost_system.a libyaml-cpp.a ${EXT_LIBS} ${CUDA_LIB_PATH} )
## cuda_add_cublas_to_target(TenTrans_xlm)

message("[TenTrans]: You are lucky, makefile generation completed !")
