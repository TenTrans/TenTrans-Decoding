# WMT2021 Efficiency Task

### 1. DataSet
Our teacher and studentmodels are trained constrained of WMT21 En-De news data（http://statmt.org/wmt21/translation-task.html）, and the development set is En-De newstest2019.

- bilingual data (En-De): Europarl v10, ParaCrawl v7.1, News Commentary, Wiki Titles v3, Tilde Rapid corpus and WikiMatrix
- monolingual data (De): NewsCrawl2020, Europarl v10, News Commentary


### 2. Model Configurations
- Teacher-base-20_6 (2xFFN): 20 encooder layes + 6 decoder layers + 8heads + 512 hidden size + 4096 ffn size;
- Student-base-20_1: 20 encooder layes + 1 decoder layers + 8heads + 512 hidden size + 2048 ffn size;
- Student-base-10_1: 10 encooder layes + 1 decoder layers + 8heads + 512 hidden size + 2048 ffn size;
- Student-tiny-20_1: 20 encooder layes + 1 decoder layers + 8heads + 256 hidden size + 1024 ffn size;
All models tie the source embedding, the target embedding, and the softmax weights. 

![alt text](https://github.com/TenTrans/TenTrans-Decoding/blob/master/examples/model_conf.png?raw=true)


### 3. Inference Optimizations
- Attention Caching: Cross-attention Caching + Self-attention Caching of decoder layer.
- Kernel Fusion
![alt text](https://github.com/TenTrans/TenTrans-Decoding/blob/master/examples/kernel_fusion.png?raw=true)

- Early-stop
- Sorted & Greedy Search


### 4. Performance of different models

![alt text](https://github.com/TenTrans/TenTrans-Decoding/blob/master/examples/performance.png?raw=true)



### 5. Usage 
We relase all models and corresponding docker images, you can follow the instructions below to run our models.

#### Docker Images
- **Teacher-base-20_6(2xFFN)**: danielkxwu/wmt2021_tentrans_transformer-teacher-enc20dec6-h512-ffn4096_gpu_throughput_cuda11
- **Student-base-20_1**: danielkxwu/wmt2021_tentrans_transformer-student-enc20dec1-h512-ffn2048_gpu_throughput_cuda11
- **Student-base-10_1**: danielkxwu/wmt2021_tentrans_transformer-student-enc10dec1-h512-ffn2048_gpu_throughput_cuda11
- **Student-tiny-20_1**: danielkxwu/wmt2021_tentrans_transformer-teacher-enc20dec6-h512-ffn4096_gpu_throughput_cuda11

#### Run Docker

```shell
# the input file must be raw text
infile_name=newstest2019-ende.en
outfile_name=newstest2019-ende.en.trans
container_name=translator

image_from_docker_hub=danielkxwu/wmt2021_tentrans_transformer-student-enc20dec1-h256-ffn1024_gpu_throughput_cuda11
# image_from_docker_hub=danielkxwu/wmt2021_tentrans_transformer-student-enc20dec1-h512-ffn2048_gpu_throughput_cuda11
# image_from_docker_hub=danielkxwu/wmt2021_tentrans_transformer-student-enc10dec1-h512-ffn2048_gpu_throughput_cuda11
# image_from_docker_hub=danielkxwu/wmt2021_tentrans_transformer-teacher-enc20dec6-h512-ffn4096_gpu_throughput_cuda11

# step1: pull docker image
docker pull ${image_from_docker_hub}
# step2: run docker image enviroment
docker run --runtime=nvidia --name=${container_name} -itd ${image_from_docker_hub}
# step3: copy local input data to docker
docker cp ${infile_name} ${container_name}:/scripts/${infile_name}
# step4: run translation pipeline in docker
docker exec ${container_name} /run.sh GPU throughput /scripts/${infile_name} /scripts/${infile_name}.trans >run.stdout 2>&1
# step5: copy the translation data from docker to local
docker cp ${container_name}:/scripts/${infile_name}.trans ${outfile_name}
