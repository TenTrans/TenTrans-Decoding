# WMT2021 Efficiency Task

### 1. Data
Our teacher and studentmodels are trained WMT21 En-De news data, and the development set is En-De newstest2019.

### 2. Model Configurations

![alt text](https://github.com/TenTrans/TenTrans-Decoding/blob/master/examples/model_conf.png?raw=true)



### 3. Performance

![alt text](https://github.com/TenTrans/TenTrans-Decoding/blob/master/examples/performance.png?raw=true)



### 4. Usage 
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
