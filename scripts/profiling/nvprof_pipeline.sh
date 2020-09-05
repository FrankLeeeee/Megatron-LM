#!/bin/bash

gpus_per_node=${1:-"8"}
gpus=${2:-"0,1,2,3,4,5,6,7"}
gpu_type=${3:-"2080Ti"}
batch_size=${4:-"16"}
mid_dim=${5:-"128"}
split_ratio=${6:-"0"}

pipeline=""

if [ "$split_ratio" = "0" ] ; then
    echo "No pipeline"
    pipeline=""
else
    echo "Pipeline enabled"
    pipeline="--pipeline"
fi

# set output path
output_path="./nvprof_outputs/${gpu_type}/gpu_${gpus_per_node}/bs_${batch_size}_mid-dim_${mid_dim}_split_${split_ratio}"
mkdir -p $output_path

# set GPUS
export CUDA_VISIBLE_DEVICES=$gpus
export PATH=/usr/local/cuda-10.2/bin/:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-10.2/lib64

GPUS_PER_NODE=$gpus_per_node
MASTER_ADDR=localhost
MASTER_PORT=6002
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

/usr/local/cuda/bin/nvprof --profile-from-start off --profile-child-processes \
	-f -o ${output_path}/bert_trace_%p.prof \
	python -m torch.distributed.launch $DISTRIBUTED_ARGS pipeline.py \
	--batch_size $batch_size \
	--mid_dim $mid_dim \
	--split_ratio $split_ratio $pipeline
