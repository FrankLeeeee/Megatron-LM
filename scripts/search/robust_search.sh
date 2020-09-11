
gpus_per_node=${1:-"8"}
gpus=${2:-"0,1,2,3,4,5,6,7"}


export CUDA_VISIBLE_DEVICES=$gpus
GPUS_PER_NODE=$gpus_per_node
MASTER_ADDR=localhost
MASTER_PORT=6002
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS robust_search.py