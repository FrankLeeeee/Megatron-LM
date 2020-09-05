gpus_per_node=${1:-"8"}
gpus=${2:-"0,1,2,3,4,5,6,7"}
gpu_type=${3:-"2080Ti"}

for bs in 4 16 32 64
do
    for mid_dim in 128 256 512 1024 2048
    do
        for split_ratio in 0 0.25 0.5 0.75
        do
            bash ./nvprof_pipeline.sh $gpus_per_node $gpus $gpu_type $bs $mid_dim $split_ratio
        done
    done
done