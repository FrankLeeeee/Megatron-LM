#!/bin/bash -l

#SBATCH --job-name=summa_pytorch
#SBATCH --time=00:5:00
#SBATCH --nodes=64
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=gpu

method=${1:-"summa"}
bs=${2:-"16"}
row=${3:-"64"}
dim=${4:-"128"}
outpath=${5:-"./profiling/default"}

module load daint-gpu
source /users/youyang9/miniconda3/bin/activate summa
module load Horovod
module load PyTorch

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Environment variables needed by the NCCL backend
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=ipogif0
export NCCL_IB_CUDA_SUPPORT=1

srun -C gpu python main.py --method $method --batch_size $bs --input_row $row --hidden_dim $dim --output_path $outpath