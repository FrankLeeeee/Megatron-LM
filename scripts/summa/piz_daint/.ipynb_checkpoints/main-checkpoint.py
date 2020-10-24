import torch
import torch.multiprocessing as mp
import torch.distributed as dist


from summa_mlp import summa_mlp_run
from megatron_mlp import megatron_mlp_run
import argparse
import os
import math


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str,
                        choices=['summa', 'megatron'], required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--input_row", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--output_path", type=str, default='./profiling/default')
    args = parser.parse_args()
    return args


def main():
    os.environ['NCCL_DEBUG'] = 'INFO'
    # get args
    args = parse_args()
    args_dict = args.__dict__
    parallel_method = args_dict.pop('method')

    if parallel_method == 'summa':
        run = summa_mlp_run
    elif parallel_method == 'megatron':
        run = megatron_mlp_run
    
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    
    world_size = int(os.environ['SLURM_NTASKS'])
    args.rank = int(os.environ['SLURM_PROCID'])
    args.world_size = world_size
    args.dim_row = int(math.sqrt(world_size))
    args.dim_col = int(math.sqrt(world_size))
    
    run(**args_dict)


if __name__ == "__main__":
    main()
