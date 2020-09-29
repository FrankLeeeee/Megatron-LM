import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from summa_mlp_cpu import summa_mlp_run
from megatron_mlp_cpu import megatron_mlp_run
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str,
                        choices=['summa', 'megatron'], required=True)
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--input_row", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dim_row", type=int, default=2)
    parser.add_argument("--dim_col", type=int, default=2)
    args = parser.parse_args()
    return args


def main():
    # get args
    args = parse_args()
    parallel_method = args.__dict__.pop('method')

    if parallel_method == 'summa':
        run = summa_mlp_run
    elif parallel_method == 'megatron':
        run = megatron_mlp_run

    # init env var
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '23456'
    os.environ['GLOO_SOCKET_IFNAME'] = "lo"

    mp.spawn(run,
             args=(
                 args.world_size,
                 args.batch_size,
                 args.input_row,
                 args.hidden_dim,
                 args.dim_row,
                 args.dim_col),
             nprocs=args.world_size,
             join=True)


if __name__ == "__main__":
    main()
