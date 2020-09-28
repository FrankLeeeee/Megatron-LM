import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parameter import Parameter

import os
import re
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--input_row", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dim_row", type=int, default=2)
    parser.add_argument("--dim_col", type=int, default=2)
    args = parser.parse_args()
    return args


class SummaMLP(nn.Module):
    """SUMMA-based MLP
    """

    def __init__(self,
                 hidden_dim,
                 rank,
                 world_size,
                 dim_row,
                 dim_col,
                 rank_row,
                 rank_col):
        self.hidden_dim = hidden_dim

        # init dist params
        self.rank = rank
        self.world_size = world_size
        self.dim_row = dim_row
        self.dim_col = dim_col
        self.rank_row = rank_row
        self.rank_col = rank_col

        # init linear layers
        self.w1 = Parameter(torch.Tensor(self.hidden_dim * 4 // self.dim_col,
                                         self.hidden_dim))

        self.w2 = Parameter(torch.Tensor(self.hidden_dim // self.dim_col,
                                         self.hidden_dim * 4 // self.dim_row))

        print("SUMMA-MLP initialized on rank: {}".format(rank))

    def forward(self, x, row_group, col_group):
        # init final output tensor
        batch, dim = x.size()
        out = torch.zeros([batch, dim, self.hidden_dim //
                           self.dim_col]).float().cuda()

        out_1 = F.linear(x, self.w1)

        for step in range(dim_col):
            # TODO
            # fix broadcast sequence
            # broadcast row
            out_1_temp = out_1.clone()
            if rank_row == 0:
                dist.broadcast(out_1_temp, self.rank_row *
                               self.dim_col + step, row_group)
            torch.distributed.barrier()

            if rank_row == 1:
                dist.broadcast(out_1_temp, self.rank_row *
                               self.dim_col + step, row_group)
            torch.distributed.barrier()
            print("rank:{} get broadcast row from rank:{}".format(
                self.rank, self.rank_row * self.dim_col + step))

            # broadcast colum
            w2_temp = self.w2.clone()

            if rank_col == 0:
                dist.broadcast(w2_temp, step * self.dim_col +
                               self.rank_col, col_group)
            torch.distributed.barrier()

            if rank_col == 1:
                dist.broadcast(w2_temp, step*self.dim_col +
                               self.rank_col, col_group)
            torch.distributed.barrier()
            print("rank:{} get broadcast colum from rank:{}".format(
                rank, step*dim_col+rank_col))

            out += torch.matmul(out_1_temp, w2_temp)
            torch.distributed.barrier()

        return out


def run(rank,
        world_size,
        batch_size,
        input_row,
        hidden_dim,
        dim_row,
        dim_col,
        ):

    # init env var
    os.environ['MASTER_ADDR'] = '192.168.41.6'
    os.environ['MASTER_PORT'] = '29525'

    # set cuda device
    torch.cuda.set_device(rank)

    # dist params
    rank_row = rank // dim_col
    rank_col = rank % dim_col

    assert world_size == dim_row * dim_col

    # init group ranks
    row_ranks = list(range(rank_row * dim_col, (rank_row + 1) * dim_col))
    col_ranks = list(range(rank_col, world_size, dim_col))
    print("rank: {}, rank_row: {}, row_ranks: {}".format(
        rank, rank_row, row_ranks))
    print("rank: {}, rank_col: {}, col_ranks: {}".format(
        rank, rank_col, col_ranks))

    # init groups
    torch.distributed.init_process_group(backend='nccl',
                                         rank=rank,
                                         world_size=world_size)
    row_group = torch.distributed.new_group(ranks=row_ranks)
    print("rank:{} row_group:{}".format(rank, row_ranks))
    col_group = torch.distributed.new_group(ranks=col_ranks)
    print("rank:{} col_group:{}".format(rank, col_ranks))

    # init input tensor
    input_tensor = torch.rand((batch_size, input_row, hidden_dim)).cuda()
    input_tensor = torch.split(input_tensor, input_row//dim_row, dim=1)
    input_tensor = input_tensor[rank_row]
    print(input_tensor.shape)

    # init MLP layers
    mlp_layer = SummaMLP(hidden_dim=hidden_dim,
                         rank=rank,
                         world_size=world_size,
                         dim_row=dim_row,
                         dim_col=dim_col,
                         rank_row=rank_row,
                         rank_col=rank_col).cuda()

    torch.cuda.cudart().cudaProfilerStart()
    output = mlp_layer(input_tensor, row_group, col_group)
    torch.cuda.cudart().cudaProfilerStop()


def main():
    args = parse_args()
    os.environ['NCCL_SOCKET_IFNAME'] = 'enp1s0f1'
#    os.environ['NCCL_DEBUG'] = 'INFO'
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
