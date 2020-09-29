import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import os
import re
import argparse


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
        super().__init__()
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

        self.w2 = Parameter(torch.Tensor(self.hidden_dim * 4 // self.dim_row,
                                         self.hidden_dim // self.dim_col,))

        print("SUMMA-MLP initialized on rank: {}".format(rank))

    def forward(self, x, row_group, col_group):
        # init final output tensor
        batch_size, input_row, hidden_dim = x.size()
        out = torch.zeros([batch_size, input_row, self.hidden_dim //
                           self.dim_col]).float()

        out_1 = F.linear(x, self.w1)

        for step in range(self.dim_col):
            # broadcast row
            out_1_temp = out_1.clone()

            for i in range(self.dim_row):
                if self.rank_row == i:
                    dist.broadcast(out_1_temp, self.rank_row *
                                   self.dim_col + step, row_group)
                    print("rank:{} get broadcast row from rank:{}".format(
                        self.rank, self.rank_row * self.dim_col + step))
                dist.barrier()

            # broadcast colum
            w2_temp = self.w2.clone()

            for j in range(self.dim_col):
                if self.rank_col == j:
                    dist.broadcast(w2_temp, step * self.dim_col +
                                   self.rank_col, col_group)
                    dist.barrier()
                    print("rank:{} get broadcast colum from rank:{}".format(
                        self.rank, step*self.dim_col+self.rank_col))
                dist.barrier()

            out += torch.matmul(out_1_temp, w2_temp)
            dist.barrier()

        output_list = [torch.ones_like(out)] * self.world_size
        dist.all_gather(output_list, out)

        output = torch.cat(output_list, dim=2)
        output = output.view(batch_size, input_row *
                             self.dim_row, hidden_dim)
        return output


def summa_mlp_run(rank,
                  world_size,
                  batch_size,
                  input_row,
                  hidden_dim,
                  dim_row,
                  dim_col,
                  ):
    # set cuda device
    # torch.cuda.set_device(rank)

    # dist params
    rank_row = rank // dim_col
    rank_col = rank % dim_col

    assert dim_row == dim_col
    assert world_size == dim_row * dim_col

    # init group ranks
    row_ranks = list(range(rank_row * dim_col, (rank_row + 1) * dim_col))
    col_ranks = list(range(rank_col, world_size, dim_col))
    print("rank: {}, rank_row: {}, row_ranks: {}".format(
        rank, rank_row, row_ranks))
    print("rank: {}, rank_col: {}, col_ranks: {}".format(
        rank, rank_col, col_ranks))

    # init groups
    dist.init_process_group(backend='gloo',
                            rank=rank,
                            world_size=world_size)
    for i in range(dim_row):
        if rank_row == i:
            row_group = dist.new_group(ranks=row_ranks)
            print("rank:{} row_group:{}".format(rank, row_ranks))
        dist.barrier()

    for j in range(dim_col):
        if rank_col == j:
            col_group = dist.new_group(ranks=col_ranks)
            print("rank:{} col_group:{}".format(rank, col_ranks))
        dist.barrier()

    # init input tensor
    input_tensor = torch.rand((batch_size, input_row, hidden_dim))
    input_tensor = torch.split(input_tensor, input_row//dim_row, dim=1)
    input_tensor = input_tensor[rank_row]
    dist.barrier()

    # init MLP layers
    mlp_layer = SummaMLP(hidden_dim=hidden_dim,
                         rank=rank,
                         world_size=world_size,
                         dim_row=dim_row,
                         dim_col=dim_col,
                         rank_row=rank_row,
                         rank_col=rank_col)
    dist.barrier()

    with torch.autograd.profiler.profile() as prof:
        output = mlp_layer(input_tensor, row_group, col_group)

    if rank == 0:
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
