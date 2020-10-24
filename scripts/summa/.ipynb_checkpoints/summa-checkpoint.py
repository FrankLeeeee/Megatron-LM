import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import os
import re
import argparse


def run(rank, world_size):
    # params
    batch = 2
    A_row = 4
    A_col = 4
    B_row = 4
    B_col = 4
    D_row = 4
    D_col = 4

    dim_row = 2
    dim_col = 2

    rank_row = rank // dim_row
    rank_col = rank % dim_col

    assert world_size == dim_row*dim_col
    assert dim_row == dim_col

    # init group ranks
    row_ranks = list(range(rank_row*dim_col, (rank_row+1)*dim_col))
    col_ranks = list(range(rank_col, world_size, dim_row))

    # init env var
    os.environ['MASTER_ADDR'] = '192.168.41.6'
    os.environ['MASTER_PORT'] = '29525'

    # set cuda device
    torch.cuda.set_device(rank)

    # init distributed
    torch.distributed.init_process_group(backend='nccl',
                                         rank=rank,
                                         world_size=world_size)
    row_group = torch.distributed.new_group(ranks=row_ranks)
    print("rank:{} row_group:{}".format(rank, row_ranks))
    col_group = torch.distributed.new_group(ranks=col_ranks)
    print("rank:{} col_group:{}".format(rank, col_ranks))

    # ---- SUMMA ----
    # init matrix
    A = torch.rand([batch, A_row, A_col]).float().cuda()
    B = torch.rand([B_row, B_col]).float().cuda()
    D = torch.rand([D_row, D_col]).float().cuda()
    E = torch.zeros([batch, A_row, D_col]).float().cuda()
    dist.broadcast(A, 0)
    dist.broadcast(B, 0)
    dist.broadcast(D, 0)
    torch.distributed.barrier()

    # calculate results
    if rank == 0:
        print("A * B * D is :{}".format(torch.matmul(torch.matmul(A, B), D)))

    # step1: split A along the row
    A = torch.split(A, A_row//dim_row, dim=1)
    A = A[rank_row]

    # step2: split B along the column
    B = torch.split(B, B_col//dim_col, dim=1)
    B = B[rank_col]

    # step3: split D along the row & column
    D = torch.split(D, D_row//dim_row, dim=0)
    D = D[rank_row]
    D = torch.split(D, D_col//dim_col, dim=1)
    D = D[rank_col]

    # step3: calculate the first GEMM: A*B=C
    C = torch.matmul(A, B)

    # step3: split E along the row&colum
    E = torch.split(E, A_row//dim_row, dim=1)
    E = E[rank_row]
    E = torch.split(E, D_col//dim_col, dim=2)
    E = E[rank_col]

    # step4: calculate the second GEMM using summa: C*D
    for step in range(dim_col):
        # broadcast row
        C_temp = C.clone()
        if rank_row == 0:
            dist.broadcast(C_temp, rank_row*dim_col+step, row_group)
        torch.distributed.barrier()

        if rank_row == 1:
            dist.broadcast(C_temp, rank_row*dim_col+step, row_group)
        torch.distributed.barrier()
        print("rank:{} get broadcast row from rank:{}".format(
            rank, rank_row*dim_col+step))

        # broadcast colum
        D_temp = D.clone()

        if rank_col == 0:
            dist.broadcast(D_temp, step*dim_col+rank_col, col_group)
        torch.distributed.barrier()

        if rank_col == 1:
            dist.broadcast(D_temp, step*dim_col+rank_col, col_group)
        torch.distributed.barrier()
        print("rank:{} get broadcast colum from rank:{}".format(
            rank, step*dim_col+rank_col))

        E += torch.matmul(C_temp, D_temp)
        torch.distributed.barrier()

    print("E: {}".format(E))


def main():
    world_size = 4
    os.environ['NCCL_SOCKET_IFNAME'] = 'enp1s0f1'
#    os.environ['NCCL_DEBUG'] = 'INFO'
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
