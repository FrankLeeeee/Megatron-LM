import torch
import torch.multiprocessing as mp
import os


def run(rank, world_size):
    # init env var
    os.environ['MASTER_ADDR'] = '192.168.41.6'
    os.environ['MASTER_PORT'] = '29500'
    rank_row = rank // 2
    rank_col = rank % 2

    # init group ranks
    g0_ranks = list(range(rank_row*2, (rank_row+1)*2))
    g1_ranks = list(range(rank_col, world_size, 2))

    # set cuda device
    torch.cuda.set_device(rank)

    # init groups
    torch.distributed.init_process_group(backend='nccl',
                                         rank=rank,
                                         world_size=world_size)
    g0 = torch.distributed.new_group(ranks=g0_ranks)
    g1 = torch.distributed.new_group(ranks=g1_ranks)

    # tensor to bcast over group
    t1 = torch.rand(1).float().cuda()
    t2 = torch.rand(1).float().cuda()
    print("rank_row: {}, rank_col: {}, g0_ranks: {}, g1_ranks: {},  {}".format(rank_row, rank_col, g0_ranks, g1_ranks,  t1))

    torch.distributed.broadcast(t1, rank_row*2, group=g0)
    print('rank: {} - src: {}: row broadcast'.format(rank, rank_row*2))
    torch.distributed.barrier()

    if rank_col == 0:
        torch.distributed.broadcast(t2, rank_col, group=g1)
    torch.distributed.barrier()
    print('rank: {} - src: {}: col broadcast'.format(rank, rank_col))

    if rank_col == 1:
        torch.distributed.broadcast(t2, rank_col, group=g1)
    torch.distributed.barrier()



def main():
    world_size = 4
    os.environ['NCCL_SOCKET_IFNAME'] = 'enp1s0f1'
    os.environ['NCCL_DEBUG'] = 'INFO'
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
