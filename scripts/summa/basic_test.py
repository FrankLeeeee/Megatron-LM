import torch
import torch.multiprocessing as mp
import os


def run(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29502'
    # global group
    torch.distributed.init_process_group(backend='nccl',
                                        # init_method="tcp://localhost:29502",
                                         rank=rank,
                                         world_size=world_size)
    torch.cuda.set_device(rank)
    # torch.distributed.init_process_group(backend='gloo', init_method='env://')
    rank_row = rank // 2
    rank_col = rank % 2

    g0_ranks = list(range(rank_row*2, (rank_row+1)*2))
    g1_ranks = list(range(rank_col, world_size, 2))

    g0 = torch.distributed.new_group(ranks=g0_ranks)
    g1 = torch.distributed.new_group(ranks=g1_ranks)

    # tensor to bcast over group
    t1 = torch.rand(2, 2).float().cuda()
    t2 = torch.rand(2, 2).float().cuda()

    torch.distributed.broadcast(t1, rank_row*2, group=g0)
    print('rank: {} - src: {}: row broadcast'.format(rank, rank_row*2))

    # if rank_col == 0:
    #     torch.distributed.broadcast(t2, rank_col, group=g1)
    # torch.distributed.barrier()

    # if rank_col == 1:
    #     torch.distributed.broadcast(t2, rank_col, group=g1)
    # torch.distributed.barrier()

    # print('rank: {} - src: {}: col broadcast'.format(rank, rank_col))


def main():
    world_size = 4
    os.environ['NCCL_DEBUG'] = 'INFO'
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
