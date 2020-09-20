import torch
import torch.multiprocessing as mp
import os


def run(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    torch.cuda.set_device(rank)
    # global group
    torch.distributed.init_process_group(backend='nccl', 
                                        rank=rank, 
                                        world_size=world_size)
                                        
    # torch.distributed.init_process_group(backend='gloo', init_method='env://')
    g0 = torch.distributed.new_group(ranks=[0,1])
    g1 = torch.distributed.new_group(ranks=[2,3])
    # tensor to bcast over group
    t = torch.tensor([1]).float().cuda().fill_(rank)
    if rank < 2:
        torch.distributed.all_reduce(t, group=g0)
    else:
        torch.distributed.all_reduce(t, group=g1)
    print('rank: {} - val: {}'.format(rank, t.item()))


def main():
    world_size = 4
    mp.spawn(run,
        args=(world_size,),
        nprocs=world_size,
        join=True)

if __name__=="__main__":
    main()
