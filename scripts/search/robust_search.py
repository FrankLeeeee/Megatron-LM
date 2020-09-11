import torch
import torch.nn.functional as F
import time
import pandas as pd


def init_dist():
    import os
    import re
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    args = parser.parse_args()

    torch.distributed.init_process_group("nccl", init_method='env://')
    torch.cuda.set_device(args.local_rank)
    return args.local_rank


def no_overlap(inputs, weight, warmup_round=10, test_round=20):
    # warm up
    for _ in range(warmup_round):
        output = F.linear(inputs, weight)
        torch.distributed.all_reduce(output)
    # sync the whole system
    torch.distributed.all_reduce(output)
    
    start = time.time()
    for _ in range(test_round):
        output = F.linear(inputs, weight)
        torch.distributed.all_reduce(output)
    torch.cuda.synchronize()
    return 1000*(time.time()-start)/test_round
    
def overlap(micro_batch, weight, warmup_round=10, test_round=20):
    # warm up
    for _ in range(warmup_round):
        output1 = F.linear(micro_batch[0], weight)
        handle1 = torch.distributed.all_reduce(output1, async_op=True)
        output2 = F.linear(micro_batch[1], weight)
        torch.distributed.all_reduce(output2)
        handle1.wait()
    # sync the whole system
    torch.distributed.all_reduce(output2)
    
    start = time.time()
    for _ in range(test_round):
        output1 = F.linear(micro_batch[0], weight)
        handle1 = torch.distributed.all_reduce(output1, async_op=True)
        output2 = F.linear(micro_batch[1], weight)
        torch.distributed.all_reduce(output2)
        handle1.wait()
    torch.cuda.synchronize()
    return 1000*(time.time()-start)/test_round


def generate_data(batch, mid_dim, micro_batch_first_batch=None):
    weight_shape = [1024, mid_dim] 
    input_shape = [batch, 512, mid_dim]
    weight = torch.rand(weight_shape).cuda().half()
    inputs = torch.rand(input_shape).cuda().half()
    if micro_batch_first_batch is not None:
        inputs = torch.split(inputs, (micro_batch_first_batch, batch-micro_batch_first_batch), dim=0)
    return inputs, weight
        
    
if __name__ == "__main__":

    rank = init_dist()
    mid_dims = [128, 256, 512, 1024, 2048]
    batch_sizes = [4,8, 16, 32, 64]

    if rank == 0:
        pd_data = {
            'mid_dim': [], 
            'batch_size': [], 
            'baseline': [],
            'minimal_batch': [], 
            'minimal_time':[],
            'improvement': []
            }

    for mid_dim in mid_dims:
        for batch_size in batch_sizes:
            # get baseline
            inputs, weight = generate_data(batch_size, mid_dim)
            baseline = no_overlap(inputs, weight)
            if rank==0:
                print("baseline (dim:{} batch:{}): {}ms".format(mid_dim, batch_size, baseline))
                pd_data['mid_dim'].append(mid_dim)
                pd_data['batch_size'].append(batch_size)
                pd_data['baseline'].append(baseline)

            # get overlap for different micro batches
            minimal_time = None
            minimal_batch = None
            for first_batch in range(1, batch_size):
                inputs, weight = generate_data(batch_size, mid_dim, first_batch)
                overlap_time = overlap(inputs, weight)
                if rank==0:
                    print("overlap (first batch:{}): {}ms".format(first_batch, overlap_time))
                    if minimal_time is None or overlap_time < minimal_time:
                        minimal_time = overlap_time
                        minimal_batch = first_batch
            
            if rank == 0:
                pd_data['minimal_batch'].append(minimal_batch)
                pd_data['minimal_time'].append(minimal_time)
                
            if rank==0:
                print("minimal: batch: {} time: {}ms".format(minimal_batch, minimal_time))
                improvement = 100*(baseline-minimal_time)/baseline
                pd_data['improvement'].append(improvement)

                if minimal_time < baseline:
                    print("find solution: {}".format(improvement))
    
    if rank == 0:
        pd = pd.DataFrame.from_dict(pd_data)
        pd.to_csv("search_results.csv")
                    
