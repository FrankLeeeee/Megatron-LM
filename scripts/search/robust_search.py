import torch
import torch.nn.functional as F
import time


def init_dist():
    import os
    import re
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    local_rank = int(os.environ['SLURM_LOCALID'])

    node_list = str(os.environ['SLURM_NODELIST'])
    node_parts = re.findall('[0-9]+', node_list)
    host_ip = '{}.{}.{}.{}'.format(node_parts[1], node_parts[2], node_parts[3], node_parts[4])
    port = "23456"
    init_method = 'tcp://{}:{}'.format(host_ip, port)

    torch.distributed.init_process_group("nccl", init_method=init_method,
                                         world_size=world_size, rank=rank)
    torch.cuda.set_device(local_rank)
    return rank


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

    for mid_dim in mid_dims:
        for batch_size in batch_sizes:
            # get baseline
            inputs, weight = generate_data(batch_size, mid_dim)
            baseline = no_overlap(inputs, weight)
            if rank==0:
                print("baseline (dim:{} batch:{}): {}ms".format(mid_dim, batch_size, baseline))
            # get overlap for different micro batches
            minimial_time = None
            minimial_batch = None
            for first_batch in range(1, batch_size):
                inputs, weight = generate_data(batch_size, mid_dim, first_batch)
                overlap_time = overlap(inputs, weight)
                if rank==0:
                    print("overlap (first batch:{}): {}ms".format(first_batch, overlap_time))
                    if minimial_time is None or overlap_time < minimial_time:
                        minimial_time = overlap_time
                        minimial_batch = first_batch
            if rank==0:
                print("minimial: batch: {} time: {}ms".format(minimial_batch, minimial_time))
                if minimial_time < baseline:
                    print("find solution: {}".format(100*(baseline-minimial_time)/baseline))
