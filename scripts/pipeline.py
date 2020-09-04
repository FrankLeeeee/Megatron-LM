import torch
import torch.nn.functional as F
if __name__ == "__main__":
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

    mid_dim = 128
    batch = 64
    weight_shape = [1024, mid_dim] 
    input_shape = [batch, 512, mid_dim]
    weight = torch.rand(weight_shape).cuda().half()
    inputs = torch.rand(input_shape).cuda().half()

    
    import time
    total_time = 0.0
    for _ in range(10):
        torch.cuda.synchronize()
        output = F.linear(inputs, weight)
        torch.distributed.all_reduce(output)
    torch.distributed.all_reduce(output)
    
    total_time = 0.0
    with torch.cuda.profiler.profile():
        with torch.autograd.profiler.emit_nvtx():
            micro_batch = torch.split(inputs, (32, 32), dim=0)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                '''
                output = F.linear(inputs, weight)
                torch.distributed.all_reduce(output)
                '''
                output1 = F.linear(micro_batch[0], weight)
                handle1 = torch.distributed.all_reduce(output1, async_op=True)
                output2 = F.linear(micro_batch[1], weight)
                handle2 = torch.distributed.all_reduce(output2, async_op=True)
                handle1.wait()
                handle2.wait()
            torch.cuda.synchronize()
            print("pipeline: {}ms".format((time.time()-start)*1000))
