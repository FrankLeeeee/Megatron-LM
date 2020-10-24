import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parameter import Parameter
# import torch.autograd.profiler as profiler

from pytorch_memlab import MemReporter
from helper import get_system_info

import os
import re
import argparse
import subprocess
import time
import sys
from io import StringIO



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

#         print("SUMMA-MLP initialized on rank: {}".format(rank))
    
#     @pytorch_memlab.profile
    def forward(self, x, row_group, col_group):
        # init final output tensor
        batch_size, input_row, hidden_dim = x.size()
        out = torch.zeros([batch_size, input_row, self.hidden_dim //
                           self.dim_col]).float().cuda()
        output_list = [torch.ones_like(out)] * self.world_size
        out_1 = F.linear(x, self.w1)
        
        for step in range(self.dim_col):
            out_1_temp = out_1.clone()
            w2_temp = self.w2.clone()
        
            dist.broadcast(out_1_temp, 
                        src=self.rank_row * self.dim_col + step, 
                        group=row_group,
                        async_op=False)
#             print("rank:{} get broadcast row from rank:{}".format(
#                 self.rank, self.rank_row * self.dim_col + step))
            
            dist.broadcast(w2_temp, 
                        step * self.dim_col + self.rank_col, 
                        col_group,
                        async_op=False)
#             print("rank:{} get broadcast colum from rank:{}".format(
#                 self.rank, step*self.dim_col+self.rank_col))
        
            out += torch.matmul(out_1_temp, w2_temp)
            dist.barrier()

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
                  output_path,
                  ):
    # set cuda device
    proc_id = int(os.environ['SLURM_PROCID'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)

    # dist params
    rank_row = rank // dim_col
    rank_col = rank % dim_col

    assert dim_row == dim_col
    assert world_size == dim_row * dim_col, 'world size： {}, dim_row: {}, dim_col: {}'.format(world_size, dim_row, dim_col)

    # init default group
    local_rank = int(os.environ['SLURM_LOCALID'])
    node_list = os.environ['SLURM_NODELIST']
    host_ip = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    port = "29500"
    init_method = 'tcp://{}:{}'.format(host_ip, port)

    torch.distributed.init_process_group("nccl", init_method=init_method,
                                         world_size=world_size, rank=rank)
    
    row_groups = []
    col_groups = []
    
    for i in range(dim_row):
        row_ranks = list(range(i * dim_col, (i + 1) * dim_col))
        row_group = dist.new_group(ranks=row_ranks)
        row_groups.append(row_group)
#         print("rank:{} row_group:{}".format(rank, row_ranks))

    for j in range(dim_col):
        col_ranks = list(range(j, world_size, dim_col))
        col_group = dist.new_group(ranks=col_ranks)
        col_groups.append(col_group)
#         print("rank:{} col_group:{}".format(rank, col_ranks))

    # init input tensor
    input_tensor = torch.rand((batch_size, input_row, hidden_dim)).cuda()
    dist.broadcast(input_tensor, 0)
    input_tensor = torch.split(input_tensor, input_row//dim_row, dim=1)
    input_tensor = input_tensor[rank_row]
    dist.barrier()    

    # init MLP layers
    mlp = SummaMLP(hidden_dim=hidden_dim,
                         rank=rank,
                         world_size=world_size,
                         dim_row=dim_row,
                         dim_col=dim_col,
                         rank_row=rank_row,
                         rank_col=rank_col).cuda()
    dist.barrier()
    
    # pre-run
    with torch.no_grad():
        output = mlp(input_tensor, row_groups[rank_row], col_groups[rank_col])
    
    # profile for memory
    reporter = MemReporter(mlp)

    with torch.no_grad():
        # intercept system output into out
        out = StringIO()
        sys.stdout = out
        
        output = mlp(input_tensor, row_groups[rank_row], col_groups[rank_col])
        reporter.report()
        
        # recover system output
        sys.stdout = sys.__stdout__
    
    memory_log = 'summa_world_{}_bs_{}_row_{}_dim_{}_memory.txt'.format(
        world_size, batch_size, input_row, hidden_dim
    )
    memory_log = os.path.join(output_path, memory_log)
    
    with open(memory_log , 'w')as f:
        print(out.getvalue(), file=f)
    
    
    # profiling for speed
    # timing for num_iters iterations
    duration_list = []
    num_iters = 10
    
    for i in range(num_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            output = mlp(input_tensor, row_groups[rank_row], col_groups[rank_col])
        # Waits for everything to finish running
        torch.cuda.synchronize()
        end.record()
        duration_list.append(start.elapsed_time(end))
    
    if rank == 0:
        import matplotlib.pyplot as plt
        
        duration_list = [t for t in duration_list]
        avg_duration = sum(duration_list) / num_iters
        print("Duration/ms: {}".format(avg_duration))
        
        # write duration to file
        duration_file_name = 'summa_world_{}_bs_{}_row_{}_dim_{}_iter_{}_duration.txt'.format(
            world_size, batch_size, input_row, hidden_dim, num_iters
        )
        duration_file_name = os.path.join(output_path, duration_file_name)
        with open(duration_file_name, 'w') as f:
            for d in duration_list:
                f.write("{}\n".format(d))
            f.write("\nAverage: {}\n".format(avg_duration))
        
        # plot line chart
        plt.plot(range(num_iters), duration_list)
        img_name = 'summa_world_{}_bs_{}_row_{}_dim_{}_iter_{}_duration.jpg'.format(
            world_size, batch_size, input_row, hidden_dim, num_iters)
        img_name = os.path.join(output_path, img_name)
        plt.savefig(img_name)
    
    # get system info
    if rank == 0:
        sys_info = get_system_info()
        sys_info = sorted(sys_info.items())
        
        sys_info_file_name = 'summa_world_{}_bs_{}_row_{}_dim_{}_sysinfo.txt'.format(
            world_size, batch_size, input_row, hidden_dim
        )
        sys_info_file_name = os.path.join(output_path, sys_info_file_name)
        
        with open(sys_info_file_name, 'w') as f:
            for k, v in sys_info:
                f.write("{}: {}\n".format(k, v))