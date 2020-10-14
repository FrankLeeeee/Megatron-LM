import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.autograd.profiler as profiler

import os
import subprocess
import time


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(
        numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


class ColumnParallelLinear(nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gether on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """

    def __init__(self,
                 input_size,
                 output_size,
                 bias=True,
                 gather_output=True,
                 #  init_method=init.xavier_normal_,
                 stride=1,
                 world_size=8,
                 keep_master_weight_for_test=False):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        self.output_size_per_partition = divide(output_size, world_size)

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.weight = Parameter(torch.Tensor(self.output_size_per_partition,
                                             self.input_size))
        if bias:
            self.bias = Parameter(torch.Tensor(self.output_size_per_partition))
            self.bias.model_parallel = True
            self.bias.partition_dim = 0
            self.bias.stride = stride
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

        # Initialize weight.
        # self.master_weight = _initialize_affine_weight(
        #     self.weight, self.output_size, self.input_size,
        #     self.output_size_per_partition, 0, init_method,
        #     stride=stride, return_master_weight=keep_master_weight_for_test)

    def forward(self, input_):
        # Set up backprop all-reduce.
        # input_parallel = copy_to_model_parallel_region(input_)
        # input_parallel = input_

        # Matrix multiply.
        output_parallel = F.linear(input_, self.weight, self.bias)
        return output_parallel


class RowParallelLinear(nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """

    def __init__(self,
                 input_size,
                 output_size, bias=True,
                 input_is_parallel=False,
                 #  init_method=init.xavier_normal_,
                 stride=1,
                 world_size=8,
                 keep_master_weight_for_test=False):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        self.input_size_per_partition = divide(input_size, world_size)

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.weight = Parameter(torch.Tensor(self.output_size,
                                             self.input_size_per_partition))
        if bias:
            self.bias = Parameter(torch.Tensor(self.output_size))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

        # Initialize weight.
        # self.master_weight = _initialize_affine_weight(
        #     self.weight, self.output_size, self.input_size,
        #     self.input_size_per_partition, 1, init_method,
        #     stride=stride, return_master_weight=keep_master_weight_for_test)

    def forward(self, input_):
        # Set up backprop all-reduce.
        # input_parallel = input_

        # Matrix multiply.
        output_ = F.linear(input_, self.weight)
        # All-reduce across all the partitions.
        torch.distributed.all_reduce(output_)

        if self.bias is not None:
            output = output_ + self.bias
        else:
            output = output_
        return output


class ParallelMLP(nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    """

    def __init__(self,
                 hidden_size,
                 world_size,
                 #  mlp_activation_func,
                 #  init_method,
                 #  output_layer_init_method,
                 ):
        super(ParallelMLP, self).__init__()
        self.hidden_size = hidden_size

        # Project to 4h.
        self.dense_h_to_4h = ColumnParallelLinear(
            self.hidden_size,
            4 * self.hidden_size,
            world_size=world_size,
            gather_output=False,
            # init_method=init_method
        )

        # self.activation_func = mlp_activation_func

        # Project back to h.
        self.dense_4h_to_h = RowParallelLinear(
            4 * self.hidden_size,
            self.hidden_size,
            world_size=world_size,
            input_is_parallel=True,
            # init_method=output_layer_init_method
        )

        # self.dropout = torch.nn.Dropout(args.hidden_dropout)

    def forward(self, hidden_states):

        # [b, s, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        # intermediate_parallel = self.activation_func(intermediate_parallel)

        # [b, s, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        # output = self.dropout(output)
        return output


def megatron_mlp_run(rank, world_size, batch_size, input_row, hidden_dim, **args):
    # set cuda device
    proc_id = int(os.environ['SLURM_PROCID'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)

    # init default group
    local_rank = int(os.environ['SLURM_LOCALID'])
    node_list = os.environ['SLURM_NODELIST']
    host_ip = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    port = "29502"
    init_method = 'tcp://{}:{}'.format(host_ip, port)

    torch.distributed.init_process_group("nccl", init_method=init_method,
                                         world_size=world_size, rank=rank)
    
#     print("{} : default group initialized".format(rank))
    
    # init input tensor
    input_tensor = torch.rand((batch_size, input_row, hidden_dim)).cuda()
    dist.broadcast(input_tensor, 0)

    # init MLP layers
    mlp = ParallelMLP(hidden_size=hidden_dim, world_size=world_size).cuda()
#     print("{} : mlp initialized".format(rank))
    
    # pre-run
    with torch.no_grad():
        output = mlp(input_tensor)
    
    # profiling
    with torch.no_grad():
        with profiler.profile(profile_memory=True, use_cuda=True) as prof:
            output = mlp(input_tensor)

    if rank == 0:
        profiling_file_name = './profiling/megatron_world_{}_bs_{}_row_{}_dim_{}_profiler.txt'.format(
            world_size, batch_size, input_row, hidden_dim
        )
        with open(profiling_file_name , 'w')as f:
            print(prof.key_averages().table(sort_by="cpu_time_total", 
                                            row_limit=20), 
                  file=f)
    
    # timing for num_iters iterations
    duration_list = []
    num_iters = 100
    
    
    for i in range(num_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            output = mlp(input_tensor)
        end.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        duration_list.append(start.elapsed_time(end))
#         print(start.elapsed_time(end))

#         torch.cuda.synchronize()
#         start_time = time.time()
#         output = mlp(input_tensor, row_groups[rank_row], col_groups[rank_col])
#         torch.cuda.synchronize()
#         elapsed_time = time.time() - start_time
#         duration_list.append(elapsed_time)
    
    if rank == 0:
        import matplotlib.pyplot as plt
        
        duration_list = [t for t in duration_list]
#         avg_duration = sum(duration_list) * 1000 / num_iters
        avg_duration = sum(duration_list) / num_iters
        print("Duration/ms: {}".format(avg_duration))
        
        # write duration to file
        duration_file_name = './profiling/megatron_world_{}_bs_{}_row_{}_dim_{}_iter_{}_duration.txt'.format(
            world_size, batch_size, input_row, hidden_dim, num_iters
        )
        with open(duration_file_name, 'w') as f:
            for d in duration_list:
                f.write("{}\n".format(d))
            f.write("\nAverage: {}\n".format(avg_duration))
        
        # plot line chart
        plt.plot(range(num_iters), duration_list)
        img_name = './profiling/megatron_world_{}_bs_{}_row_{}_dim_{}_iter_{}_duration.jpg'.format(
            world_size, batch_size, input_row, hidden_dim, num_iters)
        plt.savefig(img_name)