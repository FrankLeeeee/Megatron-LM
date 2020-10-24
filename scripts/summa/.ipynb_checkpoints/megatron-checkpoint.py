import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


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
        output_parallel = F.linear(input_, self.weight)
        # All-reduce across all the partitions.
        output_ = torch.distributed.all_reduce(output_parallel)

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


def run(rank, world_size, batch_size, input_row, hidden_dim):
    # init env var
    os.environ['MASTER_ADDR'] = '192.168.41.6'
    os.environ['MASTER_PORT'] = '29525'

    # set cuda device
    torch.cuda.set_device(rank)

    # init input tensor
    input_tensor = torch.rand((batch_size, input_row, hidden_dim)).cuda()

    # init MLP layers
    mlp = ParallelMLP(hidden_size=hidden_dim, world_size=world_size)

    torch.cuda.cudart().cudaProfilerStart()
    output = mlp(input_tensor)
    torch.cuda.cudart().cudaProfilerStop()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--input_row", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.environ['NCCL_SOCKET_IFNAME'] = 'enp1s0f1'
#    os.environ['NCCL_DEBUG'] = 'INFO'
    mp.spawn(run,
             args=(args.world_size, args.batch_size,
                   args.input_row, args.hidden_dim),
             nprocs=args.world_size,
             join=True)
