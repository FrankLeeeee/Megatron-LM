import torch
import torch.nn.functional as F

import argparse
import time
import ctypes


_cudart = ctypes.CDLL('libcudart.so')

def cu_prof_start():
  ret = _cudart.cudaProfilerStart()
  if ret != 0:
    raise Exception('cudaProfilerStart() returned %d' % ret)


def cu_prof_stop():
  ret = _cudart.cudaProfilerStop()
  if ret != 0:
    raise Exception('cudaProfilerStop() returned %d' % ret)

if __name__ == "__main__":
    # init distributed environment
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--mid_dim', default=128, type=int, help='the dimension of each channel')
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--split_ratio', default=0.5, type=float, help='node rank for distributed training')
    parser.add_argument('--pipeline', action="store_true", help='node rank for distributed training')
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group("nccl", init_method='env://')
    
    # profiling
    mid_dim = args.mid_dim
    batch = args.batch_size
    
    weight_shape = [1024, mid_dim] 
    input_shape = [batch, 512, mid_dim]
    weight = torch.rand(weight_shape).cuda().half()
    inputs = torch.rand(input_shape).cuda().half()

#     for _ in range(10):
#         torch.cuda.synchronize()
#         output = F.linear(inputs, weight)
#         torch.distributed.all_reduce(output)
#     torch.distributed.all_reduce(output)

    cu_prof_start()
    with torch.cuda.profiler.profile():
        with torch.autograd.profiler.emit_nvtx():
            split_size = (int(batch*args.split_ratio), batch - int(batch*args.split_ratio))
            micro_batch = torch.split(inputs, split_size, dim=0)
            torch.cuda.synchronize()

            start = time.time()
            for _ in range(10):
                if not args.pipeline:
                    output = F.linear(inputs, weight)
                    torch.distributed.all_reduce(output)
                else:
                    output1 = F.linear(micro_batch[0], weight)
                    handle1 = torch.distributed.all_reduce(output1, async_op=True)
                    output2 = F.linear(micro_batch[1], weight)
                    handle2 = torch.distributed.all_reduce(output2, async_op=True)
                    handle1.wait()
                    handle2.wait()
            
            torch.cuda.synchronize()
            print("pipeline: {}ms".format((time.time()-start)*1000))
            cu_prof_stop()
