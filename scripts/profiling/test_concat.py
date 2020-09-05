import torch
import time

def test_concat():
    # init two tensors
    tensor_1 = torch.rand(4, 512, 1024).cuda()
    tensor_2 = torch.rand(4, 512, 1024).cuda()

    # count timing
    torch.cuda.synchronize()
    start = time.time()
    output = torch.cat((tensor_1, tensor_2), dim=0)
    torch.cuda.synchronize()
    end = time.time()
    duration = end - start

    # clear
    del tensor_1, tensor_2, output
    torch.cuda.empty_cache()

    return duration


if __name__ == "__main__":
    torch.cuda.init()
    torch.cuda.empty_cache()

    duration_list = []

    # experiment
    for i in range(20):
        duration = test_concat()
        duration_list.append(duration)
    
    print("Duration: {}".format(duration_list))
    print("Mean: {}".format(sum(duration_list) / len(duration_list)))
    print("Max: {}".format(max(duration_list)))
    print("Min: {}".format(min(duration_list)))
    
