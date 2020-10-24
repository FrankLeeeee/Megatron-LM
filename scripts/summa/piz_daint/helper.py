import multiprocessing
import platform
import subprocess
import os
import re

import torch

def get_system_info():
    # get other system info
    cpu_count = multiprocessing.cpu_count()
    system = platform.system()
    cpu_model_name = ""
    
    command = "cat /proc/cpuinfo"
    all_info = subprocess.getoutput(command).strip()
    for line in all_info.split("\n"):
        if "model name" in line:
            cpu_model_name = re.sub( ".*model name.*:", "", line, 1)
            break
    
    world_size = world_size = int(os.environ['SLURM_NTASKS'])
    gpu_name = torch.cuda.get_device_name(0)
    gpu_count = torch.cuda.device_count()
    gpu_ram = torch.cuda.get_device_properties(0).total_memory
    
    info=dict(
        cpu_count=cpu_count,
        system=system,
        cpu_model_name=cpu_model_name,
        world_size=world_size,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        gpu_ram=gpu_ram
    )
    
    return info


if __name__ == "__main__":
    print(get_system_info())
    
