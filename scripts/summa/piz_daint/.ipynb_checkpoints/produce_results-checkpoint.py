import os
import os.path as osp
import matplotlib.pyplot as plt

# enums
methods = ['summa', 'megatron']
batch_size = [16]
input_row = [64]
hidden_dim = [25600]
nodes = [4, 16, 64]

def plot_results(source_path, output_path):
    for bs in batch_size:
        for row in input_row:
            for dim in hidden_dim:
                plt.clf()
                
                for method in methods:
                    x = nodes
                    y = []
                    for node in nodes:
                        txt_file = "{}_world_{}_bs_{}_row_{}_dim_{}_iter_10_duration.txt".format(method, node, bs, row, dim)
                        txt_file = osp.join(source_path, txt_file)
                        
                        with open(txt_file, 'r') as f:
                            text = f.readlines()
                            text = text[:10]
                            text = sorted([float(line) for line in text])
                            avg_min_duration = sum(text) / len(text)
                            
#                             avg_duration = float(text[101].split()[-1])
                            y.append(avg_min_duration)
                    plt.plot(x, y, label=method, marker='*')
                plt.xlabel('number of nodes')
                plt.ylabel('forward time/ms')
                plt.legend()
                
                img_name = 'bs_{}_row_{}_dim_{}.jpg'.format(bs, row, dim)
                img_name = osp.join(output_path, img_name)
                plt.savefig(img_name)
                
                
def profiling(source_path, output_path):
    for bs in batch_size:
        for row in input_row:
            for dim in hidden_dim:
                for method in methods:
                    for node in nodes:
                        # get system information
                        sys_info_file_name = "{}_world_{}_bs_{}_row_{}_dim_{}_sysinfo.txt".format(
                            method, node, bs, row, dim
                        )
                        sys_info_file_name = os.path.join(source_path, sys_info_file_name)
                        
                        with open(sys_info_file_name, 'r') as f:
                            text = f.readlines()
                        
                        
                        # get memory information
                        memory_file_name = "{}_world_{}_bs_{}_row_{}_dim_{}_memory.txt".format(
                            method, node, bs, row, dim
                        )
                        memory_file_name = os.path.join(source_path, memory_file_name)
                        
                        with open(memory_file_name, 'r') as f:
                            memory_text = f.readlines()
                            
                            for line in memory_text:
                                if 'Total Tensors:' in line:
                                    text.append(line)
                                
                                if 'The allocated memory' in line:
                                    text.append(line)
                        
                        # get speed information
                        speed_file = "{}_world_{}_bs_{}_row_{}_dim_{}_iter_10_duration.txt".format(
                            method, node, bs, row, dim
                        )
                        speed_file = os.path.join(source_path, speed_file)
                        
                        with open(speed_file, 'r') as f:
                            speed_text = f.readlines()
                            speed_text = speed_text[:10]
                            speed_text = [float(line) for line in speed_text]
                            
                            min_speed = min(speed_text)
                            max_speed = max(speed_text)
                            avg_speed = sum(speed_text) / len(speed_text)
                            
                            text.append('min time for forward/ms: {}\n'.format(min_speed))
                            text.append('max time for forward/ms: {}\n'.format(max_speed))
                            text.append('avg time for forward/ms: {}\n'.format(avg_speed))
                        
                        # output
                        output_file_name = "{}_world_{}_bs_{}_row_{}_dim_{}_iter_10_profiling.txt".format(
                            method, node, bs, row, dim
                        )
                        output_file_name = os.path.join(output_path, output_file_name)
                        
                        with open(output_file_name, 'w') as f:
                            f.writelines(text)
                            
                        

                
if __name__ == '__main__':
    source_path = './profiling/2020_10_16_2'
    output_path = './profiling/2020_10_16_2/results'
    
    if not osp.exists(output_path):
        os.mkdir(output_path)
        
    plot_results(source_path, output_path)
    profiling(source_path, output_path)
