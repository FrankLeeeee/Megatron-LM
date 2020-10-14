import os
import os.path as osp
import matplotlib.pyplot as plt

# enums
methods = ['summa', 'megatron']
batch_size = [32, 64]
input_row = [32, 64]
hidden_dim = [128, 256, 512]
nodes = [1, 4, 16, 64]

def plot_results(source_path, output_path):
    for bs in batch_size:
        for row in input_row:
            for dim in hidden_dim:
                plt.clf()
                
                for method in methods:
                    x = nodes
                    y = []
                    for node in nodes:
                        txt_file = "{}_world_{}_bs_{}_row_{}_dim_{}_iter_100_duration.txt".format(method, node, bs, row, dim)
                        txt_file = osp.join(source_path, txt_file)
                        with open(txt_file, 'r') as f:
                            text = f.readlines()
                            text = text[:100]
                            text = sorted([float(line) for line in text])
                            text = text[:20]
                            avg_min_20_duration = sum(text) / 20
                            
#                             avg_duration = float(text[101].split()[-1])
                            y.append(avg_min_20_duration)
                    plt.plot(x, y, label=method, marker='*')
                plt.xlabel('number of nodes')
                plt.ylabel('forward time/ms')
                plt.legend()
                
                img_name = 'bs_{}_row_{}_dim_{}.jpg'.format(bs, row, dim)
                img_name = osp.join(output_path, img_name)
                plt.savefig(img_name)

                
if __name__ == '__main__':
    source_path = './profiling'
    output_path = './results'
    
    if not osp.exists(output_path):
        os.mkdir(output_path)
        
    plot_results(source_path, output_path)
