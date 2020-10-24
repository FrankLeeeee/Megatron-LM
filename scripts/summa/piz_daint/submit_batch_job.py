import subprocess
import os

methods = ['summa', 'megatron']
batch_size = [8]
input_row = [64]
hidden_dim = [25600]
nodes = [1, 4, 16, 64]
output_path = './profiling/2020_10_19_1'
log_book_path = os.path.join(output_path, 'logbook.txt')

# for testing only
# methods = ['summa']
# batch_size = [8]
# input_row = [8]
# hidden_dim = [64]
# nodes = [4]

if not os.path.exists(output_path):
    os.mkdir(output_path)

log_book = open(log_book_path, 'w')
log_book.write('job_id\tnode\tmethod\tbs\trow\tdim\n')

for node in nodes:
    #change node setting in the batch job script
    with open("./test.sh", "r") as file: 
        text = file.readlines()
    
    text[4] = "#SBATCH --nodes={}\n".format(node)
    
    with open("./test.sh", "w") as file: 
        file.writelines(text)
    
    for bs in batch_size:
        for row in input_row:
            for dim in hidden_dim:
                for method in methods:
                    output = subprocess.getoutput("sbatch ./test.sh {} {} {} {} {}".format(
                        method, bs, row, dim, output_path
                    ))

                    job_id = output.split()[-1]
                    
                    log_book.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                        job_id, node, method, bs, row, dim
                    ))
                    
                    print("job {}: submitted".format(job_id))

log_book.close()
                    
                