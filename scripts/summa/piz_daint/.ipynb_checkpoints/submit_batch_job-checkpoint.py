import subprocess
import os

methods = ['summa', 'megatron']
batch_size = [32, 64]
input_row = [32, 64]
hidden_dim = [128, 256, 512]
nodes = [1, 4, 16, 64]

# for testing only
# methods = ['summa']
# batch_size = [8]
# input_row = [8]
# hidden_dim = [64]
# nodes = [4]

log_book = open('logbook.txt', 'w')
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
                    output = subprocess.getoutput("sbatch ./test.sh {} {} {} {}".format(
                        method, bs, row, dim
                    ))

                    job_id = output.split()[-1]
                    
                    log_book.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                        job_id, node, method, bs, row, dim
                    ))
                    
                    print("job {}: submitted".format(job_id))

log_book.close()
                    
                