import pandas as pd
from collections import defaultdict
import sys
if len(sys.argv) < 3:
    print ('No enough input files')
    sys.exit()
#interested_pair_set = pd.read_csv('E:\\backup\\nn_project2\\nn_project2\\nn_project2\\github\\data\\hesc1_TF_target.csv')
### input gene list reference   known gene pair list   path of generated gene pair list    path of number list of generated gene pair list
gene_list = list(pd.read_csv(sys.argv[1],sep='\t',header=None).iloc[:,0])

count = 0
count_exclude = 0
exclude_list = []
xx_list = []
h_tf_target = defaultdict(list)
s = open (sys.argv[2])
for line in s:
    info = line.split()
    if info[0] in gene_list and info[1] in gene_list:
        if not h_tf_target[info[0]]:
            h_tf_target[info[0]].append(info[1])
        elif info[1] not in h_tf_target[info[0]]:
            h_tf_target[info[0]].append(info[1])
        else:
            count = count + 1
            xx_list.append(line)
    else:
        count_exclude = count_exclude+1
        exclude_list.append(info)
s.close()

target_num_total = 0
import random
gene_pairs = []
countx_list = [0]
for tf in h_tf_target.keys():
    target_num = len(h_tf_target[tf])
    target_num_total =  target_num_total + target_num
    print (target_num)
    no_target_list = [i for i in range (len(gene_list)) if gene_list[i] not in h_tf_target[tf]]
    no_target_random_list = random.choices(no_target_list, k = target_num)
    countx_list.append(countx_list[-1]+target_num*3)
    for i in range (target_num):
        gene_pairs.append(tf+'\t'+h_tf_target[tf][i]+'\t1')
        gene_pairs.append(h_tf_target[tf][i] + '\t' + tf + '\t2')
        gene_pairs.append(tf + '\t' + gene_list[no_target_random_list[i]] + '\t0')


s = open (sys.argv[4],'w')
for i in countx_list:
    s.write(str(i)+'\n')

s.close()

s = open (sys.argv[3],'w')
for i in gene_pairs:
    s.write(i+'\n')

s.close()