import numpy as np
import pandas as pd
from mi3gpu.utils.seqload import loadSeqs

def compute_hams_dist(gen_seqs_file, label, data_home, parent_dir_name, name, keep):
    print("\t\t\t\tcompute_hams() for:")
    print("\t\t\t\t\t" + label + "\tkeep:\t" + str(keep))
    

    if label == "randomSeqs":
        load_name = parent_dir_name + "/" + name + "_" + label
        print("\t\t\t\t\t\trandom seqs")
        print("\t\t\t\t\t\tloading seqs file for hams:\t", load_name)
        seqs = loadSeqs(load_name)[0][0:keep]
    else:
        seqs = loadSeqs(data_home + "/" + gen_seqs_file)[0][0:keep]

    N, L = seqs.shape
    Npairs = N*(N-1)//2
    hamming = np.empty(Npairs, dtype='i4')

    c = 0
    for i in range(N-1):
        hamming[c:c+(N-i-1)] = np.sum(seqs[i,:] != seqs[i+1:,:], axis=1)
        c += N-i-1
    
    h_counter = dict()

    for h in hamming:
        if h not in h_counter.keys():
            h_counter[h] = 1
        else:
            h_counter[h] += 1
   
    # impute 0's for missing hams
    for x in range(1, L+1):
        if x not in h_counter.keys():
            h_counter[x] = 0

    df = pd.DataFrame(h_counter.items())
    df.columns = ['ham', 'freq']
    hamdist_file_name = "hamdist_" + label + "_" + name + "_" + str(keep) + ".csv"
    df.to_csv(parent_dir_name + "/" + hamdist_file_name, index=False)
    
    return hamdist_file_name
