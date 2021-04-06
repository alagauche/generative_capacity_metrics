#!/usr/bin/env python
import numpy as np
from numpy import random
import sys, os
from mi3gpu.utils import seqload
from multiprocessing import Pool, set_start_method, Lock
from highmarg import highmarg
set_start_method('fork')

###############################################################################
# load data in

print("starting homs_script")
seq_lim = 10000
reps = int(sys.argv[1])
targetSeqs = seqload.loadSeqs(sys.argv[2])[0]
refSeqs = seqload.loadSeqs(sys.argv[3])[0]
mi3Seqs = seqload.loadSeqs(sys.argv[4])[0]
randomSeqs = seqload.loadSeqs(sys.argv[5])[0]
indepSeqs = seqload.loadSeqs(sys.argv[6])[0]
ref_trunc = seqload.loadSeqs(sys.argv[7])[0]        # this has only 5.99M from mi3Seqs
target_trunc = seqload.loadSeqs(sys.argv[8])[0]     # this has only 10K from mi3Seqs
print("loaded target_trunc")
deepSeqs = seqload.loadSeqs(sys.argv[9])[0]
# progenSeqs = seqload.loadSeqs(sys.argv[10])[0]
parent_dir_name = sys.argv[10]
start = int(sys.argv[11])
end = int(sys.argv[12])
synth_nat = sys.argv[13]
output_dir = sys.argv[14]
weights = None

print(len(sys.argv))
print(sys.argv)
'''
if len(sys.argv) > 7:
    weights = [np.load(sys.argv[7]).astype('f4'), None, None, None]
    print("using weights")
'''
#dataseqs, sampleseqs = dataseqs[:(len(dataseqs)-2*Ns)], dataseqs[-Ns:]

Ntarget = targetSeqs.shape[0]
#Ndw = np.sum(weights[0]) if weights is not None else Nd
Nref = refSeqs.shape[0]
Nm = mi3Seqs.shape[0]
Nv = randomSeqs.shape[0]
Ni = indepSeqs.shape[0]
Nd = deepSeqs.shape[0]
#Np = progenSeqs.shape[0]
Nreftrunc = ref_trunc.shape[0]
Ntargettrunc = target_trunc.shape[0]
print("MSA lengths:", 
    "\n\ttarget:\t", Ntarget,
    "\n\tref:\t", Nref,
    "\n\tmi3:\t", Nm,
    "\n\tvae:\t", Nv,
    "\n\tindep:\t", Ni,
    "\n\tdeepSeq:\t", Nd,
    # \n\tprogen:\t", Np,
    "\n\tref-trunc:\t", Nreftrunc,
    "\n\ttarget-trunc:\t", Ntargettrunc
)
L = targetSeqs.shape[1]


# msas = [trainSeqs, mi3Seqs, randomSeqs, indepSeqs, targetSeqs]    # old line
msas = [targetSeqs, mi3Seqs, randomSeqs, indepSeqs, deepSeqs, refSeqs, ref_trunc, target_trunc]  # mod line
# we will get a speedup in get_subseq if msas are in fortran memory-order
msas = [np.asfortranarray(msa) for msa in msas]

'''
print("Dataset sizes:")
print("Data: {}\nModel: {}\nIndep: {}\nFSample: {}".format(Ndw, Nm, Ni, Ns))
'''

###############################################################################

def get_subseq(msa, pos, out=None):
    if out is None:
        ret = np.empty((msa.shape[0], len(pos)), dtype='u1')
    else:
        assert(out.shape == (msa.shape[0], len(pos)))
        assert(out.dtype == np.dtype('u1'))
        ret = out

    for n,p in enumerate(pos):
        ret[:,n] = msa[:,p]
    return ret

def job(arg):
    npos, i, seed = arg
    rng = np.random.default_rng(seed)

    pos = np.sort(rng.choice(L, npos, replace=False))

    subseqs = [get_subseq(m, pos) for m in msas]

    counts = highmarg(subseqs, weights=weights).T.copy()
    ftarget, fm, fv, fi, fd, fref, fref_trunc, ftarget_trunc = counts/np.sum(counts, axis=1, keepdims=True)
    #ftarget, fm, fv, fi, fd, fp, fref, fref_trunc, ftarget_trunc = counts/np.sum(counts, axis=1, keepdims=True)
    #ftarget, fm, fv, fi, fref, fref_trunc, ftarget_trunc = counts/np.sum(counts, axis=1, keepdims=True)    # orig
    process_marg(npos, i, pos, ftarget, fm, fv, fi, fd, fref, fref_trunc, ftarget_trunc, locks[npos]) 
    #process_marg(npos, i, pos, ftarget, fm, fv, fi, fd, fp, fref, fref_trunc, ftarget_trunc, locks[npos])
    #process_marg(npos, i, pos, ftarget, fm, fv, fi, fref, fref_trunc, ftarget_trunc, locks[npos])

def run_jobs(npos_range, reps):
    root_seed = np.random.SeedSequence()

    print("Starting workers...")
    with Pool() as pool:
        pool.map(job, ((n, i, root_seed.spawn(1)[0]) for n in npos_range for i in range(reps)))

###############################################################################
# User-defined statistics here. Define "init_marg_files" and "process_marg"
from scipy.stats import pearsonr as pr

#def process_marg(npos, i, pos, ftarget, fm, fv, fi, fref, fref_trunc, ftarget_trunc, lock):    # orig
#def process_marg(npos, i, pos, ftarget, fm, fv, fi, fd, fp, fref, fref_trunc, ftarget_trunc, lock):    
def process_marg(npos, i, pos, ftarget, fm, fv, fi, fd, fref, fref_trunc, ftarget_trunc, lock):
    top20d = np.argsort(ftarget)[-20:]
    if "nat" in synth_nat:
        top20d_trunc = np.argsort(ftarget_trunc)[-20:]  # different target, so sort by that
        #top20s = np.argsort(fs)[-20:]
     
    with lock:
        print("npos {:3d},  set {:3d},   n_unique={:10d}".format(
            npos, i, ftarget.shape[0]))
        with open(output_dir + "/r20_{}".format(npos), "at") as f:
            if "nat" in synth_nat:
            # only nat uses target_trunc and ref_trunc for black line
                print(pr(ftarget[top20d], fm[top20d]), # target vs mi3   
	                pr(ftarget[top20d], fv[top20d]), # targt vs vae
                    pr(ftarget[top20d], fi[top20d]), # target vs indep
                    pr(ftarget[top20d], fd[top20d]), # target vs deep
                    # pr(ftarget[top20d], fp[top20d]), # target vs progen
                    pr(ftarget_trunc[top20d_trunc], fref_trunc[top20d_trunc]), # target_trunc vs ref_trunc
                    file=f)
            # both synthetics, 10K and 1M, use normal target and ref for black line
            else:
                print(pr(ftarget[top20d], fm[top20d]), # target vs mi3   
	                pr(ftarget[top20d], fv[top20d]), # targt vs vae
                    pr(ftarget[top20d], fi[top20d]), # target vs indep
                    pr(ftarget[top20d], fd[top20d]), # target vs deep
                    # pr(ftarget[top20d], fp[top20d]), # target vs progen
                    pr(ftarget[top20d], fref[top20d]), # target vs ref
                    file=f)

def init_marg_files(npos):
    # clear file
    with open(output_dir + "/r20_{}".format(npos), "wt") as f:
        f.write("")

###############################################################################
# Choose npos to compute, and start jobs

npos_range = range(start, end)
for n in npos_range:
    init_marg_files(n)
locks = {n: Lock() for n in npos_range}

run_jobs(npos_range, reps)
