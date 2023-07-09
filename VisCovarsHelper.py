import numpy as np
import pandas as pd
from mi3gpu.utils.seqload import loadSeqs
import argparse
import sys


def get_marginals(seqs, A, weights, nrmlz=True):
    nSeq, L = seqs.shape

    if weights == "0":
        weights = None

    if A > 16:  # the x + A*y operation below may overflow for u1
        seqs = seqs.astype("i4")

    if nrmlz:
        nrmlz = lambda x: x / np.sum(x, axis=-1, keepdims=True)
    else:
        nrmlz = lambda x: x

    def freqs(s, bins):
        return np.bincount(s, minlength=bins, weights=weights)

    f = nrmlz(np.array([freqs(seqs[:, i], A) for i in range(L)]))
    ff = nrmlz(
        np.array(
            [
                freqs(seqs[:, j] + A * seqs[:, i], A * A)
                for i in range(L - 1)
                for j in range(i + 1, L)
            ]
        )
    )
    return f, ff


"""
def get_marginals(msa_file, A, model_name):
    # A = alphabet_size
    # import msa using import_seqs
    print("\t\t\t\timporting msa:\t\t\t", msa_file)
    msa = seqload.loadSeqs(msa_file)[0]
    print("\t\t\t\tfinished msa import:\t\t", msa_file)
    #compute marginals, save to .npy files
    print("\t\t\t\tcomputing msa marginals:\t", msa_file)
    #uvms, bvms = getMarginals(msa, 21, weights)
    uvms, bvms = compute_marginals(msa, A, 0)
    uvms_file_name = model_name + "_uvms.npy"
    bvms_file_name = model_name + "_bvms.npy"
    np.save(parent_dir_name uvms_file_name, uvms)
    np.save(bvms_file_name, bvms)
    print("\t\t\t\tfinished computing marginals:\t", msa_file))
"""


def getL(size):
    return int(((1 + np.sqrt(1 + 8 * size)) // 2) + 0.5)


def getLq(J):
    return getL(J.shape[0]), int(np.sqrt(J.shape[1]) + 0.5)


def getUnimarg(ff):
    L, q = getLq(ff)
    ff = ff.reshape((L * (L - 1) // 2, q, q))
    marg = np.array(
        [np.sum(ff[0], axis=1)] + [np.sum(ff[n], axis=0) for n in range(L - 1)]
    )
    return marg / (np.sum(marg, axis=1)[:, None])  # correct any fp errors


def indepF(fab):
    L, q = getLq(fab)
    fabx = fab.reshape((fab.shape[0], q, q))
    fa1, fb2 = np.sum(fabx, axis=2), np.sum(fabx, axis=1)
    fafb = np.array([np.outer(fa, fb).flatten() for fa, fb in zip(fa1, fb2)])
    return fafb


def getM(x, diag_fill=0):
    L = getL(len(x))
    M = np.empty((L, L))
    M[np.triu_indices(L, k=1)] = x
    M = M + M.T
    M[np.diag_indices(L)] = diag_fill
    return M


def get_covars(label, bvms_file, A, parent_dir_name, data_home, model_name):
    covars_file_name = "covars_ " + label + "_" + model_name + ".npy"
    # randomSeqs of VAE are in parent_dir, all others are in data_home
    if label is "randomSeqs":
        bvms_load_name = parent_dir_name + "/" + bvms_file
        save_name = parent_dir_name + "/" + covars_file_name
    else:
        bvms_load_name = data_home + "/" + bvms_file
        save_name = data_home + "/" + covars_file_name
    C = bimarg - indepF(bvms_load_name)
    np.save(save_name, C)


def get_bvms(label, msa_file, A, parent_dir_name, data_home, model_name):
    bvms_file_name = "bvms_" + label + "_" + model_name + ".npy"
    print("bvms_file_name:\t", bvms_file_name)
    print("parent:\t", parent_dir_name)
    print("data_home:\t", data_home)
    print("model_name:\t", model_name)
    # randomSeqs of VAE are in parent_dir, all others are in data_home
    if label == "randomSeqs":
        load_name = parent_dir_name + "/" + msa_file
        save_name = parent_dir_name + "/" + bvms_file_name
    else:
        load_name = data_home + "/" + msa_file
        save_name = data_home + "/" + bvms_file_name
    print("\t\t\t\timporting msa for:\t", label, "\t", load_name)
    msa = seqload.loadSeqs(load_name)[0]
    print("\t\t\t\tfinished msa import for:\t", label)
    print("\t\t\t\tcomputing bvms for:\t", label)
    bvms = compute_bvms(msa, A, 0)
    np.save(save_name, bvms)
    print("\t\t\t\tfinished computing bvms for:\t", label)
    return bvms_file_name


def compute_bvms(seqs, q, weights, nrmlz=True):
    nSeq, L = seqs.shape

    if weights == "0":
        weights = None

    if q > 16:  # the x + q*y operation below may overflow for u1
        seqs = seqs.astype("i4")

    if nrmlz:
        nrmlz = lambda x: x / np.sum(x, axis=-1, keepdims=True)
    else:
        nrmlz = lambda x: x

    def freqs(s, bins):
        return np.bincount(s, minlength=bins, weights=weights)

    f = nrmlz(np.array([freqs(seqs[:, i], q) for i in range(L)]))
    ff = nrmlz(
        np.array(
            [
                freqs(seqs[:, j] + q * seqs[:, i], q * q)
                for i in range(L - 1)
                for j in range(i + 1, L)
            ]
        )
    )
    return ff


"""

def compute_bvms(seqs, A, weights, nrmlz=True):
    nSeq, L = seqs.shape

    if weights == '0':
        weights = None

    if A > 16: # the x + A*y operation below may overflow for u1
        seqs = seqs.astype('i4')

    if nrmlz:
        nrmlz = lambda x: x/np.sum(x, axis=-1, keepdims=True)
    else:
        nrmlz = lambda x: x

    def freqs(s, bins):
        return np.bincount(s, minlength=bins, weights=weights)

    ff = nrmlz(np.array([freqs(seqs[:,j] + A*seqs[:,i], A*A) for i in range(L-1) for j in range(i+1, L)]))
    return ff
"""
