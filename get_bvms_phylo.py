"""
- This script will take an MSA as input in the format of a CSV.
- MSA sequences will have the 'seq' as the column name in the header.
- The MSA is imported using Pandas "csv_read" and seqload helper functions.
- MSA importation in this way is a necessary pre-processing step
    for getMarginals.
- After importation, getMarginals will extract the marginals and
    output them as .npy files.
- These .npy files can then be imported by downstream analysis and
    visualization scripts.

Run the script like this:
<script_name> <msa_as_csv> <msa_type> <msa_protein>
python get_marginals.py natural_msa.csv natural kinase
# FIXME docstring above is not correct
"""
import sys
from pathlib import Path

import numpy as np
import fire

from mi3gpu.utils import seqload

def compute_bvms(seqs, q: int, weights, nrmlz=True):
    nSeq, L = seqs.shape

    # if weights != '0':
    #    weights = np.load(weights)

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

    # f = nrmlz(np.array([freqs(seqs[:,i], q) for i in range(L)]))
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


def get_bvms(label: str, msa_file: Path, source: Path, dest: Path, A: int, num_seqs: int):
    # randomSeqs of VAE are in parent_dir, all others are in data_home
    print("inside get_bvms_phylo, calling get_bvms on ", label)

    load_name = source / msa_file
    bvms_file_name = f"bvms_{label}.npy"
    save_name = dest / bvms_file_name
    msa = seqload.loadSeqs(load_name)[0][:num_seqs]

    print("\t\t\t\timporting msa for:\t", label, "\t", load_name)
    print("\t\t\t\tfinished msa import for:\t", label)
    print("\t\t\t\tcomputing bvms for:\t", label)
    bvms = compute_bvms(msa, A, "0")
    np.save(save_name, bvms)
    print("\t\t\t\tfinished computing bvms for:\t", label)
    return bvms_file_name

if __name__ == "__main__":
    fire.Fire(get_bvms)
