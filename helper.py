# builtins
import time
import os

# external libraries
from scipy.special import softmax
import numpy as np


def timer(function):
    def clock(*args):
        start = time.perf_counter()
        result = function(*args)
        end = time.perf_counter() - start
        name = function.__name__
        # arg_str = ', '.join(repr(arg) for arg in args)
        print(f"[{end:.8f}] {name}")
        return result

    return clock


def get_argmax_char(chunk, int_to_char):
    return int_to_char[np.argmax(softmax(chunk))]


def get_char(chunk, alphabet):
    norm_chunk = get_norm(chunk)
    return np.random.choice(alphabet, 1, p=norm_chunk)[0]


def get_norm(chunk):
    chunk /= chunk.sum(axis=-1, keepdims=True)
    return np.clip(chunk, 1e-7, 1 - 1e-7)


def get_msa_onehot(msa):
    alpha = "ACDEFGHIKLMNPQRSTVWY-"
    seqs = np.array([[alpha.index(c) for c in s] for s in msa], dtype="u1")

    N, L = seqs.shape
    q = len(alpha)

    onehot = np.zeros((N, L, q), dtype="u1")
    onehot[np.arange(N)[:, None], np.arange(L)[None, :], seqs] = 1

    return onehot.reshape((N, L * q))


def print_msa_lengths(msas, msa_names):
    print("\tMSA lengths:")
    for msa, name in zip(msas, msa_names):
        print("\t" + name + ":\t" + str(len(msa)))


def get_splits(splits, msa):
    print("\trunning get_splits()")
    msa_keys = list(msa.keys())
    len_msa = len(msa_keys)
    train_end = int(round(splits[0] * len_msa, 0))
    val_end = int(round(train_end + len_msa * splits[1], 0))

    train_msa = msa_keys[:train_end]
    val_msa = msa_keys[train_end:val_end]
    test_msa = msa_keys[val_end:]

    return [train_msa, val_msa, test_msa]


"""

# args-based folder name, for hyperparameter tuning
def get_parent_dir_name(args):
    arg_strings = [str(x) for x in args][2:]
    dir_name = "vae_"
    for arg in arg_strings:
        dir_name += arg
        if arg != arg_strings[-1]:
            dir_name += "_"
    return dir_name

"""


def make_dir(dir_name):
    try:
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
            print("/tDirectory ", dir_name, " Created ")
        else:
            print("/tDirectory ", dir_name, " already exists")
    except OSError:
        pass
