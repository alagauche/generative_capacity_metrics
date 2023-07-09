"""
XXX seems like whole file is unused
"""

# builtins
import time
import os

# external libraries
from scipy.special import softmax
import numpy as np


# XXX unused
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


# XXX unused
# inverse one hot & decode, see below suggestion that could improve this
def get_argmax_char(chunk, int_to_char):
    return int_to_char[np.argmax(softmax(chunk))]


def get_char(chunk, alphabet):
    norm_chunk = get_norm(chunk)
    return np.random.choice(alphabet, 1, p=norm_chunk)[0]


def get_norm(chunk):
    chunk /= chunk.sum(axis=-1, keepdims=True)
    return np.clip(chunk, 1e-7, 1 - 1e-7)


"""
NOTE possible rewrite of encoding/decoding

STOI = { ch: i for i, ch in enumerate(alpha) }
ITOS = { i: ch for i, ch in enumerate(alpha) }

def encode(s):
  return [STOI[c] for c in s]

def decode(l):
  return ''.join([ITOS[i] for i in l])
"""

# XXX unused
def get_msa_onehot(msa):
    # XXX duplicate constant defined in config_vis.json, might want to pull from config and make a static constant
    # I think this is also in the mi3gpu folder that's a fork
    alpha = "ACDEFGHIKLMNPQRSTVWY-"
    seqs = np.array([[alpha.index(c) for c in s] for s in msa], dtype="u1")

    N, L = seqs.shape
    q = len(alpha)

    # XXX NOTE below works but is a bit tough to understand, I think using np.eye is easier to reason about
    # F_ONE_HOT = np.eye(len(alpha))
    # onehot = F_ONE_HOT[seqs]
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


# XXX pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True) is a better way to do this
def make_dir(dir_name):
    try:
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
            print("/tDirectory ", dir_name, " Created ")
        else:
            print("/tDirectory ", dir_name, " already exists")
    except OSError:
        pass
