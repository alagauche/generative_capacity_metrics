from pathlib import Path

import numpy as np
import fire

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


def get_covars(label: str, bvms_file: Path, path: Path):
    # randomSeqs of VAE are in parent_dir, all others are in data_home

    save_name = path / f"covars_{label}.npy"
    bvms_path = path / bvms_file
    bimarg = np.load(bvms_path)
    C = bimarg - indepF(bimarg)
    np.save(save_name, C)


if __name__ == '__main__':
    fire.Fire(get_covars)
