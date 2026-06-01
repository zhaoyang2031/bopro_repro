#!/usr/bin/env python3
"""Add docking scores to existing offline datasets (first 1500 molecules per seed)."""
import torch
import warnings
import sys
import os
from multiprocessing import Pool
warnings.filterwarnings('ignore')

sys.path.insert(0, '/data/xk/zhaoyang/ram_repro/MolStitch-main')

data_dir = '/data/xk/zhaoyang/ram_repro/MolStitch-main/main/offline_cluster/data/offline_dataset'
docking_targets = ['parp1', 'fa7', '5ht1b', 'jak2', 'braf']
n_workers = 48


def compute_one_docking(args):
    smi, target = args
    try:
        from evaluators.dock.qvina2 import QuickVina2
        dock = QuickVina2(target=target)
        score = dock(smi)
        return (smi, target, score)
    except:
        return (smi, target, 0.0)


def main():
    for seed in range(10):
        fpath = os.path.join(data_dir, 'MolStitch_offline_dataset[%d].pt' % seed)
        if not os.path.exists(fpath):
            continue

        ds = torch.load(fpath, weights_only=False)
        fk = list(ds.keys())[0]

        # Check if docking scores already exist
        missing = [t for t in docking_targets if t not in ds[fk]]
        if not missing:
            print('Seed %d: already has all docking scores' % seed)
            continue

        # Only process first 1500 molecules
        smis = list(ds.keys())[:1500]
        print('Seed %d: adding %s for %d molecules' % (seed, missing, len(smis)))

        # Build tasks for missing targets only
        tasks = [(smi, target) for smi in smis for target in missing]
        print('  Computations: %d' % len(tasks))

        # Parallel compute
        with Pool(n_workers) as pool:
            for i, (smi, target, score) in enumerate(pool.imap_unordered(compute_one_docking, tasks, chunksize=50)):
                ds[smi][target] = score
                if (i + 1) % 500 == 0:
                    print('  %d/%d' % (i + 1, len(tasks)))

        torch.save(ds, fpath)
        print('Seed %d: saved' % seed)

    print('Done!')


if __name__ == '__main__':
    main()
