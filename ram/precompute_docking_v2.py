#!/usr/bin/env python3
"""Pre-compute QuickVina2 docking scores for offline datasets.
Runs targets sequentially, each using 10 internal subprocesses.
Temp files go to /dev/shm via symlink.
"""
import torch
import sys
import os
import pickle
import time

sys.path.insert(0, '/data/xk/zhaoyang/ram_repro/MolStitch-main')
from evaluators.dock.qvina2 import QuickVina2

DOCKING_TARGETS = ['parp1', 'fa7', '5ht1b', 'jak2', 'braf']
DATA_DIR = '/data/xk/zhaoyang/ram_repro/MolStitch-main/main/offline_cluster/data/offline_dataset'
CACHE_DIR = '/data/xk/zhaoyang/ram_repro/docking_cache'
N_WORKERS = 10  # internal subprocesses per DockingVina


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Collect all unique SMILES
    all_smiles = set()
    for seed in range(10):
        fpath = os.path.join(DATA_DIR, 'MolStitch_offline_dataset[%d].pt' % seed)
        if os.path.exists(fpath):
            ds = torch.load(fpath, weights_only=False)
            all_smiles.update(ds.keys())
    all_smiles = sorted(all_smiles)
    print("Total unique SMILES: %d" % len(all_smiles))

    # Check which targets need computing
    fpath0 = os.path.join(DATA_DIR, 'MolStitch_offline_dataset[0].pt')
    ds = torch.load(fpath0, weights_only=False)
    fk = list(ds.keys())[0]
    todo = [t for t in DOCKING_TARGETS if ds[fk].get(t, 0.0) == 0.0]
    if not todo:
        print("All docking scores already computed!")
        return
    print("Targets to compute: %s" % todo)

    for target in todo:
        t_start = time.time()
        print("\n[%s] Starting %d molecules..." % (target, len(all_smiles)))

        dock = QuickVina2(target=target)
        raw_scores = dock.vina_oracle.predict(all_smiles)
        scores = {smi: float(s) for smi, s in zip(all_smiles, raw_scores)}

        # Save cache
        cache_path = os.path.join(CACHE_DIR, '%s_scores.pkl' % target)
        pickle.dump(scores, open(cache_path, 'wb'))
        elapsed = time.time() - t_start
        print("[%s] Computed in %.1f min" % (target, elapsed / 60.0))

        # Update seed files
        for seed in range(10):
            fpath = os.path.join(DATA_DIR, 'MolStitch_offline_dataset[%d].pt' % seed)
            if not os.path.exists(fpath):
                continue
            ds = torch.load(fpath, weights_only=False)
            for smi in ds:
                if smi in scores:
                    ds[smi][target] = scores[smi]
            torch.save(ds, fpath)
        print("[%s] Saved to all seed files" % target)

    # Verify
    print("\n=== Verification ===")
    ds = torch.load(fpath0, weights_only=False)
    for t in DOCKING_TARGETS:
        vals = [data.get(t, 0.0) for data in list(ds.values())[:20]]
        non_zero = sum(1 for v in vals if v != 0.0 and v > 0.001)
        sample = vals[:5]
        print("%s: %d/20 non-zero, sample: %s" % (t, non_zero, [round(v, 4) for v in sample]))

    print("\nDone!")


if __name__ == '__main__':
    main()
