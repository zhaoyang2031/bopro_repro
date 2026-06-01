#!/usr/bin/env python3
"""Pre-compute QuickVina2 docking scores for offline datasets (parallel)."""
import torch
import sys
import os
import pickle
from multiprocessing import Process

sys.path.insert(0, '/data/xk/zhaoyang/ram_repro/MolStitch-main')

docking_targets = ['parp1', 'fa7', '5ht1b', 'jak2', 'braf']
data_dir = '/data/xk/zhaoyang/ram_repro/MolStitch-main/main/offline_cluster/data/offline_dataset'
results_dir = '/data/xk/zhaoyang/ram_repro/docking_results'


def compute_target(target, all_smiles):
    """Worker: compute docking for one target."""
    from evaluators.dock.qvina2 import QuickVina2
    print("[%s] Starting docking for %d molecules..." % (target, len(all_smiles)), flush=True)
    dock = QuickVina2(target=target)
    scores = dock.vina_oracle.predict(all_smiles)
    result = {smi: float(s) for smi, s in zip(all_smiles, scores)}
    out_path = os.path.join(results_dir, '%s_scores.pkl' % target)
    pickle.dump(result, open(out_path, 'wb'))
    print("[%s] Done: %d scores saved" % (target, len(result)), flush=True)


def main():
    os.makedirs(results_dir, exist_ok=True)

    # Collect all unique SMILES
    all_smiles = set()
    for seed in range(10):
        fpath = os.path.join(data_dir, 'MolStitch_offline_dataset[%d].pt' % seed)
        if os.path.exists(fpath):
            ds = torch.load(fpath, weights_only=False)
            all_smiles.update(ds.keys())
    all_smiles = sorted(all_smiles)
    print("Total unique SMILES: %d" % len(all_smiles))

    # Check which targets need computing
    ds_check = torch.load(os.path.join(data_dir, 'MolStitch_offline_dataset[0].pt'), weights_only=False)
    fk = list(ds_check.keys())[0]
    needs_compute = [t for t in docking_targets if ds_check[fk].get(t, 0.0) == 0.0]
    if not needs_compute:
        print("All docking scores already computed!")
        return
    print("Targets to compute: %s" % needs_compute)

    # Launch all targets in parallel
    procs = []
    for target in needs_compute:
        p = Process(target=compute_target, args=(target, all_smiles))
        p.start()
        procs.append(p)
        print("Launched %s (PID %d)" % (target, p.pid))

    for p in procs:
        p.join()
    print("All targets computed.")

    # Load results and update datasets
    for target in needs_compute:
        result_path = os.path.join(results_dir, '%s_scores.pkl' % target)
        if not os.path.exists(result_path):
            print("WARNING: %s results not found!" % target)
            continue
        scores = pickle.load(open(result_path, 'rb'))
        print("Loading %s: %d scores" % (target, len(scores)))

        for seed in range(10):
            fpath = os.path.join(data_dir, 'MolStitch_offline_dataset[%d].pt' % seed)
            if not os.path.exists(fpath):
                continue
            ds = torch.load(fpath, weights_only=False)
            for smi in ds:
                if smi in scores:
                    ds[smi][target] = scores[smi]
            torch.save(ds, fpath)
        print("  %s saved to all seed files" % target)

    # Verify
    ds = torch.load(os.path.join(data_dir, 'MolStitch_offline_dataset[0].pt'), weights_only=False)
    fk = list(ds.keys())[0]
    for t in docking_targets:
        vals = [ds[s].get(t, 0.0) for s in list(ds.keys())[:10]]
        non_zero = sum(1 for v in vals if v != 0.0)
        print("%s: %d/10 non-zero, sample: %s" % (t, non_zero, vals[:5]))

    print("All done!")


if __name__ == '__main__':
    main()
