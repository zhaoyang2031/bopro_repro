#!/usr/bin/env python3
"""Pre-compute docking scores for offline datasets and save them."""
import torch
import warnings
import sys
import os
import numpy as np
warnings.filterwarnings('ignore')

sys.path.insert(0, '/data/xk/zhaoyang/ram_repro/MolStitch-main')

# Check if qvina2 binary exists
qvina_path = '/data/xk/zhaoyang/ram_repro/MolStitch-main/evaluators/dock/qvina02'
if not os.path.exists(qvina_path):
    print("ERROR: qvina02 binary not found at", qvina_path)
    sys.exit(1)

from evaluators.dock.qvina2 import QVina2

docking_targets = ['parp1', 'fa7', '5ht1b', 'jak2', 'braf']
data_dir = '/data/xk/zhaoyang/ram_repro/MolStitch-main/main/offline_cluster/data/offline_dataset'

# Collect all unique SMILES across all seeds
all_smiles = set()
for seed in range(10):
    fpath = os.path.join(data_dir, 'MolStitch_offline_dataset[%d].pt' % seed)
    if os.path.exists(fpath):
        ds = torch.load(fpath, weights_only=False)
        all_smiles.update(ds.keys())

all_smiles = list(all_smiles)
print("Total unique SMILES: %d" % len(all_smiles))

# Compute docking scores
docking_scores = {}  # smi -> {target: score}
for target in docking_targets:
    print("Computing %s..." % target)
    pdbqt = '/data/xk/zhaoyang/ram_repro/MolStitch-main/evaluators/dock/%s.pdbqt' % target
    dock = QVina2(protein=pdbqt)
    for i, smi in enumerate(all_smiles):
        if smi not in docking_scores:
            docking_scores[smi] = {}
        try:
            docking_scores[smi][target] = dock(smi)
        except:
            docking_scores[smi][target] = 0.0
        if (i + 1) % 100 == 0:
            print("  %s: %d/%d" % (target, i + 1, len(all_smiles)))

# Save intermediate results
import pickle
pickle.dump(docking_scores, open('/data/xk/zhaoyang/ram_repro/docking_scores_cache.pkl', 'wb'))
print("Saved docking_scores_cache.pkl")

# Update offline datasets
for seed in range(10):
    fpath = os.path.join(data_dir, 'MolStitch_offline_dataset[%d].pt' % seed)
    if not os.path.exists(fpath):
        continue
    ds = torch.load(fpath, weights_only=False)
    modified = False
    for smi in ds:
        for target in docking_targets:
            if target not in ds[smi]:
                ds[smi][target] = docking_scores.get(smi, {}).get(target, 0.0)
                modified = True
    if modified:
        torch.save(ds, fpath)
        print("Updated seed %d" % seed)

print("Done!")
