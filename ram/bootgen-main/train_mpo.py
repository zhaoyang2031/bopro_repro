"""
BootGen adapter for MolStitch MPO (Multi-Property Optimization).

Core BootGen architecture preserved:
  - DropoutRegressor: MLP proxy with MC dropout
  - Rank-weighted training + bootstrap augmentation
  - Generator: REINVENT-style RNN (from GeneticGFN)

Adapted for SMILES:
  - Tokenizer: GeneticGFN SMILES vocabulary
  - Evaluation: HV/R2 matching MolStitch protocol

Key insight from original BootGen (train.py):
  - Warm-up: first 83% of stages use uniform sampling
  - Bootstrap: only after warm-up, every 5 stages, add top-2 proxy-scored molecules
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import wandb

from rdkit import Chem

# Add paths
bootgen_dir = os.path.dirname(os.path.realpath(__file__))
gfn_dir = os.path.join(bootgen_dir, '..', 'genetic_gfn-main', 'pmo', 'main', 'genetic_gfn')
ms_dir = os.path.join(bootgen_dir, '..', 'MolStitch-main')

sys.path.insert(0, bootgen_dir)
sys.path.insert(0, os.path.join(bootgen_dir, '..'))

from lib.proxy.regression import DropoutRegressor

# MolStitch evaluators
sys.path.insert(0, ms_dir)
from evaluators.hypervolume import get_hypervolume, get_hypervolume_pygmo, get_pareto_fronts

# GeneticGFN modules (for pretrained RNN generator)
if gfn_dir not in sys.path:
    sys.path.insert(0, gfn_dir)


# ============ SMILES Tokenizer (from GeneticGFN) ============

import re

def replace_halogen(string):
    br = re.compile('Br')
    cl = re.compile('Cl')
    string = br.sub('R', string)
    string = cl.sub('L', string)
    return string

def tokenize_smiles(smiles):
    regex = '(\[[^\[\]]{1,6}\])'
    smiles = replace_halogen(smiles)
    char_list = re.split(regex, smiles)
    tokenized = []
    for char in char_list:
        if char.startswith('['):
            tokenized.append(char)
        else:
            chars = [unit for unit in char]
            [tokenized.append(unit) for unit in chars]
    return tokenized

class SMILESVocabulary:
    def __init__(self, voc_file):
        with open(voc_file, 'r') as f:
            chars = f.read().split()
        self.special_tokens = ['EOS', 'GO']
        self.chars = chars + self.special_tokens
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, char_list):
        return [self.vocab.get(ch, 0) for ch in char_list]

    def decode(self, indices):
        chars = []
        for i in indices:
            if i == self.vocab['EOS']:
                break
            chars.append(self.reversed_vocab.get(i, ''))
        smiles = "".join(chars)
        smiles = smiles.replace("L", "Cl").replace("R", "Br")
        return smiles

    def __len__(self):
        return self.vocab_size

class SMILESTokenizer:
    def __init__(self, voc_file):
        self.vocab = SMILESVocabulary(voc_file)
        self.num_tokens = self.vocab.vocab_size
        self.go_token = self.vocab.vocab['GO']
        self.eos_token = self.vocab.vocab['EOS']

    def process(self, token_lists):
        lens = [len(t) for t in token_lists]
        max_len = max(lens)
        padded = [t + [0] * (max_len - len(t)) for t in token_lists]
        return torch.tensor(padded, dtype=torch.long)

    def tokenize_and_encode(self, smiles_list):
        encoded = []
        for smi in smiles_list:
            tokens = tokenize_smiles(smi)
            indices = self.vocab.encode(tokens)
            indices.append(self.eos_token)
            encoded.append(indices)
        return encoded


# ============ Offline Dataset ============

class MolOfflineDataset:
    """Load offline dataset and extract multi-objective scores."""
    def __init__(self, pt_file, obj_names, obj_weights, valid_ratio=0.2):
        self.obj_names = obj_names

        offline_dataset = torch.load(pt_file, weights_only=False)

        smiles_list = []
        scores_list = []
        multi_scores_dict = {}
        for smi, data in offline_dataset.items():
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            total = 0.0
            multi = []
            for name, weight in zip(obj_names, obj_weights):
                s = data.get(name, 0.0)
                multi.append(s)
                total += s * weight
            smiles_list.append(smi)
            scores_list.append(total)
            multi_scores_dict[smi] = multi

        # Normalize weighted scores to [0, 1]
        scores_np = np.array(scores_list)
        y_min, y_max = scores_np.min(), scores_np.max()
        if y_max > y_min:
            scores_np = (scores_np - y_min) / (y_max - y_min)
        else:
            scores_np = np.zeros_like(scores_np)

        n = len(smiles_list)
        n_valid = max(1, int(n * valid_ratio))
        n_train = n - n_valid
        indices = np.random.permutation(n)
        train_idx = indices[:n_train]
        valid_idx = indices[n_train:]

        self.train_smiles = [smiles_list[i] for i in train_idx]
        self.train_scores = scores_np[train_idx]
        self.valid_smiles = [smiles_list[i] for i in valid_idx]
        self.valid_scores = scores_np[valid_idx]

        self.all_smiles = smiles_list
        self.all_scores = scores_np
        self.multi_scores_dict = multi_scores_dict

        print(f"MolOfflineDataset: {n_train} train, {n_valid} valid (from {n} total)")


# ============ HV/R2 Computation (matching MolStitch) ============

def compute_hv_r2(multi_scores, num_obj):
    """Compute HV and R2 using MolStitch evaluators on Pareto front."""
    if len(multi_scores) < 1:
        return 0.0, 0.0
    try:
        all_scores_arr = np.array(list(multi_scores.values()))
        if num_obj >= 4:
            HV, R2 = get_hypervolume_pygmo(None, all_scores_arr, num_obj)
        else:
            HV, R2 = get_hypervolume(None, all_scores_arr, num_obj)
        return float(HV), float(R2)
    except Exception as e:
        print(f"HV/R2 error: {e}")
        return 0.0, 0.0


# ============ Decode RNN sequences to SMILES ============

def seqs_to_smiles(seqs, voc):
    smiles_list = []
    for seq in seqs:
        tokens = []
        for t in seq:
            t = int(t)
            if t == voc.vocab['EOS']:
                break
            if t in voc.reversed_vocab:
                tokens.append(voc.reversed_vocab[t])
        smi = ''.join(tokens).replace('L', 'Cl').replace('R', 'Br')
        if smi:
            smiles_list.append(smi)
    return smiles_list


# ============ Main ============

def main():
    parser = argparse.ArgumentParser(description="BootGen for MolStitch MPO")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max_oracle_calls", type=int, default=10000)
    parser.add_argument("--max_len", type=int, default=120)
    parser.add_argument("--oracles", type=str, default="jnk3 gsk3b qed sa")
    parser.add_argument("--weights", type=str, default="1 1 1 1")
    parser.add_argument("--wandb", type=str, default="online")
    parser.add_argument("--run_name", type=str, default="")
    # Table 17 aligned hyperparameters
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--sigma", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--gen_steps", type=int, default=5000)
    hparams = parser.parse_args()

    obj_names = hparams.oracles.split()
    obj_weights = [float(w) for w in hparams.weights.split()]
    num_obj = len(obj_names)
    assert num_obj == len(obj_weights)

    device = torch.device(f'cuda:{hparams.device}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(hparams.device)
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    np.random.seed(hparams.seed)

    # Wandb
    run_name = hparams.run_name or f"BootGen_4obj_seed{hparams.seed}"
    if hparams.wandb != 'disabled':
        wandb.login(key='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')
        wandb.init(project='repro_ram', entity='1585515136-', name=run_name, config=vars(hparams))

    # Load offline dataset
    offline_data_path = os.path.join(
        bootgen_dir, '..', 'MolStitch-main', 'main', 'offline_cluster',
        'data', 'offline_dataset', f'MolStitch_offline_dataset[{hparams.seed}].pt'
    )
    mol_dataset = MolOfflineDataset(offline_data_path, obj_names, obj_weights)
    n_offline = len(mol_dataset.all_smiles)

    # ===== Initial HV/R2 from offline dataset =====
    all_multi_scores = dict(mol_dataset.multi_scores_dict)
    n_oracle = n_offline
    HV, R2 = compute_hv_r2(all_multi_scores, num_obj)
    print(f"Initial (offline): HV={HV:.6f}, R2={R2:.6f}, n_oracle={n_oracle}")
    if hparams.wandb != 'disabled':
        wandb.log({"HV": HV, "R2": R2, "n_oracle": n_oracle})

    # ===== Load pretrained RNN (from GeneticGFN) =====
    from model import RNN
    from data_structs import Vocabulary

    voc = Vocabulary(init_from_file=os.path.join(gfn_dir, 'data', 'Voc'))

    prior = RNN(voc)
    prior.rnn.load_state_dict(torch.load(os.path.join(gfn_dir, 'data', 'Prior.ckpt'),
                                           map_location=device))
    for param in prior.rnn.parameters():
        param.requires_grad = False

    agent = RNN(voc)
    agent.rnn.load_state_dict(torch.load(os.path.join(gfn_dir, 'data', 'Prior.ckpt'),
                                           map_location=device))
    agent.rnn.to(device)

    # ===== Train proxy model (BootGen's DropoutRegressor) =====
    tokenizer = SMILESTokenizer(os.path.join(gfn_dir, 'data', 'Voc'))
    proxy_num_token = tokenizer.num_tokens
    max_len = hparams.max_len

    # Prepare proxy training data
    train_tokens = tokenizer.tokenize_and_encode(mol_dataset.train_smiles)
    valid_tokens = tokenizer.tokenize_and_encode(mol_dataset.valid_smiles)

    class MolProxyDataset:
        def __init__(self, train_tokens, train_scores, valid_tokens, valid_scores):
            self.train_tokens = train_tokens
            self.train_scores = train_scores
            self.valid_tokens = valid_tokens
            self.valid_scores = valid_scores

        def sample(self, batch_size):
            n = len(self.train_tokens)
            idx = np.random.choice(n, size=min(batch_size, n), replace=False)
            tokens = [self.train_tokens[i] for i in idx]
            scores = self.train_scores[idx]
            return tokens, scores

        def validation_set(self):
            return self.valid_tokens, self.valid_scores

    proxy_data = MolProxyDataset(
        train_tokens, mol_dataset.train_scores,
        valid_tokens, mol_dataset.valid_scores
    )

    print("Training proxy model...")
    proxy = DropoutRegressor(tokenizer, proxy_num_token, max_len)
    proxy.fit(proxy_data)

    # ===== Phase 1: BootGen training with warm-up + rank-weighted + bootstrap =====
    print(f"\n=== Phase 1: BootGen training ({hparams.gen_steps} steps) ===")

    # Pre-tokenize offline data
    offline_seqs = []
    offline_rewards = []
    for smi, score in zip(mol_dataset.all_smiles, mol_dataset.all_scores):
        try:
            tokens = voc.tokenize(smi)
            encoded = voc.encode(tokens)
            offline_seqs.append(torch.tensor(encoded, dtype=torch.long))
            offline_rewards.append(float(score))
        except:
            pass
    n_offline_seqs = len(offline_seqs)

    def collate_min(seqs):
        ml = max(s.size(0) for s in seqs)
        batch = torch.zeros(len(seqs), ml, dtype=torch.long)
        for i, s in enumerate(seqs):
            batch[i, :s.size(0)] = s
        return batch

    optimizer = torch.optim.Adam(agent.rnn.parameters(), lr=hparams.lr)
    sigma = hparams.sigma
    batch_sz = hparams.batch_size
    aug_rounds = 8  # Table 17: Augmentation round = 8

    # BootGen: warm-up first 80% with uniform, then rank-weighted + bootstrap
    warmup_steps = int(hparams.gen_steps * 0.8)
    print(f"Warm-up: {warmup_steps} steps (uniform), then {hparams.gen_steps - warmup_steps} steps (rank-weighted + bootstrap)")

    # Bootstrap augmentation data
    all_seqs = list(offline_seqs)
    all_rewards = list(offline_rewards)

    for step in range(hparams.gen_steps):
        if step < warmup_steps:
            # === WARM-UP: Uniform random sampling (like original BootGen stages 1-1250) ===
            idx = np.random.choice(n_offline_seqs, size=min(batch_sz, n_offline_seqs), replace=False)
            batch = collate_min([offline_seqs[i] for i in idx]).to(device).long()
            batch_r = torch.tensor([offline_rewards[i] for i in idx]).to(device)
        else:
            # === RANK-WEIGHTED: Sample by rank (like original BootGen after warm-up) ===
            scores_np = np.array(offline_rewards)
            ranks = np.argsort(np.argsort(-1 * scores_np))
            weights = 1.0 / (1e-2 * len(scores_np) + ranks)
            idx = list(torch.utils.data.WeightedRandomSampler(
                weights=weights, num_samples=min(batch_sz, len(scores_np)), replacement=True))
            batch = collate_min([offline_seqs[i] for i in idx]).to(device).long()
            batch_r = torch.tensor([offline_rewards[i] for i in idx]).to(device)

        agent_ll, _ = agent.likelihood(batch)
        prior_ll, _ = prior.likelihood(batch)

        loss = torch.pow(agent_ll - prior_ll + sigma * batch_r, 2).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.rnn.parameters(), 1.0)
        optimizer.step()

        # === 8 AUGMENTATION ROUNDS (exp_replay=300, uniform sampling) ===
        exp_replay_size = 300
        for _ in range(aug_rounds):
            aidx = np.random.choice(n_offline_seqs, size=min(exp_replay_size, n_offline_seqs), replace=False)
            a_batch = collate_min([all_seqs[i] for i in aidx]).to(device).long()
            a_r = torch.tensor([all_rewards[i] for i in aidx], device=device)

            a_agent_ll, _ = agent.likelihood(a_batch)
            a_prior_ll, _ = prior.likelihood(a_batch)
            a_loss = torch.pow(a_agent_ll - a_prior_ll + sigma * a_r, 2).mean()

            optimizer.zero_grad()
            a_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.rnn.parameters(), 1.0)
            optimizer.step()

        # === BOOTSTRAP AUGMENTATION (after warm-up, every 100 steps) ===
        if step >= warmup_steps and (step + 1) % 100 == 0:
            agent.rnn.eval()
            gen_seqs, _, _ = agent.sample(128)
            gen_smiles = seqs_to_smiles(gen_seqs, voc)

            new_seqs = []
            new_rewards = []
            for smi in gen_smiles:
                if Chem.MolFromSmiles(smi) is None:
                    continue
                try:
                    encoded = voc.encode(voc.tokenize(smi))
                    seq_tensor = torch.tensor(encoded, dtype=torch.long)
                    # Score with proxy
                    x_list = [tokenizer.tokenize_and_encode([smi])[0]]
                    with torch.no_grad():
                        proxy_score, _ = proxy.eval(x_list)
                    proxy_val = proxy_score.mean().item()
                    new_seqs.append(seq_tensor)
                    new_rewards.append(proxy_val)
                except:
                    pass

            # Add top-2 proxy-scored molecules to training set (BootGen bootstrap)
            if len(new_seqs) >= 2:
                sorted_idx = np.argsort(new_rewards)[-2:]
                for i in sorted_idx:
                    all_seqs.append(new_seqs[i])
                    all_rewards.append(new_rewards[i])

            proxy_mean = np.mean(new_rewards) if new_rewards else 0.0
            valid_count = sum(1 for s in gen_smiles if Chem.MolFromSmiles(s) is not None)
            print(f"Step {step+1}/{hparams.gen_steps}: loss={loss.item():.4f}, "
                  f"proxy_mean={proxy_mean:.4f}, valid={valid_count}/{len(gen_smiles)}, "
                  f"dataset_size={len(all_seqs)}")
            if hparams.wandb != 'disabled':
                wandb.log({"step": step+1, "loss": loss.item(), "proxy_mean": proxy_mean,
                           "valid_ratio": valid_count/max(len(gen_smiles),1),
                           "dataset_size": len(all_seqs)})
            agent.rnn.train()

        if (step + 1) % 500 == 0:
            print(f"Step {step+1}/{hparams.gen_steps}: loss={loss.item():.4f}")

    # ===== Phase 2: Evaluation (NO training, only sampling + oracle) =====
    print(f"\n=== Phase 2: Evaluation ===")
    agent.rnn.eval()

    from tdc import Oracle as TDCOracle
    tdc_oracles = {}
    for name in obj_names:
        if name in ('jnk3', 'gsk3b'):
            tdc_oracles[name] = TDCOracle(name=name.upper())
        elif name == 'qed':
            tdc_oracles[name] = TDCOracle(name='QED')
        elif name == 'sa':
            tdc_oracles[name] = TDCOracle(name='SA')

    eval_budget = hparams.max_oracle_calls - n_oracle
    round_idx = 0
    print(f"Eval budget: {eval_budget} calls")

    while n_oracle < hparams.max_oracle_calls:
        gen_seqs, _, _ = agent.sample(128)
        gen_smiles = seqs_to_smiles(gen_seqs, voc)

        for smi in gen_smiles:
            if Chem.MolFromSmiles(smi) is None:
                continue
            if smi in all_multi_scores:
                continue
            scores = []
            for name in obj_names:
                scores.append(tdc_oracles[name](smi))
            all_multi_scores[smi] = scores
            n_oracle += 1

        round_idx += 1
        HV, R2 = compute_hv_r2(all_multi_scores, num_obj)
        print(f"Round {round_idx}: {len(all_multi_scores)} molecules, "
              f"HV={HV:.6f}, R2={R2:.6f}, n_oracle={n_oracle}")
        if hparams.wandb != 'disabled':
            wandb.log({"HV": HV, "R2": R2, "n_oracle": n_oracle})

    # Final
    HV, R2 = compute_hv_r2(all_multi_scores, num_obj)
    print(f"\nFinal: HV={HV:.6f}, R2={R2:.6f}, n_oracle={n_oracle}")
    if hparams.wandb != 'disabled':
        wandb.log({"final_HV": HV, "final_R2": R2, "n_oracle": n_oracle})

    import json
    os.makedirs("results", exist_ok=True)
    with open(f"results/bootgen_mpo_seed{hparams.seed}.json", 'w') as f:
        json.dump({"HV": HV, "R2": R2, "n_oracle": n_oracle, "seed": hparams.seed}, f, indent=2)


if __name__ == "__main__":
    main()
