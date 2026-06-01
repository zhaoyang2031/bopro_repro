"""
RAM (Offline-RaM) adapter for MolStitch MPO (Multi-Property Optimization).

Core RaM architecture preserved:
  - SimpleMLP: ranking surrogate model
  - Ranking losses: ListNet, RankNet, LambdaRank, etc.
  - Learning-to-rank paradigm: relative ordering > absolute scores

Changes from original:
  - Data: MolStitch offline .pt dataset instead of Design-Bench
  - Features: Morgan fingerprints instead of Design-Bench task inputs
  - Generator: REINVENT-style GRU (from GeneticGFN prior) for molecule generation
  - Training: surrogate model trained with ranking loss, then used as reward for generator
  - Evaluation: HV/R2 matching MolStitch protocol (run_gfn_proxy_pref.py)
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
from copy import deepcopy
import wandb

from rdkit import Chem
from rdkit.Chem import AllChem
# Add paths
ram_dir = os.path.dirname(os.path.realpath(__file__))
gfn_dir = os.path.join(ram_dir, '..', 'genetic_gfn-main', 'pmo', 'main', 'genetic_gfn')
ms_dir = os.path.join(ram_dir, '..', 'MolStitch-main')

# Import RAM modules first (before adding gfn_dir to path)
sys.path.insert(0, ram_dir)
from losses import get_loss_fn
from model import SimpleMLP

# MolStitch evaluators
sys.path.insert(0, ms_dir)
from evaluators.hypervolume import get_hypervolume, get_hypervolume_pygmo, get_pareto_fronts


# ============ Molecular Fingerprint Feature Extractor ============

class MolecularFingerprint:
    def __init__(self, radius=2, n_bits=2048):
        self.radius = radius
        self.n_bits = n_bits

    def smiles_to_fp(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self.n_bits)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
        return np.array(fp, dtype=np.float32)

    def batch_smiles_to_fp(self, smiles_list):
        fps = np.array([self.smiles_to_fp(smi) for smi in smiles_list])
        return torch.tensor(fps, dtype=torch.float32)


# ============ Offline Dataset ============

class MolOfflineDataset:
    def __init__(self, pt_file, obj_names, obj_weights, fp_extractor, valid_ratio=0.2):
        offline_dataset = torch.load(pt_file, weights_only=False)

        smiles_list = []
        scores_list = []
        multi_scores_dict = {}
        for smi, data in offline_dataset.items():
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            total = 0.0
            scores = []
            for name, weight in zip(obj_names, obj_weights):
                s = data.get(name, 0.0)
                scores.append(s)
                total += s * weight
            smiles_list.append(smi)
            scores_list.append(total)
            multi_scores_dict[smi] = scores

        # Normalize scores to [0, 1]
        scores_np = np.array(scores_list)
        y_min, y_max = scores_np.min(), scores_np.max()
        if y_max > y_min:
            scores_np = (scores_np - y_min) / (y_max - y_min)
        else:
            scores_np = np.zeros_like(scores_np)

        fps = np.array([fp_extractor.smiles_to_fp(smi) for smi in smiles_list])

        n = len(smiles_list)
        n_valid = max(1, int(n * valid_ratio))
        n_train = n - n_valid
        indices = np.random.permutation(n)
        train_idx = indices[:n_train]
        valid_idx = indices[n_train:]

        self.train_x = torch.tensor(fps[train_idx], dtype=torch.float32)
        self.train_y = torch.tensor(scores_np[train_idx], dtype=torch.float32).unsqueeze(1)
        self.valid_x = torch.tensor(fps[valid_idx], dtype=torch.float32)
        self.valid_y = torch.tensor(scores_np[valid_idx], dtype=torch.float32).unsqueeze(1)

        self.smiles = smiles_list
        self.all_scores = scores_np
        self.multi_scores_dict = multi_scores_dict  # {smi: [score_obj1, ..., score_objN]}

        print(f"MolOfflineDataset: {n_train} train, {n_valid} valid (from {n} total)")


# ============ HV/R2 Computation (matching MolStitch run_gfn_proxy_pref.py L305-319) ============

def compute_hv_r2(multi_scores, num_obj):
    """Compute HV and R2 using MolStitch evaluators on Pareto front.

    HV on Pareto front = HV on all scores (dominated points don't contribute).
    Uses pygmo for 4+ objectives (matching MolStitch optimizer.py L130-141).
    """
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


# ============ Ranking Model Training ============

def train_ranking_model(model, loss_fn, x_train, y_train, x_valid, y_valid,
                        n_epochs=100, list_length=128, num_samples=500, lr=3e-4, device='cuda'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_valid = x_valid.to(device)
    y_valid = y_valid.to(device)

    n = x_train.shape[0]
    min_loss = float('inf')
    best_state = None

    # Pre-compute listwise data like original RaM (create_special_dataset_fast_unique)
    print(f"Pre-computing {num_samples} lists of length {list_length}...")
    indices = torch.stack([torch.randperm(n)[:list_length] for _ in range(num_samples)])
    x_lists = x_train[indices]  # (num_samples, list_length, features)
    y_lists = y_train[indices]  # (num_samples, list_length, 1)

    # Create DataLoader for batching
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(x_lists, y_lists)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

    # Pre-compute validation lists
    n_valid = len(x_valid)
    val_indices = torch.stack([torch.randperm(n_valid)[:list_length] for _ in range(num_samples)])
    x_val_lists = x_valid[val_indices]
    y_val_lists = y_valid[val_indices]
    val_dataset = TensorDataset(x_val_lists, y_val_lists)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)  # (batch_size, list_length, features)
            y_batch = y_batch.to(device)  # (batch_size, list_length, 1)

            # Forward pass
            features = x_batch.shape[-1]
            y_pred = model(x_batch.reshape(-1, features))  # (batch_size*list_length, 1)
            y_pred = y_pred.reshape(x_batch.shape[0], list_length)  # (batch_size, list_length)
            y_batch = y_batch.squeeze(-1)  # (batch_size, list_length)

            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                features = x_batch.shape[-1]
                y_pred = model(x_batch.reshape(-1, features)).reshape(x_batch.shape[0], list_length)
                y_batch = y_batch.squeeze(-1)
                val_losses.append(loss_fn(y_pred, y_batch).item())
        val_loss = np.mean(val_losses)

        if val_loss < min_loss:
            min_loss = val_loss
            best_state = deepcopy(model.state_dict())

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}")
            wandb.log({"ranking_epoch": epoch+1, "train_loss": avg_loss, "val_loss": val_loss})

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ============ REINVENT-style Generator (from GeneticGFN) ============

def train_generator_with_ranking_reward(agent, prior, ranking_model, fp_extractor,
                                         offline_smiles, offline_scores,
                                         n_steps=5000, batch_size=200, sigma=500,
                                         lr=5e-4, device='cuda'):
    agent.rnn.to(device)
    prior.rnn.to(device)
    ranking_model = ranking_model.to(device)
    ranking_model.eval()

    for param in prior.rnn.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(agent.rnn.parameters(), lr=lr)

    import importlib.util
    if gfn_dir not in sys.path:
        sys.path.insert(0, gfn_dir)
    ds_path = os.path.join(gfn_dir, 'data_structs.py')
    spec = importlib.util.spec_from_file_location("gfn_data_structs", ds_path)
    gfn_ds = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gfn_ds)
    voc = gfn_ds.Vocabulary(init_from_file=os.path.join(gfn_dir, 'data', 'Voc'))

    offline_seqs = []
    for smi in offline_smiles:
        try:
            tokens = voc.tokenize(smi)
            encoded = voc.encode(tokens)
            offline_seqs.append(torch.tensor(encoded, dtype=torch.long))
        except:
            pass

    n_offline = len(offline_seqs)
    print(f"Generator training: {n_offline} offline molecules, {n_steps} steps")

    def collate_min(seqs):
        max_len = max(s.size(0) for s in seqs)
        batch = torch.zeros(len(seqs), max_len, dtype=torch.long)
        for i, s in enumerate(seqs):
            batch[i, :s.size(0)] = s
        return batch

    for step in range(n_steps):
        # === MAIN STEP (batch_size=200) ===
        idx = np.random.choice(n_offline, size=min(batch_size, n_offline), replace=False)
        batch = collate_min([offline_seqs[i] for i in idx]).to(device).long()

        agent_ll, _ = agent.likelihood(batch)
        prior_ll, _ = prior.likelihood(batch)

        smiles_batch = []
        for seq in batch:
            tokens = [voc.reversed_vocab.get(int(t), '') for t in seq if int(t) != voc.vocab['EOS']]
            smi = ''.join(tokens).replace('L', 'Cl').replace('R', 'Br')
            smiles_batch.append(smi)

        with torch.no_grad():
            fps = fp_extractor.batch_smiles_to_fp(smiles_batch).to(device)
            ranking_scores = torch.sigmoid(ranking_model(fps).squeeze())

        loss = torch.pow(agent_ll - prior_ll + sigma * ranking_scores, 2).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.rnn.parameters(), 1.0)
        optimizer.step()

        # === 8 AUGMENTATION ROUNDS (exp_replay=300, uniform sampling) ===
        exp_replay_size = 300
        aug_rounds = 8
        for _ in range(aug_rounds):
            aidx = np.random.choice(n_offline, size=min(exp_replay_size, n_offline), replace=False)
            a_batch = collate_min([offline_seqs[i] for i in aidx]).to(device).long()

            a_agent_ll, _ = agent.likelihood(a_batch)
            a_prior_ll, _ = prior.likelihood(a_batch)

            a_smiles_batch = []
            for seq in a_batch:
                tokens = [voc.reversed_vocab.get(int(t), '') for t in seq if int(t) != voc.vocab['EOS']]
                smi = ''.join(tokens).replace('L', 'Cl').replace('R', 'Br')
                a_smiles_batch.append(smi)

            with torch.no_grad():
                a_fps = fp_extractor.batch_smiles_to_fp(a_smiles_batch).to(device)
                a_ranking_scores = torch.sigmoid(ranking_model(a_fps).squeeze())

            a_loss = torch.pow(a_agent_ll - a_prior_ll + sigma * a_ranking_scores, 2).mean()

            optimizer.zero_grad()
            a_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.rnn.parameters(), 1.0)
            optimizer.step()

        if (step + 1) % 500 == 0:
            print(f"Step {step+1}/{n_steps}: loss={loss.item():.4f}")

    return agent


# ============ Main ============

def main():
    parser = argparse.ArgumentParser(description="RAM (Offline-RaM) for MolStitch MPO")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max_oracle_calls", type=int, default=10000)
    parser.add_argument("--oracles", type=str, default="jnk3 gsk3b qed sa")
    parser.add_argument("--weights", type=str, default="1 1 1 1")
    parser.add_argument("--loss", type=str, default="listnet",
                        choices=["listnet", "ranknet", "lambdarank", "rankcosine",
                                 "listmle", "approxndcg", "mse"])
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--fp_bits", type=int, default=2048)
    parser.add_argument("--gen_steps", type=int, default=5000)
    parser.add_argument("--gen_lr", type=float, default=5e-4)
    parser.add_argument("--sigma", type=int, default=500)
    parser.add_argument("--wandb", type=str, default="online")
    parser.add_argument("--run_name", type=str, default="")
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

    fp_extractor = MolecularFingerprint(radius=2, n_bits=hparams.fp_bits)

    # Load offline dataset (includes multi_scores for initial HV/R2)
    offline_data_path = os.path.join(
        ram_dir, '..', 'MolStitch-main', 'main', 'offline_cluster',
        'data', 'offline_dataset', f'MolStitch_offline_dataset[{hparams.seed}].pt'
    )
    mol_dataset = MolOfflineDataset(offline_data_path, obj_names, obj_weights, fp_extractor)
    n_offline = len(mol_dataset.smiles)

    # Wandb
    run_name = hparams.run_name or f"RAM_{hparams.loss}_seed{hparams.seed}"
    wandb.login(key='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')
    wandb.init(project='repro_ram', entity='1585515136-', name=run_name, config=vars(hparams))

    # ===== Initial HV/R2 from offline dataset (matching MolStitch step 0) =====
    all_multi_scores = dict(mol_dataset.multi_scores_dict)  # {smi: [score1, ..., scoreN]}
    n_oracle = n_offline
    HV, R2 = compute_hv_r2(all_multi_scores, num_obj)
    print(f"Initial (offline): HV={HV:.6f}, R2={R2:.6f}, n_oracle={n_oracle}")
    wandb.log({"HV": HV, "R2": R2, "n_oracle": n_oracle})

    # ===== Step 1: Train ranking model =====
    print(f"\n=== Step 1: Training ranking model ({hparams.loss}) ===")
    input_dim = hparams.fp_bits
    ranking_model = SimpleMLP(input_dim=input_dim, hidden_dim=[2048, 2048], output_dim=1)
    loss_fn = get_loss_fn(hparams.loss)

    ranking_model = train_ranking_model(
        ranking_model, loss_fn,
        mol_dataset.train_x, mol_dataset.train_y,
        mol_dataset.valid_x, mol_dataset.valid_y,
        n_epochs=hparams.n_epochs, list_length=128, num_samples=500,
        lr=hparams.lr, device=device
    )

    # ===== Step 2: Train REINVENT generator with ranking model as reward =====
    print(f"\n=== Step 2: Training generator with ranking reward ===")

    import importlib.util
    if gfn_dir not in sys.path:
        sys.path.insert(0, gfn_dir)
    gfn_model_path = os.path.join(gfn_dir, 'model.py')
    spec = importlib.util.spec_from_file_location("gfn_model", gfn_model_path)
    gfn_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gfn_model)

    ds_path = os.path.join(gfn_dir, 'data_structs.py')
    spec2 = importlib.util.spec_from_file_location("gfn_data_structs", ds_path)
    gfn_ds = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(gfn_ds)

    voc = gfn_ds.Vocabulary(init_from_file=os.path.join(gfn_dir, 'data', 'Voc'))

    prior = gfn_model.RNN(voc)
    prior.rnn.load_state_dict(torch.load(os.path.join(gfn_dir, 'data', 'Prior.ckpt'),
                                           map_location=device))
    agent = gfn_model.RNN(voc)
    agent.rnn.load_state_dict(torch.load(os.path.join(gfn_dir, 'data', 'Prior.ckpt'),
                                           map_location=device))

    agent.rnn.to(device)
    prior.rnn.to(device)
    agent = train_generator_with_ranking_reward(
        agent, prior, ranking_model, fp_extractor,
        mol_dataset.smiles, mol_dataset.all_scores,
        n_steps=hparams.gen_steps, batch_size=200, sigma=hparams.sigma,
        lr=hparams.gen_lr, device=device
    )

    # ===== Step 3: Evaluation (NO training, only sampling + oracle) =====
    print(f"\n=== Step 3: Evaluation ===")
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

    while n_oracle < hparams.max_oracle_calls:
        # Generate molecules
        seqs, _, _ = agent.sample(128)
        gen_smiles = []
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
                gen_smiles.append(smi)

        # Score with TDC oracles
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

        # Compute HV/R2 on ALL molecules (offline + generated)
        round_idx += 1
        HV, R2 = compute_hv_r2(all_multi_scores, num_obj)
        print(f"Round {round_idx}: {len(all_multi_scores)} molecules, "
              f"HV={HV:.6f}, R2={R2:.6f}, n_oracle={n_oracle}")
        wandb.log({"HV": HV, "R2": R2, "n_oracle": n_oracle})

    # Final
    HV, R2 = compute_hv_r2(all_multi_scores, num_obj)
    print(f"\nFinal: HV={HV:.6f}, R2={R2:.6f}, n_oracle={n_oracle}")
    wandb.log({"final_HV": HV, "final_R2": R2, "n_oracle": n_oracle})

    import json
    os.makedirs("results", exist_ok=True)
    with open(f"results/ram_mpo_{hparams.loss}_seed{hparams.seed}.json", 'w') as f:
        json.dump({"HV": HV, "R2": R2, "n_oracle": n_oracle, "seed": hparams.seed,
                    "loss": hparams.loss}, f, indent=2)


if __name__ == "__main__":
    main()
