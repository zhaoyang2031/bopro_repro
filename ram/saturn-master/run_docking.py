"""Saturn (Mamba) full-offline docking for parp1/fa7/jak2/braf/5ht1b x 10 seeds.
Same protocol as REINVENT4 run_docking.py.
"""
import os, sys, argparse, torch, numpy as np, random
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

sys.path.insert(0, '/data/xk/zhaoyang/ram_repro/MolStitch-main')
from evaluators.hypervolume import get_hypervolume, get_pareto_fronts
from botorch.utils.multi_objective.hypervolume import Hypervolume

sys.path.insert(0, '/data/xk/zhaoyang/ram_repro/saturn-master')
from models.generator import Generator

PRIOR_PATH = '/data/xk/zhaoyang/ram_repro/saturn-master/experimental_reproduction/checkpoint_models/zinc-250k-mamba-epoch-50.prior'
OFFLINE_DIR = '/data/xk/zhaoyang/ram_repro/MolStitch-main/main/offline_cluster/data/offline_dataset'

DOCKING_TARGETS = ['parp1', 'fa7', 'jak2', 'braf', '5ht1b']

TASK_CFG = {}
for dt in DOCKING_TARGETS:
    TASK_CFG[f'dock_{dt}'] = {
        'names': [dt, 'qed', 'sa'],
        'keys': [dt, 'qed', 'sa'],
        'w': [1/3, 1/3, 1/3],
    }


def normalize_offline_score(name, raw_score):
    if name in DOCKING_TARGETS:
        s = -raw_score / 20.0
        return max(s, 0.0)
    return raw_score


def normalize_online_score(name, raw_score):
    if name == 'sa':
        return (10.0 - raw_score) / 9.0
    elif name in DOCKING_TARGETS:
        s = -raw_score / 20.0
        return max(s, 0.0)
    return raw_score


def make_oracles(names):
    from evaluators.dock.qvina2 import QuickVina2
    from tdc import Oracle
    oracles = []
    for n in names:
        if n in DOCKING_TARGETS:
            oracles.append(QuickVina2(n))
        else:
            oracles.append(Oracle(n))
    return oracles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True,
                        choices=[f'dock_{t}' for t in DOCKING_TARGETS])
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--wandb', default='online')
    parser.add_argument('--run_name', default=None)
    parser.add_argument('--max_oracle_calls', type=int, default=3000)
    parser.add_argument('--offline_limit', type=int, default=1500)
    args = parser.parse_args()

    device = args.device
    seed = args.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if args.wandb != 'disabled':
        import wandb
        name = args.run_name or f'Saturn_{args.task}_seed{seed}'
        wandb.init(project='repro_ram', entity='1585515136-', name=name, reinit=True)

    cfg = TASK_CFG[args.task]
    oracle_names = cfg['names']
    data_keys = cfg['keys']
    obj_weights = cfg['w']
    n_obj = len(obj_weights)

    print(f"Loading Mamba prior from: {PRIOR_PATH}")
    prior = Generator.load_from_file(PRIOR_PATH, device)
    agent = Generator.load_from_file(PRIOR_PATH, device)

    for param in prior.network.parameters():
        param.requires_grad = False

    # Load offline data (same preprocessing as REINVENT4)
    offline_path = f'{OFFLINE_DIR}/MolStitch_offline_dataset[{seed}].pt'
    offline_data = torch.load(offline_path, weights_only=False)
    offline_smiles, offline_scores, multi_scores = [], [], {}
    skipped = 0
    for smi, data in offline_data.items():
        if len(offline_smiles) >= args.offline_limit:
            break
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        Chem.RemoveStereochemistry(mol)
        smi_clean = Chem.MolToSmiles(mol, isomericSmiles=False)
        try:
            tokens = prior.tokenizer.tokenize(smi_clean)
            prior.vocabulary.encode(tokens)
        except Exception:
            skipped += 1
            continue
        scores = [normalize_offline_score(k, data.get(k, 0.0)) for k in data_keys]
        multi_scores[smi_clean] = scores
        offline_smiles.append(smi_clean)
        offline_scores.append(sum(s * w for s, w in zip(scores, obj_weights)))
    print(f'Loaded {len(offline_smiles)} offline molecules (skipped {skipped})')

    agent.network.train()

    sigma, lr = 500.0, 5e-4
    batch_size = 200
    exp_replay_size = 300
    aug_rounds = 8
    offline_steps = 5000
    oracle_budget = args.max_oracle_calls
    n_calls = len(offline_smiles)
    optimizer = torch.optim.Adam(agent.network.parameters(), lr=lr)

    # ===== Offline training (NO SMILES randomization, aligned with REINVENT4) =====
    print(f'Offline training: {offline_steps} steps...')
    for step in range(offline_steps):
        idx = np.random.choice(len(offline_smiles), size=min(batch_size, len(offline_smiles)), replace=False)
        b_smi = [offline_smiles[i] for i in idx]
        b_r = torch.tensor([np.clip(offline_scores[i], 0, 1) for i in idx], device=device)

        try:
            agent_ll = agent.likelihood_smiles(b_smi)
            prior_ll = prior.likelihood_smiles(b_smi)
            loss = torch.pow(agent_ll - prior_ll + sigma * b_r, 2).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.network.parameters(), max_norm=1.0)
            optimizer.step()
        except RuntimeError:
            continue

        for _ in range(aug_rounds):
            aidx = np.random.choice(len(offline_smiles), size=min(exp_replay_size, len(offline_smiles)), replace=False)
            a_smi = [offline_smiles[i] for i in aidx]
            a_r = torch.tensor([np.clip(offline_scores[i], 0, 1) for i in aidx], device=device)
            try:
                agent_ll_a = agent.likelihood_smiles(a_smi)
                prior_ll_a = prior.likelihood_smiles(a_smi)
                a_loss = torch.pow(agent_ll_a - prior_ll_a + sigma * a_r, 2).mean()
                optimizer.zero_grad()
                a_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.network.parameters(), max_norm=1.0)
                optimizer.step()
            except RuntimeError:
                continue

        if (step + 1) % 500 == 0:
            print(f'  Step {step + 1}/{offline_steps}')

    print('Offline training done. Starting evaluation...')

    # ===== Evaluation =====
    oracles = make_oracles(oracle_names)
    from tdc import Evaluator, Oracle
    sa_scorer = Oracle(name='SA')
    div_eval = Evaluator(name='Diversity')

    mol_buffer = multi_scores.copy()
    agent.network.eval()

    log_interval = 100
    log_multiple = 1
    while n_calls < oracle_budget:
        seqs, smiles_list, _ = agent.sample_sequences_and_smiles(batch_size)
        smiles_list = [s for s in smiles_list if s and Chem.MolFromSmiles(s)]
        valid_new = [s for s in smiles_list if s not in mol_buffer]

        for smi in valid_new:
            if n_calls >= oracle_budget:
                break
            scores = [float(o(smi)) for o in oracles]
            for si, nm in enumerate(oracle_names):
                scores[si] = normalize_online_score(nm, scores[si])
            mol_buffer[smi] = scores
            n_calls += 1

        if n_calls >= log_interval * log_multiple:
            all_scores = np.array(list(mol_buffer.values()))
            HV, R2 = 0.0, 0.0
            try:
                hv_calc = Hypervolume(ref_point=torch.zeros(n_obj))
                HV = float(hv_calc.compute(torch.tensor(all_scores)))
            except Exception:
                pass
            try:
                _, pareto_scores = get_pareto_fronts(None, all_scores)
                if len(pareto_scores) > 1:
                    _, R2 = get_hypervolume(None, pareto_scores, n_obj)
                    R2 = float(R2)
            except Exception:
                pass

            items = sorted(mol_buffer.items(),
                           key=lambda x: sum(x[1][i] * obj_weights[i] for i in range(n_obj)),
                           reverse=True)
            top100_smi = [x[0] for x in items[:100]]
            top100_sc = [sum(x[1][i] * obj_weights[i] for i in range(n_obj)) for x in items[:100]]

            t1 = max(top100_sc) if top100_sc else 0
            t10 = np.mean(sorted(top100_sc, reverse=True)[:10])
            t100 = np.mean(top100_sc)
            sa = np.mean([sa_scorer(s) for s in top100_smi[:100]])
            div = div_eval(top100_smi[:100]) if top100_smi else 0

            print(f'{n_calls}/{oracle_budget} | avg_top1: {t1:.3f} | avg_top10: {t10:.3f} | '
                  f'avg_top100: {t100:.3f} | avg_sa: {sa:.3f} | HV: {HV:.6f} | R2: {R2:.6f} | div: {div:.3f}')

            if args.wandb != 'disabled':
                wandb.log({'avg_top1': t1, 'avg_top10': t10, 'avg_top100': t100,
                           'avg_sa': sa, 'diversity_top100': div,
                           'HV': HV, 'R2': R2, 'n_oracle': n_calls})
            log_multiple += 1

    all_scores = np.array(list(mol_buffer.values()))
    final_HV, final_R2 = 0.0, 0.0
    try:
        hv_calc = Hypervolume(ref_point=torch.zeros(n_obj))
        final_HV = float(hv_calc.compute(torch.tensor(all_scores)))
    except Exception:
        pass
    try:
        _, pareto_scores = get_pareto_fronts(None, all_scores)
        if len(pareto_scores) > 1:
            _, final_R2 = get_hypervolume(None, pareto_scores, n_obj)
            final_R2 = float(final_R2)
    except Exception:
        pass
    print(f'Final HV: {final_HV:.6f} R2: {final_R2:.6f}')
    if args.wandb != 'disabled':
        wandb.log({'final_HV': final_HV, 'final_R2': final_R2, 'n_oracle': n_calls})
        wandb.finish()

    print('Done!')


if __name__ == '__main__':
    main()
