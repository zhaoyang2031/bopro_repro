"""Full-offline REINVENT training for MPO experiments.
Loads offline data, trains with REINVENT loss (no oracle calls), evaluates by sampling.
"""
import os, sys, argparse, torch, numpy as np, random
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')  # suppress RDKit warnings

sys.path.insert(0, '/data/xk/zhaoyang/ram_repro/MolStitch-main')
from evaluators.hypervolume import get_hypervolume, get_pareto_fronts

sys.path.insert(0, '/data/xk/zhaoyang/ram_repro/REINVENT4-main')

PRIOR_PATH = '/data/xk/zhaoyang/ram_repro/REINVENT4-main/priors/reinvent.prior'
OFFLINE_DIR = '/data/xk/zhaoyang/ram_repro/MolStitch-main/main/offline_cluster/data/offline_dataset'

TASK_CFG = {
    'mpo2obj': {'names': ['jnk3', 'gsk3b'],            'keys': ['jnk3', 'gsk3b'],            'w': [0.5, 0.5]},
    'mpo3obj': {'names': ['jnk3', 'gsk3b', 'qed'],      'keys': ['jnk3', 'gsk3b', 'qed'],      'w': [1/3, 1/3, 1/3]},
    'mpo4obj': {'names': ['jnk3', 'gsk3b', 'qed', 'sa'], 'keys': ['jnk3', 'gsk3b', 'qed', 'sa'], 'w': [0.25, 0.25, 0.25, 0.25]},
}


def load_model(path, device):
    """Load REINVENT4 model from saved .prior file."""
    from reinvent.models.reinvent.models.model import Model
    from reinvent.models import ReinventAdapter
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model = Model(ckpt['vocabulary'], ckpt['tokenizer'], ckpt['metadata'],
                  ckpt['network_params'], max_sequence_length=ckpt['max_sequence_length'],
                  device=torch.device(device))
    model.network.load_state_dict(ckpt['network'])
    adapter = ReinventAdapter(model)
    adapter.model.network.to(device)
    return adapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['mpo2obj', 'mpo3obj', 'mpo4obj'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--wandb', default='online')
    parser.add_argument('--run_name', default=None)
    args = parser.parse_args()

    device = args.device
    seed = args.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if args.wandb != 'disabled':
        import wandb
        name = args.run_name or f'REINVENT_{args.task}_seed{seed}'
        wandb.init(project='repro_ram', entity='1585515136-', name=name, reinit=True)

    cfg = TASK_CFG[args.task]
    oracle_names = cfg['names']
    data_keys = cfg['keys']
    obj_weights = cfg['w']
    n_obj = len(obj_weights)

    # Load models
    prior = load_model(PRIOR_PATH, device)
    agent = load_model(PRIOR_PATH, device)

    # Load offline data (remove stereochemistry for prior vocabulary compat)
    offline_path = f'{OFFLINE_DIR}/MolStitch_offline_dataset[{seed}].pt'
    offline_data = torch.load(offline_path, weights_only=False)
    offline_smiles, offline_scores, multi_scores = [], [], {}
    skipped = 0
    for smi, data in offline_data.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        Chem.RemoveStereochemistry(mol)
        smi_clean = Chem.MolToSmiles(mol, isomericSmiles=False)
        # Accept molecule if vocabulary can encode it; skip if not
        try:
            prior.model.vocabulary.encode(prior.model.tokenizer.tokenize(smi_clean))
        except RuntimeError:
            skipped += 1
            continue
        scores = [data.get(k, 0.0) for k in data_keys]
        multi_scores[smi_clean] = scores
        offline_smiles.append(smi_clean)
        offline_scores.append(sum(s * w for s, w in zip(scores, obj_weights)))
    print(f'Loaded {len(offline_smiles)} offline molecules (skipped {skipped} with unknown tokens)')

    prior.set_mode('training')
    agent.set_mode('training')

    sigma, lr = 500.0, 5e-4
    batch_size = 200
    exp_replay_size = 300
    aug_rounds = 8
    offline_steps = 5000
    oracle_budget = 10000
    n_calls = len(offline_smiles)
    optimizer = torch.optim.Adam(agent.model.network.parameters(), lr=lr)

    # ===== Offline training =====
    print(f'Offline training: {offline_steps} steps...')
    for step in range(offline_steps):
        # Sample batch and compute REINVENT loss
        idx = np.random.choice(len(offline_smiles), size=min(batch_size, len(offline_smiles)), replace=False)
        b_smi = [offline_smiles[i] for i in idx]
        b_r = torch.tensor([np.clip(offline_scores[i], 0, 1) for i in idx], device=device)

        try:
            agent_ll = agent.likelihood_smiles(b_smi)
            prior_ll = prior.likelihood_smiles(b_smi)
            loss = torch.pow(agent_ll - prior_ll + sigma * b_r, 2).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.model.network.parameters(), max_norm=1.0)
            optimizer.step()
        except RuntimeError:
            continue

        # Augmentation rounds
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
                torch.nn.utils.clip_grad_norm_(agent.model.network.parameters(), max_norm=1.0)
                optimizer.step()
            except RuntimeError:
                continue

        if (step + 1) % 500 == 0:
            print(f'  Step {step + 1}/{offline_steps}')

    print('Offline training done. Starting evaluation...')

    # ===== Evaluation =====
    from tdc import Oracle, Evaluator
    oracles = [Oracle(name=n) for n in oracle_names]
    sa_scorer = Oracle(name='SA')
    div_eval = Evaluator(name='Diversity')

    mol_buffer = multi_scores.copy()
    agent.set_mode('inference')

    log_interval = 100
    log_multiple = 1
    while n_calls < oracle_budget:
        batch = agent.sample(batch_size)
        smiles_list = [s for s in batch.items2 if s and Chem.MolFromSmiles(s)]
        valid_new = [s for s in smiles_list if s not in mol_buffer]

        for smi in valid_new:
            if n_calls >= oracle_budget:
                break
            scores = [float(o(smi)) for o in oracles]
            for si, nm in enumerate(oracle_names):
                if nm == 'sa':
                    scores[si] = (10.0 - scores[si]) / 9.0  # normalize SA to [0,1]
            mol_buffer[smi] = scores
            n_calls += 1

        if n_calls > log_interval * log_multiple:
            all_scores = np.array(list(mol_buffer.values()))
            _, pareto = get_pareto_fronts(None, all_scores)
            hv, r2 = 0.0, 0.0
            if len(pareto) > 1:
                hv, r2 = get_hypervolume(None, pareto, n_obj)
                hv, r2 = float(hv), float(r2)

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
                  f'avg_top100: {t100:.3f} | avg_sa: {sa:.3f} | HV: {hv:.6f} | R2: {r2:.6f} | div: {div:.3f}')

            if args.wandb != 'disabled':
                import wandb
                wandb.log({'avg_top1': t1, 'avg_top10': t10, 'avg_top100': t100,
                           'avg_sa': sa, 'diversity_top100': div,
                           'HV': hv, 'R2': r2, 'n_oracle': n_calls})
            log_multiple += 1

    print('Done!')
    if args.wandb != 'disabled':
        import wandb; wandb.finish()


if __name__ == '__main__':
    main()
