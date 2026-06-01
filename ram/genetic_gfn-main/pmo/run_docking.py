"""GFN (GFlowNets) full-offline docking for parp1/fa7/jak2/braf/5ht1b x 10 seeds.
Same protocol as REINVENT4 run_docking.py.
"""
import os, sys, argparse, torch, numpy as np, random
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(path_here, 'main', 'genetic_gfn'))

from model import RNN
from data_structs import Vocabulary
from utils import Variable, seq_to_smiles, unique

sys.path.insert(0, '/data/xk/zhaoyang/ram_repro/MolStitch-main')
from evaluators.hypervolume import get_hypervolume, get_pareto_fronts
from botorch.utils.multi_objective.hypervolume import Hypervolume

PRIOR_PATH = os.path.join(path_here, 'main', 'genetic_gfn', 'data', 'Prior.ckpt')
VOC_PATH = os.path.join(path_here, 'main', 'genetic_gfn', 'data', 'Voc')
OFFLINE_DIR = '/data/xk/zhaoyang/ram_repro/MolStitch-main/main/offline_cluster/data/offline_dataset'

DOCKING_TARGETS = ['parp1', 'fa7', 'jak2', 'braf', '5ht1b']

TASK_CFG = {}
for dt in DOCKING_TARGETS:
    TASK_CFG[f'dock_{dt}'] = {
        'names': [dt, 'qed', 'sa'],
        'keys': [dt, 'qed', 'sa'],
        'w': [1/3, 1/3, 1/3],
    }


def sanitize(smiles):
    canonicalized = []
    for s in smiles:
        try:
            canonicalized.append(Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True))
        except:
            pass
    return canonicalized


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


def collate_min(seqs):
    max_len = max(s.size(0) for s in seqs)
    batch = torch.zeros(len(seqs), max_len, dtype=torch.long)
    for i, s in enumerate(seqs):
        batch[i, :s.size(0)] = s
    return batch


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
        name = args.run_name or f'GFN_{args.task}_seed{seed}'
        wandb.init(project='repro_ram', entity='1585515136-', name=name, reinit=True)

    cfg = TASK_CFG[args.task]
    oracle_names = cfg['names']
    data_keys = cfg['keys']
    obj_weights = cfg['w']
    n_obj = len(obj_weights)

    # Load GFN model (RNN backbone)
    print(f"Loading GFN prior from: {PRIOR_PATH}")
    voc = Vocabulary(init_from_file=VOC_PATH)
    Prior = RNN(voc)
    Agent = RNN(voc)

    if torch.cuda.is_available():
        Prior.rnn.load_state_dict(torch.load(PRIOR_PATH))
        Agent.rnn.load_state_dict(torch.load(PRIOR_PATH))
    else:
        Prior.rnn.load_state_dict(torch.load(PRIOR_PATH, map_location=lambda storage, loc: storage))
        Agent.rnn.load_state_dict(torch.load(PRIOR_PATH, map_location=lambda storage, loc: storage))

    for param in Prior.rnn.parameters():
        param.requires_grad = False

    # GFN: learnable log_z (paper L.3: log_z=0.001, lr=5e-4)
    log_z = torch.nn.Parameter(torch.tensor([0.001]).cuda())
    optimizer = torch.optim.Adam([
        {'params': Agent.rnn.parameters(), 'lr': 5e-4},
        {'params': log_z, 'lr': 5e-4}
    ])

    # Load offline data
    offline_path = f'{OFFLINE_DIR}/MolStitch_offline_dataset[{seed}].pt'
    offline_data = torch.load(offline_path, weights_only=False)
    offline_seqs = []
    offline_rewards = []
    multi_scores = {}
    skipped = 0
    for smi, data in offline_data.items():
        if len(offline_seqs) >= args.offline_limit:
            break
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        Chem.RemoveStereochemistry(mol)
        smi_clean = Chem.MolToSmiles(mol, isomericSmiles=False)
        scores = [normalize_offline_score(k, data.get(k, 0.0)) for k in data_keys]
        multi_scores[smi_clean] = scores
        reward = np.clip(sum(s * w for s, w in zip(scores, obj_weights)), 0, 1)
        try:
            tokens = voc.tokenize(smi_clean)
            encoded = voc.encode(tokens)
            offline_seqs.append(torch.tensor(encoded, dtype=torch.long))
            offline_rewards.append(reward)
        except:
            skipped += 1
    print(f'Loaded {len(offline_seqs)} offline molecules (skipped {skipped})')

    n_offline = len(offline_seqs)
    Agent.rnn.train()

    sigma = 500.0
    batch_size = 200
    exp_replay_size = 300
    aug_rounds = 8
    offline_steps = 5000
    oracle_budget = args.max_oracle_calls
    n_calls = n_offline

    # ===== Offline training (GFN TB loss with penalty=pb, equivalent to REINVENT loss) =====
    print(f'Offline training: {offline_steps} steps (TB loss, penalty=pb, sigma={sigma}, log_z={log_z.item():.4f})...')
    for step in range(offline_steps):
        idx = np.random.choice(n_offline, size=min(batch_size, n_offline), replace=False)
        batch = collate_min([offline_seqs[i] for i in idx]).cuda().long()
        batch_r = torch.tensor([offline_rewards[i] for i in idx]).cuda()

        agent_ll, _ = Agent.likelihood(batch)
        prior_ll, _ = Prior.likelihood(batch)

        forward_flow = agent_ll + log_z
        backward_flow = prior_ll + sigma * batch_r
        loss = torch.pow(forward_flow - backward_flow, 2).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(Agent.rnn.parameters(), max_norm=1.0)
        optimizer.step()

        for _ in range(aug_rounds):
            aidx = np.random.choice(n_offline, size=min(exp_replay_size, n_offline), replace=False)
            aug_batch = collate_min([offline_seqs[i] for i in aidx]).cuda().long()
            aug_r = torch.tensor([offline_rewards[i] for i in aidx]).cuda()

            a_agent_ll, _ = Agent.likelihood(aug_batch)
            a_prior_ll, _ = Prior.likelihood(aug_batch)

            f_flow = a_agent_ll + log_z
            b_flow = a_prior_ll + sigma * aug_r
            aug_loss = torch.pow(f_flow - b_flow, 2).mean()

            optimizer.zero_grad()
            aug_loss.backward()
            torch.nn.utils.clip_grad_norm_(Agent.rnn.parameters(), max_norm=1.0)
            optimizer.step()

        if (step + 1) % 500 == 0:
            print(f'  Step {step + 1}/{offline_steps}, log_z={log_z.item():.4f}')

    print('Offline training done. Starting evaluation...')

    # ===== Evaluation =====
    oracles = make_oracles(oracle_names)
    from tdc import Evaluator, Oracle
    sa_scorer = Oracle(name='SA')
    div_eval = Evaluator(name='Diversity')

    mol_buffer = multi_scores.copy()
    Agent.rnn.eval()

    log_interval = 100
    log_multiple = 1
    while n_calls < oracle_budget:
        seqs, agent_likelihood, entropy = Agent.sample(batch_size)
        unique_idxs = unique(seqs)
        seqs = seqs[unique_idxs]
        smiles_list = seq_to_smiles(seqs, voc)
        smiles_list = sanitize(smiles_list)
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
