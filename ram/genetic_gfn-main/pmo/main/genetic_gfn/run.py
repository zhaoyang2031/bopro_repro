import os
import sys
import numpy as np
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, path_here)
sys.path.append('/'.join(path_here.rstrip('/').split('/')[:-2]))
from main.optimizer import BaseOptimizer
# Use relative imports to avoid conflict with main/utils package
import importlib.util as _iu
_spec = _iu.spec_from_file_location("_gfn_utils", os.path.join(path_here, "utils.py"))
_gfn_utils = _iu.module_from_spec(_spec); _spec.loader.exec_module(_gfn_utils)
Variable = _gfn_utils.Variable
seq_to_smiles = _gfn_utils.seq_to_smiles
unique = _gfn_utils.unique
from model import RNN
from data_structs import Vocabulary, Experience, MolData
from priority_queue import MaxRewardPriorityQueue
import torch
from rdkit import Chem
from tdc import Evaluator
from polyleven import levenshtein

import itertools
import pickle
import pandas as pd
import wandb

from time import perf_counter

from joblib import Parallel


def diversity(smiles):
    dist, normalized = [], []
    for pair in itertools.combinations(smiles, 2):
        dist.append(levenshtein(*pair))
        normalized.append(levenshtein(*pair)/max(len(pair[0]), len(pair[1])))
    evaluator = Evaluator(name = 'Diversity')
    mol_div = evaluator(smiles)
    return np.mean(normalized), np.mean(dist), mol_div


def novelty(new_smiles, ref_smiles):
    smiles_novelty = [min([levenshtein(d, od) for od in ref_smiles]) for d in new_smiles]
    smiles_norm_novelty = [min([levenshtein(d, od) / max(len(d), len(od)) for od in ref_smiles]) for d in new_smiles]
    evaluator = Evaluator(name = 'Novelty')
    mol_novelty = evaluator(new_smiles, ref_smiles)
    return np.mean(smiles_norm_novelty), np.mean(smiles_novelty), mol_novelty


def sanitize(smiles):
    canonicalized = []
    for s in smiles:
        try:
            canonicalized.append(Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True))
        except:
            pass
    return canonicalized


class Genetic_GFN_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "genetic_gfn"

    def _optimize(self, oracle, config):
        import warnings
        warnings.filterwarnings('ignore')

        self.oracle.assign_evaluator(oracle)

        path_here = os.path.dirname(os.path.realpath(__file__))
        voc = Vocabulary(init_from_file=os.path.join(path_here, "data/Voc"))

        Prior = RNN(voc)
        Agent = RNN(voc)

        if torch.cuda.is_available():
            Prior.rnn.load_state_dict(torch.load(os.path.join(path_here, 'data/Prior.ckpt')))
            Agent.rnn.load_state_dict(torch.load(os.path.join(path_here, 'data/Prior.ckpt')))
        else:
            Prior.rnn.load_state_dict(torch.load(os.path.join(path_here, 'data/Prior.ckpt'), map_location=lambda storage, loc: storage))
            Agent.rnn.load_state_dict(torch.load(os.path.join(path_here, 'data/Prior.ckpt'), map_location=lambda storage, loc: storage))

        for param in Prior.rnn.parameters():
            param.requires_grad = False

        # GFN characteristic: fixed log_z (paper L.3: "reduced the logZ value to 0.001")
        log_z = torch.tensor([0.001]).cuda()
        optimizer = torch.optim.Adam([
            {'params': Agent.rnn.parameters(), 'lr': config['learning_rate']},
        ])

        # Load MolStitch offline dataset (shared across all methods for fair comparison)
        offline_data_path = os.path.join(path_here, '..', '..', '..', '..', 'MolStitch-main',
                                          'main', 'offline_cluster', 'data', 'offline_dataset',
                                          'MolStitch_offline_dataset[%d].pt' % self.seed)
        print("Loading offline data from:", offline_data_path)
        offline_dataset = torch.load(offline_data_path, weights_only=False)
        print("Loaded %d offline molecules" % len(offline_dataset))

        obj_names = self.oracle.evaluator.name_list if hasattr(self.oracle.evaluator, 'name_list') else ['qed', 'sa', 'jnk3', 'gsk3b']
        weight_list = self.oracle.evaluator.weight_list if hasattr(self.oracle.evaluator, 'weight_list') else [1.0/len(obj_names)]*len(obj_names)

        # Pre-fill oracle buffer and pre-tokenize offline data for fast uniform sampling
        offline_seqs = []
        offline_rewards = []
        offline_smiles = []  # Store original SMILES for GA crossover
        for smi, data in offline_dataset.items():
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            score = sum(data.get(name, 0.0) * w for name, w in zip(obj_names, weight_list))
            s = np.clip(score, 0, 1)
            self.oracle.mol_buffer[smi] = [score, len(self.oracle.mol_buffer) + 1]
            if hasattr(oracle, 'multi_scores'):
                oracle.multi_scores[smi] = [data.get(name, 0.0) for name in obj_names]
            try:
                offline_seqs.append(torch.tensor(voc.encode(voc.tokenize(smi)), dtype=torch.long))
                offline_rewards.append(s)
                offline_smiles.append(smi)
            except:
                pass

        n_offline = len(offline_seqs)
        print("Oracle buffer: %d molecules, %d tokenized for training" % (len(self.oracle), n_offline))

        def collate_min(seqs):
            max_len = max(s.size(0) for s in seqs)
            batch = torch.zeros(len(seqs), max_len, dtype=torch.long)
            for i, s in enumerate(seqs):
                batch[i, :s.size(0)] = s
            return batch

        self.log_intermediate()

        # ========== Phase 1: Offline training with GFN TB loss (penalty=pb, aligned with paper L.3) ==========
        num_offline_steps = 5000
        sigma = config['beta']              # 500 (Table 17)
        batch_sz = config['batch_size']      # 200 (Table 17)
        exp_replay = config['experience_replay']  # 300 (Table 17)
        aug_rounds = config['experience_loop']    # 8 (Table 17)
        print("Phase 1: Offline GFN training for %d steps (TB loss, penalty=pb, sigma=%d, log_z=%.4f)..." %
              (num_offline_steps, sigma, log_z.item()))

        for step in range(num_offline_steps):
            # GA crossover every 100 steps (GeneticGFN's hallmark: genetic algorithm)
            if step > 0 and step % 100 == 0:
                # Import GA module from ram_repro root
                ga_path = os.path.join(path_here, '..', '..', '..', '..')
                if ga_path not in sys.path:
                    sys.path.insert(0, ga_path)
                from offline_ga import ga_crossover

                # Get top molecules
                top_scores = np.array(offline_rewards)
                top_indices = np.argsort(-top_scores)[:200]
                top_smiles = [offline_smiles[i] for i in top_indices if i < len(offline_smiles)]
                top_rewards = [offline_rewards[i] for i in top_indices]

                # Generate new molecules via GA
                new_smiles, _ = ga_crossover(top_smiles, top_rewards, num_new=20)

                # Score with parent average and add to training data
                for smi in new_smiles:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        continue
                    try:
                        # Score with average of top-10 parents
                        parent_scores = top_rewards[:10]
                        avg_score = np.mean(parent_scores)
                        s = np.clip(avg_score, 0, 1)
                        offline_seqs.append(torch.tensor(voc.encode(voc.tokenize(smi)), dtype=torch.long))
                        offline_rewards.append(s)
                        offline_smiles.append(smi)
                        n_offline = len(offline_seqs)
                    except:
                        pass
                print(f"Step {step}: GA crossover added molecules, dataset size: {n_offline}")

            # Uniform random sample from offline data (aligned with REINVENT4, no rank bias)
            idx = np.random.choice(n_offline, size=min(batch_sz, n_offline), replace=False)
            batch = collate_min([offline_seqs[i] for i in idx]).cuda().long()
            batch_r = torch.tensor([offline_rewards[i] for i in idx]).cuda()

            agent_ll, _ = Agent.likelihood(batch)
            prior_ll, _ = Prior.likelihood(batch)

            # GFN TB loss with penalty=pb:
            #   forward_flow  = agent_ll + log_z     (agent's log-likelihood + learnable constant)
            #   backward_flow = prior_ll + sigma * r  (prior's log-likelihood + reward bonus)
            #   loss = (forward - backward)^2
            # Target: agent_ll ≈ prior_ll + sigma*r - log_z  (higher reward → higher log-likelihood ✓)
            # This is mathematically equivalent to REINVENT loss (agent_ll - prior_ll - sigma*r)² when log_z ≈ 0.
            forward_flow = agent_ll + log_z
            backward_flow = prior_ll + sigma * batch_r
            loss = torch.pow(forward_flow - backward_flow, 2).mean()

            optimizer.zero_grad()
            loss.backward()
            # No gradient clipping (matching pure GeneticGFN code)
            optimizer.step()

            # Augmentation rounds on different data subsets (Table 17: 8 rounds × 300 samples)
            # Use rank-based sampling (matching pure GeneticGFN) to favor high-reward molecules
            rank_coeff = config.get('rank_coefficient', 0.01)
            for _ in range(aug_rounds):
                if rank_coeff > 0:
                    scores_np = np.array([offline_rewards[i] for i in range(n_offline)]) + 1e-10
                    ranks = np.argsort(np.argsort(-1 * scores_np))
                    weights = 1.0 / (rank_coeff * len(scores_np) + ranks)
                    sampler = torch.utils.data.WeightedRandomSampler(weights, min(exp_replay, n_offline), replacement=True)
                    aidx = list(sampler)
                else:
                    aidx = np.random.choice(n_offline, size=min(exp_replay, n_offline), replace=False)

                aug_batch = collate_min([offline_seqs[i] for i in aidx]).cuda().long()
                aug_r = torch.tensor([offline_rewards[i] for i in aidx]).cuda()

                a_agent_ll, _ = Agent.likelihood(aug_batch)
                a_prior_ll, _ = Prior.likelihood(aug_batch)

                f_flow = a_agent_ll + log_z
                b_flow = a_prior_ll + sigma * aug_r
                aug_loss = torch.pow(f_flow - b_flow, 2).mean()

                optimizer.zero_grad()
                aug_loss.backward()
                # No gradient clipping (matching pure GeneticGFN code)
                optimizer.step()

            if (step + 1) % 500 == 0:
                print("Offline step %d/%d, log_z=%.4f" % (step + 1, num_offline_steps, log_z.item()))
                self.log_intermediate()

        print("Phase 1 complete. Oracle calls: %d/%d" % (len(self.oracle), self.oracle.max_oracle_calls))

        # ========== Phase 2: Evaluation only (NO training, NO GA, NO oracle queries beyond evaluation) ==========
        print("Phase 2: Evaluation-only sampling...")

        Agent.rnn.eval()

        while not self.finish:
            seqs, agent_likelihood, entropy = Agent.sample(config['batch_size'])
            unique_idxs = unique(seqs)
            seqs = seqs[unique_idxs]
            smiles = seq_to_smiles(seqs, voc)
            if config.get('valid_only', True):
                smiles = sanitize(smiles)

            score = np.array(self.oracle(smiles))

            if self.finish:
                break

        self.log_intermediate(finish=True)
        print("Evaluation complete. Total oracle calls: %d" % len(self.oracle))
