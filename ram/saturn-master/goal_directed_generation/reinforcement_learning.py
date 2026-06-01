"""
Adapted from https://github.com/MolecularAI/Reinvent with code additions for:
    1. Augmented Memory: https://pubs.acs.org/doi/10.1021/jacsau.4c00066
    2. Beam Enumeration: https://openreview.net/forum?id=7UhxsmbdaQ
    3. Hallucinated Memory (GraphGA based: https://pubs.rsc.org/en/content/articlelanding/2019/sc/c8sc05372c)

Modifications for full-offline MBO:
    - Added offline_training phase using pre-computed scores from experience replay buffer.
"""
from typing import Tuple
import os
import logging
import time
import torch
import numpy as np
import pandas as pd
from rdkit import Chem

import utils.chemistry_utils as chemistry_utils
from goal_directed_generation.utils import sample_unique_sequences
from utils.utils import to_tensor, setup_logging

from oracles.oracle import Oracle
from goal_directed_generation.dataclass import GoalDirectedGenerationConfiguration

from models.generator import Generator
from experience_replay.replay_buffer import ReplayBuffer
from diversity_filter.diversity_filter import DiversityFilter
from hallucinated_memory.utils import initialize_hallucinator
from beam_enumeration.beam_enumeration import BeamEnumeration

# Syntheseus oracle for custom results write-out
try:
    from oracles.synthesizability.syntheseus import Syntheseus
except ImportError:
    Syntheseus = None



class ReinforcementLearningAgent:
    """
    RL agent for goal-directed generation.
    Supports full-offline mode via offline_dataset parameter.
    """
    def __init__(
        self,
        logging_frequency: int,
        logging_path: str,
        model_checkpoints_dir: str,
        oracle: Oracle,
        configuration: GoalDirectedGenerationConfiguration,
        device: str,
        offline_dataset: dict = None,
    ):
        self.prior = Generator.load_from_file(configuration.reinforcement_learning.prior, device)
        # Prior model is not updated so disable gradients
        self._disable_prior_gradients()
        self.agent = Generator.load_from_file(configuration.reinforcement_learning.agent, device)
        self.device = self.agent.device
        # In case the Agent is to be trained on CPU, move also the Prior to CPU to avoid tensors on different devices
        self.prior.network.to(self.device)

        # Seed for documentation
        self.seed = configuration.seed

        # Oracle
        self.oracle = oracle

        # RL parameters
        self.batch_size = configuration.reinforcement_learning.batch_size
        self.learning_rate = configuration.reinforcement_learning.learning_rate
        self.sigma = configuration.reinforcement_learning.sigma
        self.augmented_memory = configuration.reinforcement_learning.augmented_memory
        self.augmentation_rounds = configuration.reinforcement_learning.augmentation_rounds
        self.selective_memory_purge = configuration.reinforcement_learning.selective_memory_purge

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(parameters=configuration.experience_replay)

        # Seed the Replay Buffer (if applicable)
        self.oracle = self.replay_buffer.prepopulate_buffer(self.oracle)

        # Diversity Filter
        self.diversity_filter = DiversityFilter(configuration.diversity_filter)

        # Hallucinated Memory
        self.execute_hallucinated_memory = configuration.hallucinated_memory.execute_hallucinated_memory
        self.hallucinator = initialize_hallucinator(
            prior=self.prior,
            parameters=configuration.hallucinated_memory
        )

        # Beam Enumeration
        self.execute_beam_enumeration = configuration.beam_enumeration.execute_beam_enumeration
        self.beam_enumeration = BeamEnumeration(
            k=configuration.beam_enumeration.beam_k,
            beam_steps=configuration.beam_enumeration.beam_steps,
            substructure_type=configuration.beam_enumeration.substructure_type.lower(),
            substructure_min_size=configuration.beam_enumeration.structure_min_size,
            pool_size=configuration.beam_enumeration.pool_size,
            pool_saving_frequency=configuration.beam_enumeration.pool_saving_frequency,
            patience=configuration.beam_enumeration.patience,
            token_sampling_method=configuration.beam_enumeration.token_sampling_method,
            filter_patience_limit=configuration.beam_enumeration.filter_patience_limit
        )

        # Only the Agent is updated so the Prior does not need an optimizer
        self.optimizer = torch.optim.Adam(self.agent.get_network_parameters(), lr=self.learning_rate)

        # Model checkpointing save directory
        self.model_checkpoints_dir = model_checkpoints_dir
        os.makedirs(self.model_checkpoints_dir, exist_ok=True)
        self.logging_path = logging_path
        self.logging_frequency = logging_frequency
        self.logging_multiple = 1

        # Best Agent checkpointing
        self.best_agent_reward = float("-inf")
        self.patience = 0

        # Set up logging
        setup_logging(logging_path)

        # Offline dataset (for full-offline mode)
        self.offline_dataset = offline_dataset

    def _prepare_offline_data(self):
        """Load offline dataset and pre-fill the replay buffer with pre-computed scores."""
        if self.offline_dataset is None:
            return

        print(f"Preparing offline data: {len(self.offline_dataset)} molecules")

        # Map oracle components to data keys in the offline dataset
        data_keys = []
        for oc in self.oracle.oracle:
            name = oc.name
            if name.startswith("tdc_"):
                data_keys.append(oc.specific_parameters.get("target", name))
            elif name == "sa_score":
                data_keys.append("sa")
            else:
                data_keys.append(name)
        obj_weights = self.oracle.oracle_weights

        # Build (smiles, weighted_score) pairs from offline data
        # Store ALL molecules for offline training (no purge -> preserves diversity)
        self.offline_smiles = []
        self.offline_rewards = []
        # multi_scores dict for HV/R2 (like GeneticGFN stores individual objective scores)
        self.multi_scores = {}
        for smi, data in self.offline_dataset.items():
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            total = 0.0
            scores = []
            for key, weight in zip(data_keys, obj_weights):
                score = data.get(key, 0.0)
                scores.append(score)
                total += score * weight
            self.offline_smiles.append(smi)
            self.offline_rewards.append(total)
            self.multi_scores[smi] = scores

        # Fill replay buffer (keep all for offline training, purge happens during online phase)
        df = pd.DataFrame({"smiles": self.offline_smiles, "reward": self.offline_rewards})
        self.replay_buffer.memory = df
        # Offline data = 5000 oracle calls (protocol)
        self.oracle.calls = len(self.offline_smiles)
        # Populate oracle cache so regenerated offline molecules are NOT re-evaluated
        self.oracle.update_oracle_cache(
            smiles=np.array(self.offline_smiles),
            rewards=np.array(self.offline_rewards),
        )
        # Align logging_multiple so first log triggers after logging_frequency NEW oracle calls
        self.logging_multiple = (self.oracle.calls // self.logging_frequency) + 1
        print(f"Offline data: {len(self.offline_smiles)} molecules loaded, oracle calls: {self.oracle.calls}")

    def _offline_step(self, step=0):
        """Single offline training step: main loss + 8 augmentation rounds + GA crossover.
        Aligned with pure Saturn's online loop (steps 12-17) adapted for offline."""
        if len(self.replay_buffer.memory) == 0:
            return

        # GA crossover every 100 steps (Saturn's hallmark: genetic algorithm)
        if step > 0 and step % 100 == 0:
            self._ga_crossover_step()

        # Main step: sample from replay buffer (replaces agent sampling + oracle call)
        smiles, rewards = self.replay_buffer.sample_memory()
        if len(smiles) == 0:
            return
        rewards = np.clip(rewards.astype(float), 0, 1)

        # Compute main loss (step 12 in pure Saturn)
        loss = self.compute_loss(smiles, rewards)

        # Add experience replay loss (steps 14-16 in pure Saturn)
        er_smiles, er_rewards = self.replay_buffer.sample_memory()
        if len(er_smiles) > 0:
            er_rewards = np.clip(er_rewards.astype(float), 0, 1)
            er_loss = self.compute_loss(er_smiles, er_rewards)
            loss = torch.cat((loss, er_loss), 0)

        self.backpropagate(loss)

        # Augmentation rounds (step 17 in pure Saturn)
        if self.augmentation_rounds > 0 and len(self.replay_buffer.memory) > 0:
            for _ in range(self.augmentation_rounds):
                # Randomize SMILES from current batch (Saturn's Augmented Memory)
                randomized_smiles = chemistry_utils.randomize_smiles_batch(smiles, self.prior)
                aug_loss = self.compute_loss(randomized_smiles, rewards)

                # Also sample from buffer and randomize
                aug_smiles, aug_rewards = self.replay_buffer.augmented_memory_replay(self.prior)
                if len(aug_smiles) > 0:
                    n = min(len(aug_smiles), self.replay_buffer.sample_size)
                    indices = np.random.choice(len(aug_smiles), size=n, replace=False)
                    aug_smiles = aug_smiles[indices]
                    aug_rewards = np.clip(aug_rewards[indices].astype(float), 0, 1)
                    aug_randomized = chemistry_utils.randomize_smiles_batch(aug_smiles, self.prior)
                    aug_loss2 = self.compute_loss(aug_randomized, aug_rewards)
                    aug_loss = torch.cat((aug_loss, aug_loss2), 0)

                self.backpropagate(aug_loss)

    def _ga_crossover_step(self):
        """GA crossover: generate new molecules from top offline molecules.
        In offline setting, score with parent average (no oracle calls)."""
        import sys
        ga_path = os.path.join(os.path.dirname(__file__), '..', '..')
        if ga_path not in sys.path:
            sys.path.insert(0, ga_path)
        from offline_ga import ga_crossover

        # Get top molecules from replay buffer
        if len(self.replay_buffer.memory) == 0:
            return
        mem = self.replay_buffer.memory.sort_values("reward", ascending=False)
        top_n = min(200, len(mem))
        top_smiles = mem.head(top_n)["smiles"].values.tolist()
        top_rewards = mem.head(top_n)["reward"].values.tolist()

        # Generate new molecules via GA crossover
        new_smiles, _ = ga_crossover(top_smiles, top_rewards, num_new=20)

        if len(new_smiles) == 0:
            return

        # Score GA molecules with parent average scores
        new_rewards = []
        for smi in new_smiles:
            # Find parents and average their scores
            parent_scores = []
            for ps, pr in zip(top_smiles, top_rewards):
                if ps in [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in top_smiles[:10]]:
                    parent_scores.append(pr)
            if parent_scores:
                new_rewards.append(np.mean(parent_scores[:5]))
            else:
                new_rewards.append(np.mean(top_rewards[:10]))

        # Add to replay buffer
        new_df = pd.DataFrame({"smiles": new_smiles, "reward": new_rewards})
        self.replay_buffer.memory = pd.concat([self.replay_buffer.memory, new_df], ignore_index=True)
        logging.info(f"GA crossover: added {len(new_smiles)} new molecules (buffer size: {len(self.replay_buffer.memory)})")

    def run(self):
        start_time = time.perf_counter()
        logging.info(f"Starting RL generative experiment with oracle budget: {self.oracle.budget}")

        # Full-offline training phase: 5000 steps on offline data
        if self.offline_dataset is not None:
            self._prepare_offline_data()
            offline_steps = 5000
            logging.info(f"Starting offline training for {offline_steps} steps...")
            for step in range(offline_steps):
                self._offline_step(step=step)
                if (step + 1) % 500 == 0:
                    logging.info(f"Offline training step {step + 1}/{offline_steps}")
            logging.info("Offline training complete. Starting evaluation...")

        # Main evaluation loop (evaluation only, NO training)
        while not self.oracle.budget_exceeded():
            # 1. Sample unique SMILES from the Agent
            seqs, smiles, _ = sample_unique_sequences(self.agent, self.batch_size)

            # 2. Compute Validity and guard against Agent drift leading to invalid SMILES
            reset, validity = self._validity_drift_guard(smiles)
            if reset:
                continue

            logging.info(f"Oracle calls: {self.oracle.calls}/{self.oracle.budget} (Batch Validity: {round(validity * 100, 2)}%, {int(validity * len(smiles))} / {len(smiles)})")

            # 3. Remove molecules with radicals
            smiles = chemistry_utils.remove_molecules_with_radicals(smiles)
            if len(smiles) == 0:
                logging.info("No valid SMILES in this batch. Generating a new batch.")
                continue

            # 4. Oracle call (evaluation only, NO training: no loss, no backprop, no augmented memory, no hallucinated memory, no beam enumeration)
            smiles, penalized_rewards = self.oracle(smiles, self.diversity_filter)

            # 5. Update Replay Buffer (for result tracking, not training)
            self.replay_buffer.add(
                smiles=smiles,
                rewards=penalized_rewards
            )

            # 6. Intermediate results write-out
            if self.oracle.calls > self.logging_frequency * self.logging_multiple:
                logging.info(f"Logging intermediate results at {self.oracle.calls} oracle calls.")
                self._write_out_results()
                self.agent.save(os.path.join(self.model_checkpoints_dir, f"{self.agent.model_architecture}_{self.oracle.calls}_agent.ckpt"))
                self.logging_multiple += 1

            # 7. Checkpoint best Agent (by average reward)
            if (np.mean(penalized_rewards) > self.best_agent_reward) and (validity > 0.5):
                self.best_agent_reward = np.mean(penalized_rewards)
                self.agent.save(os.path.join(self.model_checkpoints_dir, "best_agent.ckpt"))

            # 8. Respect oracle budget precisely
            if self.oracle.calls >= self.oracle.budget:
                break

        # End of run: save final agent and write out results
        self.agent.save(os.path.join(self.model_checkpoints_dir, "final_agent.ckpt"))
        logging.info(f"End of run, logged oracle calls: {self.oracle.calls}")
        self._write_out_results(end_of_run=True)
        elapsed_time = time.perf_counter() - start_time
        logging.info(f"Total elapsed time: {elapsed_time / 60:.2f} minutes")

    def compute_loss(self, smiles, rewards):
        """Compute the REINVENT loss for a batch of SMILES."""
        if len(smiles) == 0:
            return torch.tensor([0.0], device=self.device)

        agent_ll = self.agent.likelihood_smiles(smiles)
        prior_ll = self.prior.likelihood_smiles(smiles)

        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        # REINVENT loss: (agent_NLL - prior_NLL + sigma * reward)^2
        loss = torch.pow(agent_ll - prior_ll + self.sigma * rewards, 2).mean(dim=0, keepdim=True)
        return loss

    def backpropagate(self, loss):
        """Update the agent using the computed loss."""
        self.optimizer.zero_grad()
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(self.agent.get_network_parameters(), max_norm=1.0)
        self.optimizer.step()

    def _disable_prior_gradients(self):
        """Disable gradients for the Prior model."""
        for param in self.prior.network.parameters():
            param.requires_grad = False

    def _validity_drift_guard(self, smiles):
        """Check if the Agent's validity has drifted too far from the expected range.
        Uses patience-based detection matching pure Saturn code."""
        valid_smiles = [smi for smi in smiles if Chem.MolFromSmiles(smi) is not None]
        valid_count = len(valid_smiles)
        total_count = len(smiles) if len(smiles) > 0 else 1
        if total_count == 0:
            return False, 0.0
        validity = valid_count / total_count

        if validity == 0.0:
            if self.patience == 10:
                logging.info("Resetting to best Agent checkpoint.")
                self._reset_agent()
                self.patience = 0
                return True, validity
            else:
                self.patience += 1
                return False, validity
        else:
            self.patience = 0
            return False, validity

    def _reset_agent(self):
        """Reset the Agent to the best checkpoint."""
        best_path = os.path.join(self.model_checkpoints_dir, "best_agent.ckpt")
        if os.path.exists(best_path):
            self.agent = Generator.load_from_file(best_path, self.device)
        else:
            self.agent.network.load_state_dict(self.prior.network.state_dict())

    def _update_multi_scores(self):
        """Sync individual objective scores from oracle_history to multi_scores.
        Normalizes SA scores from raw [1,10] to [0,1] for HV/R2 consistency."""
        if not hasattr(self, 'multi_scores'):
            self.multi_scores = {}
        try:
            df = self.oracle.oracle_history
            smiles_col = df['smiles'].values
            raw_cols = [c for c in df.columns if c.endswith('_raw_values')]
            if raw_cols:
                scores_arr = df[raw_cols].values.astype(float)
                # Normalize SA: oracle returns raw [1,10], multi_scores expects [0,1]
                for j, col in enumerate(raw_cols):
                    if col.startswith('sa_score_'):
                        scores_arr[:, j] = (10.0 - scores_arr[:, j]) / 9.0
                for i, smi in enumerate(smiles_col):
                    if smi and smi not in self.multi_scores:
                        self.multi_scores[smi] = scores_arr[i].tolist()
        except Exception:
            pass

    def _write_out_results(self, end_of_run=False):
        """Write out intermediate results with HV/R2 and wandb logging (aligned with GeneticGFN)."""
        import json
        from tdc import Oracle, Evaluator

        n_calls = self.oracle.calls

        # Update multi_scores from latest oracle calls
        self._update_multi_scores()

        # Top SMILES from replay buffer
        top_smiles = []
        top_scores = []
        if len(self.replay_buffer.memory) > 0:
            mem = self.replay_buffer.memory.sort_values("reward", ascending=False)
            top100 = mem.head(100)
            top_smiles = top100["smiles"].values.tolist()
            top_scores = top100["reward"].values.tolist()

        avg_top1 = float(np.max(top_scores)) if top_scores else 0.0
        avg_top10 = float(np.mean(sorted(top_scores, reverse=True)[:10])) if top_scores else 0.0
        avg_top100 = float(np.mean(top_scores)) if top_scores else 0.0

        # SA from top-100
        avg_sa = 0.0
        if top_smiles:
            try:
                sa_scorer = Oracle(name='SA')
                avg_sa = float(np.mean([sa_scorer(s) for s in top_smiles[:100]]))
            except Exception:
                pass

        # Diversity
        diversity = 0.0
        if top_smiles:
            try:
                diversity_eval = Evaluator(name='Diversity')
                diversity = float(diversity_eval(top_smiles[:100]))
            except Exception:
                pass

        # HV/R2 from multi_scores (all molecules: offline + evaluation)
        hv, r2 = 0.0, 0.0
        try:
            if hasattr(self, 'multi_scores') and len(self.multi_scores) > 1:
                all_scores = np.array(list(self.multi_scores.values()))
                num_obj = all_scores.shape[1]
                import sys as _sys
                _ms = '/data/xk/zhaoyang/ram_repro/MolStitch-main'
                if _ms not in _sys.path:
                    _sys.path.insert(0, _ms)
                from evaluators.hypervolume import get_hypervolume, get_pareto_fronts
                _, pareto = get_pareto_fronts(None, all_scores)
                if len(pareto) > 0:
                    hv, r2 = get_hypervolume(None, pareto, num_obj)
                    hv, r2 = float(hv), float(r2)
        except Exception as e:
            logging.warning(f"HV/R2 error: {e}")

        print(f'{n_calls}/{self.oracle.budget} | '
              f'avg_top1: {avg_top1:.3f} | avg_top10: {avg_top10:.3f} | avg_top100: {avg_top100:.3f} | '
              f'avg_sa: {avg_sa:.3f} | HV: {hv:.6f} | R2: {r2:.6f} | div: {diversity:.3f}')

        try:
            import wandb
            wandb.log({
                "avg_top1": avg_top1, "avg_top10": avg_top10, "avg_top100": avg_top100,
                "avg_sa": avg_sa, "diversity_top100": diversity,
                "HV": hv, "R2": r2, "n_oracle": n_calls,
            })
        except Exception as e:
            logging.warning(f"wandb.log failed: {e}")

        results = {"oracle_calls": n_calls, "HV": hv, "R2": r2}
        with open(os.path.join(self.model_checkpoints_dir, "results.json"), 'w') as f:
            json.dump(results, f)

        if end_of_run:
            self.oracle.oracle_history.to_csv(
                os.path.join(self.model_checkpoints_dir, "oracle_history.csv"), index=False)
