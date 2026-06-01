from __future__ import print_function

import argparse
import yaml
import os
import sys
import numpy as np
from tdc import Oracle
from rdkit import Chem
from time import time

# Add pmo/ to sys.path ONLY when needed for method imports
_pmo_dir = os.path.dirname(os.path.abspath(__file__))


class WeightedMultiOracle:
    """Weighted scalarization of multiple TDC oracles for multi-objective optimization."""
    def __init__(self, oracle_names, weights=None, oracle_instances=None):
        self.name_list = oracle_names
        if weights is None:
            weights = [1.0] * len(oracle_names)
        total = sum(weights)
        self.weight_list = [w / total for w in weights]
        if oracle_instances is not None:
            self.oracle_list = oracle_instances
        else:
            self.oracle_list = [Oracle(name=n) for n in oracle_names]
        self.name = "+".join(f"{n}:{w}" for n, w in zip(oracle_names, self.weight_list))
        self.multi_scores = {}

    def __call__(self, smiles_lst):
        if isinstance(smiles_lst, str):
            smiles_lst = [smiles_lst]
            single = True
        else:
            single = False
        results = []
        for smi in smiles_lst:
            total = 0.0
            obj_scores = []
            for oracle, weight in zip(self.oracle_list, self.weight_list):
                score = oracle(smi)
                name = oracle.name.lower()
                if name == "sa":
                    score = (10 - score) / 9
                obj_scores.append(score)
                total += score * weight
            mol = Chem.MolFromSmiles(smi)
            canon = Chem.MolToSmiles(mol) if mol else smi
            self.multi_scores[canon] = obj_scores
            results.append(total)
        return results[0] if single else results

    def get_hv_r2(self):
        if len(self.multi_scores) < 2:
            return 0.0, 0.0
        try:
            import sys as _sys
            _molstitch = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'MolStitch-main')
            if _molstitch not in _sys.path:
                _sys.path.insert(0, _molstitch)
            from evaluators.hypervolume import get_hypervolume, get_pareto_fronts
            all_scores = np.array(list(self.multi_scores.values()))
            num_obj = all_scores.shape[1]
            _, pareto = get_pareto_fronts(None, all_scores)
            if len(pareto) > 0:
                hv, r2 = get_hypervolume(None, pareto, num_obj)
                return float(hv), float(r2)
            return 0.0, 0.0
        except Exception as e:
            print(f"HV computation error: {e}")
            return 0.0, 0.0


def main():
    start_time = time()

    parser = argparse.ArgumentParser()
    parser.add_argument('method', default='graph_ga')
    parser.add_argument('--smi_file', default=None)
    parser.add_argument('--config_default', default='hparams_default.yaml')
    parser.add_argument('--config_tune', default='hparams_tune.yaml')
    parser.add_argument('--pickle_directory', help='Directory containing pickle files', default=None)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--max_oracle_calls', type=int, default=10000)
    parser.add_argument('--freq_log', type=int, default=100)
    parser.add_argument('--n_runs', type=int, default=5)
    parser.add_argument('--seed', type=int, nargs="+", default=[0])
    parser.add_argument('--task', type=str, default="simple", choices=["tune", "simple", "production"])
    parser.add_argument('--oracles', nargs="+", default=["QED"])
    parser.add_argument('--log_results', action='store_true')
    parser.add_argument('--log_code', action='store_true')
    parser.add_argument('--wandb', type=str, default="disabled", choices=["online", "offline", "disabled"])
    parser.add_argument('--run_name', type=str, default="default")
    args = parser.parse_args()

    os.environ["WANDB_MODE"] = args.wandb
    if not args.log_code:
        os.environ["WANDB_DISABLE_CODE"] = "false"

    args.method = args.method.lower()
    path_main = os.path.join(_pmo_dir, "main", args.method)

    # Add pmo/ to sys.path for method imports
    if _pmo_dir not in sys.path:
        sys.path.insert(0, _pmo_dir)
    sys.path.append(path_main)

    print(args.method)
    if args.method == 'graph_ga':
        from main.graph_ga.run import GB_GA_Optimizer as Optimizer
    elif args.method == 'smiles_ga':
        from main.smiles_ga.run import SMILES_GA_Optimizer as Optimizer
    elif args.method == "selfies_ga":
        from main.selfies_ga.run import SELFIES_GA_Optimizer as Optimizer
    elif args.method == "genetic_gfn":
        from main.genetic_gfn.run import Genetic_GFN_Optimizer as Optimizer
    elif args.method == 'reinvent':
        from main.reinvent.run import REINVENT_Optimizer as Optimizer
    elif args.method == "mol_ga":
        from main.mol_ga.run import MolGAOptimizer as Optimizer
    elif args.method == "gflownet":
        from main.gflownet.run import GFlowNet_Optimizer as Optimizer
    elif args.method == "gflownet_al":
        from main.gflownet_al.run import GFlowNet_AL_Optimizer as Optimizer
    elif args.method == 'selfies_lstm_hc':
        from main.selfies_lstm_hc.run import SELFIES_LSTM_HC_Optimizer as Optimizer
    elif args.method == "smiles_lstm_hc":
        from main.smiles_lstm_hc.run import SMILES_LSTM_HC_Optimizer as Optimizer
    elif args.method == 'gegl':
        from main.gegl.run import GEGL_Optimizer as Optimizer
    elif args.method == 'gpbo':
        from main.gpbo.run import GPBO_Optimizer as Optimizer
    elif args.method == 'stoned':
        from main.stoned.run import Stoned_Optimizer as Optimizer
    elif args.method == "synnet":
        from main.synnet.run import SynNet_Optimizer as Optimizer
    elif args.method == 'reinvent_selfies':
        from main.reinvent_selfies.run import REINVENT_SELFIES_Optimizer as Optimizer
    elif args.method == "genetic_gfn_selfies":
        from main.genetic_gfn_selfies.run import Genetic_GFN_SELFIES_Optimizer as Optimizer
    elif args.method == "genetic_gfn_al":
        from main.genetic_gfn_al.run import Genetic_GFN_AL_Optimizer as Optimizer
    elif args.method == "reinvent_ls_gfn":
        from main.genetic_gfn.run_ls_gfn import REINVENT_LS_GFN_Optimizer as Optimizer
        path_main = os.path.join(_pmo_dir, "main", "genetic_gfn")
    else:
        raise ValueError("Unrecognized method name.")

    if args.output_dir is None:
        args.output_dir = os.path.join(path_main, "results")
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if args.pickle_directory is None:
        args.pickle_directory = path_main

    if args.task != "tune":
        print(f'Optimizing oracle functions: {args.oracles}')
        try:
            config_default = yaml.safe_load(open(args.config_default))
        except:
            config_default = yaml.safe_load(open(os.path.join(path_main, args.config_default)))

        # Create TDC oracle instances BEFORE sys.path modification
        if len(args.oracles) == 1:
            oracle = Oracle(name=args.oracles[0])
        else:
            oracle_instances = [Oracle(name=n) for n in args.oracles]
            # NOW add pmo/ to sys.path for method imports
            if _pmo_dir not in sys.path:
                sys.path.insert(0, _pmo_dir)
            oracle = WeightedMultiOracle(args.oracles, oracle_instances=oracle_instances)
        optimizer = Optimizer(args=args)

        if args.task == "simple":
            for seed in args.seed:
                print('seed', seed)
                optimizer.optimize(oracle=oracle, config=config_default, seed=seed)
        elif args.task == "production":
            optimizer.production(oracle=oracle, config=config_default, num_runs=args.n_runs)
        else:
            raise ValueError('Unrecognized task name.')

    elif args.task == "tune":
        print(f'Tuning hyper-parameters on tasks: {args.oracles}')
        try:
            config_default = yaml.safe_load(open(args.config_default))
        except:
            config_default = yaml.safe_load(open(os.path.join(path_main, args.config_default)))
        try:
            config_tune = yaml.safe_load(open(args.config_tune))
        except:
            config_tune = yaml.safe_load(open(os.path.join(path_main, args.config_tune)))
        oracles = [Oracle(name=oracle_name) for oracle_name in args.oracles]
        optimizer = Optimizer(args=args)
        optimizer.hparam_tune(oracles=oracles, hparam_space=config_tune, hparam_default=config_default, count=args.n_runs)
    else:
        raise ValueError('Unrecognized task name.')

    end_time = time()
    hours = (end_time - start_time) / 3600.0
    print('---- The whole process takes %.2f hours ----' % (hours))


if __name__ == "__main__":
    main()
