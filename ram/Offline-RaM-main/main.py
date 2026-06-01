import argparse
import datetime
from argparse import Namespace
from copy import deepcopy

import design_bench as db
import numpy as np
import torch
import torch.nn.functional as F
from design_bench.task import Task
from huggingface_hub import hf_hub_download
from safetensors import safe_open

import wandb
from hf_wrapped_model import MLPConfig, SimpleMLP
from losses import get_loss_fn
from metrics import cal_overlap_auc, spearman_corr
from search import adam_search, grad_search
from utils import (
    build_data_loader,
    load_default_config,
    load_elite_data,
    record_from_dict,
    set_seed,
    task_kwargs,
)

_TKWARGS = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.float32,
}


def run(args: Namespace):
    task_entry = f"MLP-{args.loss}"
    if args.expt_name:
        task_entry += f"-{args.expt_name}"

    set_seed(args.seed)

    task: Task = db.make(args.task, **task_kwargs(args.task))

    x = task.x.copy()
    y = task.y.copy()

    if args.eval_elites:
        x_elites, y_elites = load_elite_data(args.task, task)

    if args.normalize_ys:
        y = task.normalize_y(y)
        y_elites = task.normalize_y(y_elites)
    if task.is_discrete:
        x = task.to_logits(x)
        x_elites = task.to_logits(x_elites)
    if args.normalize_xs and (not task.is_discrete or args.normalize_logits):
        x = task.normalize_x(x)
        x_elites = task.normalize_x(x_elites)

    if args.use_wandb:
        run_name = f"{task_entry}-seed{args.seed}-{args.task}"
        ts = datetime.datetime.utcnow() + datetime.timedelta(hours=+8)
        ts_name = f"-ts-{ts.year}-{ts.month}-{ts.day}_{ts.hour}-{ts.minute}-{ts.second}"

        # wandb.login(key=args.wandb_api)

        wandb.init(
            project="Offline-Relation",
            name=run_name + ts_name,
            config=args.__dict__,
            group=f"{task_entry}",
            job_type=args.run_type,
            mode="online",
        )

    _shape0 = x.shape[1:]
    x = x.reshape(x.shape[0], -1)
    x_elites = x_elites.reshape(x_elites.shape[0], -1)

    x_elites = torch.from_numpy(x_elites).to(**_TKWARGS)
    y_elites = torch.from_numpy(y_elites).to(**_TKWARGS)

    iid_loader, _ = build_data_loader(
        x=x, y=y, batch_size=args.batch_size * 2, require_valid=False, drop_last=False
    )

    forward_model = SimpleMLP(MLPConfig(input_dim=x.shape[1]))

    weights_path = hf_hub_download(
        repo_id="trxcc2002/Offline-RaM-ListNet",
        filename=f"Offline-RaM-ListNet-{args.task}-seed{args.seed}/model.safetensors",
    )
    state_dict = {}
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
    forward_model.load_state_dict(state_dict)
    forward_model.to(**_TKWARGS)

    loss_fn = get_loss_fn(args.loss, **args.loss_config)

    y_pred = forward_model(x_elites)
    prediction = loss_fn.score(y_pred)
    elite_mse = F.mse_loss(
        input=prediction.squeeze(), target=y_elites.squeeze(), reduction="mean"
    ).item()
    elite_rank_corr = spearman_corr(prediction, y_elites).item()
    elite_auc_pr = cal_overlap_auc(prediction, y_elites)

    if args.loss != "mse":
        pred_all = []
        for x_batch, _ in iid_loader:
            x_batch = x_batch.to(**_TKWARGS)
            y_pred = forward_model(x_batch)
            pred_all.append(y_pred)
        pred_all = torch.cat(pred_all, dim=0)
        pred_mean = pred_all.mean().item()
        pred_std = pred_all.std().item()
    else:
        pred_mean = 0.0
        pred_std = 1.0

    x_init = torch.Tensor(x[np.argsort(y.squeeze())[-args.num_solutions :]]).to(
        **_TKWARGS
    )

    x_res = deepcopy(x_init)

    if args.x_opt_method.lower() == "adam":
        x_res = adam_search(
            x_init=x_init,
            forward_model=forward_model,
            score_fn=lambda x: (loss_fn.score(x) - pred_mean) / pred_std,
            x_opt_lr=(
                args.x_opt_lr["discrete"]
                if task.is_discrete
                else args.x_opt_lr["continuous"]
            ),
            x_opt_step=(
                args.x_opt_step["discrete"]
                if task.is_discrete
                else args.x_opt_step["continuous"]
            ),
        )
    elif args.x_opt_method.lower() == "grad":
        x_res = grad_search(
            x_init=x_init,
            forward_model=forward_model,
            score_fn=lambda x: (loss_fn.score(x) - pred_mean) / pred_std,
            x_opt_lr=(
                args.x_opt_lr["discrete"]
                if task.is_discrete
                else args.x_opt_lr["continuous"]
            ),
            x_opt_step=(
                args.x_opt_step["discrete"]
                if task.is_discrete
                else args.x_opt_step["continuous"]
            ),
        )
    else:
        raise NotImplementedError("unknown search method")

    x_res = x_res.reshape((x_res.shape[0],) + tuple(_shape0)).detach().cpu().numpy()

    if args.normalize_xs:
        x_res = task.denormalize_x(x_res)
    if task.is_discrete:
        x_res = task.to_integers(x_res)

    score = task.predict(x_res)
    score_100th = np.max(score)
    score_50th = np.median(score)
    score_25th = np.percentile(score, 25)
    score_75th = np.percentile(score, 75)

    dic2y = np.load("dic2y.npy", allow_pickle=True).item()
    y_min, y_max = dic2y[args.task]

    nmr_score_100th = (score_100th - y_min) / (y_max - y_min)
    nmr_score_75th = (score_75th - y_min) / (y_max - y_min)
    nmr_score_50th = (score_50th - y_min) / (y_max - y_min)
    nmr_score_25th = (score_25th - y_min) / (y_max - y_min)

    print(f"Score-100th: {nmr_score_100th}")
    print(f"Score-75th: {nmr_score_75th}")
    print(f"Score-50th: {nmr_score_50th}")
    print(f"Score-25th: {nmr_score_25th}")

    results_dict = {
        "Normalized-Score-100th": nmr_score_100th,
        "Normalized-Score-75th": nmr_score_75th,
        "Normalized-Score-50th": nmr_score_50th,
        "Normalized-Score-25th": nmr_score_25th,
        "Score-100th": score_100th,
        "Score-75th": score_75th,
        "Score-50th": score_50th,
        "Score-25th": score_25th,
        # "IID-MSE": iid_mse,
        # "IID-Rank-Correlation": iid_rank_corr,
        # "IID-AUC-PR": iid_auc_pr,
        "Elite-MSE": elite_mse,
        "Elite-Rank-Correlation": elite_rank_corr,
        "Elite-AUC-PR": elite_auc_pr,
    }
    record_from_dict(
        metric_dict=results_dict, task=args.task, model=f"{task_entry}", seed=args.seed
    )

    if args.use_wandb:
        wandb.log(
            {
                "Normalized-Score/100th": nmr_score_100th,
                "Normalized-Score/75th": nmr_score_75th,
                "Normalized-Score/50th": nmr_score_50th,
                "Normalized-Score/25th": nmr_score_25th,
                "Score/100th": score_100th,
                "Score/75th": score_75th,
                "Score/50th": score_50th,
                "Score/25th": score_25th,
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Superconductor-RandomForest-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--loss",
        type=str,
        default="listnet",
        choices=[
            "sigmoid_ce",
            "bce",
            "mse",
            "ranknet",
            "lambdarank",
            "rankcosine",
            "softmax",
            "listnet",
            "listmle",
            "approxndcg",
        ],
    )
    parser.add_argument("--list-length", type=int, default=1000)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--retrain-model", action="store_true", default=False)
    parser.add_argument("--expt-name", type=str, default="")
    args = parser.parse_args()

    default_config = load_default_config(args.loss)
    default_config.update(args.__dict__)
    args.__dict__ = default_config

    run(args)
