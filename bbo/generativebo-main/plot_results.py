#
# Adapted from VSD implementation at: https://github.com/csiro-funml/variationalsearch/blob/main/experiments/scripts/plot_ehrlich.py
#
"""Plot results of the Poli BBO experiments."""

import json
import sys
import warnings
from glob import glob
from itertools import cycle
from pathlib import Path

import click
import matplotlib as mpl
import numpy as np
from tqdm import tqdm

mpl.use("Agg")
import matplotlib.pyplot as plt

from experiments.metrics import diversity

LINECYCLE = cycle(
    [
        "-",
        "--",
        ":",
        "-.",
        (0, (3, 1, 1, 1)),
        (0, (5, 1)),
        (0, (3, 1, 1, 1, 1, 1)),
        (0, (1, 1)),
    ]
)

STR_REPS = {
    "vsd": "VSD",
    "cbas": "CbAS",
    "lambo2": "LaMBO-2",
    "ga": "GA",
    "tfm": "Tfm",
    "lstm": "LSTM",
    "mf": "MF",
    "genbo": "GenBO",
    "Balanced": "b",
    "Counter": "c",
    "Differences": "D",
    "Error": "E",
    "Forward": "f",
    "Loss": "L",
    "Preference": "P",
    "Reverse": "r",
    "Robust": "r",
    "Squared": "S",
    "randmut": "Random mutations",
    "rand": "Random mutations",
    "noprior": "np",
    "frequent-reg": "fr",
    "reg": "r",
    "00-": "-",
    "prior": "p",
    "logits": "lg",
    "pmin0p": "pc",
    "pmax0p": "",
}


@click.command()
@click.option(
    "--resultsdir",
    type=click.Path(exists=True, file_okay=False),
    default="poli",
    help="Base poli results directory.",
)
@click.option(
    "--ystar",
    type=float,
    default=1.0,
    help="Black-box function maximum.",
)
@click.option(
    "--trainsize",
    type=int,
    default=128,
    help="Training data size.",
)
@click.option(
    "--batchsize",
    type=int,
    default=128,
    help="batch size for aggregating results.",
)
@click.option("--maxiter", type=int, default=100, help="maximum number of iterations.")
@click.option("--topk", type=int, default=None, help="Top-k GenBO variants to keep")
@click.option(
    "--plot-maximum",
    is_flag=True,
    help="plot cumulative maximum, instead of simple regret",
)
@click.option(
    "--fileprefix",
    type=str,
    default="",
    help="prefix for the output file names.",
)
@click.option(
    "--filesuffix",
    type=str,
    default=None,
    help="suffix for output files. If none, use directory name.",
)
@click.option("--figwidth", type=float, default=5, help="figure width.")
@click.option("--figheight", type=float, default=4, help="figure height.")
@click.option(
    "--legend-cols", type=int, default=1, help="number of columns for legend table"
)
@click.option(
    "--legend-fontsize", type=int, default=10, help="fontsize for legend labels"
)
def plot_results(
    resultsdir,
    ystar,
    trainsize,
    batchsize,
    maxiter,
    topk,
    plot_maximum,
    fileprefix,
    filesuffix,
    figwidth,
    figheight,
    legend_cols,
    legend_fontsize,
):
    basedir = Path(resultsdir)
    if not basedir.exists():
        print(f"Error: cannot find path {basedir}")
        sys.exit(-1)

    if filesuffix is None:
        filesuffix = basedir.stem

    # Get all unique methods
    resfiles = glob((basedir / "*.npz").as_posix())
    methods = set([Path(f).stem.split("_")[0] for f in resfiles])
    method_files = {m: [f for f in resfiles if m + "_" in f] for m in methods}

    if plot_maximum:
        ystar = 0

    # Load results by method
    cumregrets = {}
    bdiversity = {}
    max_evals_found = 0
    # max_evals = maxiter * batchsize
    for m, files in tqdm(
        method_files.items(), total=len(method_files), desc="Loading results"
    ):
        if len(files) < 1:
            continue
        cumregrets[m] = []
        bdiversity[m] = []
        for f in files:
            d = np.load(f)
            regret = ystar - d["y"].flatten()
            n_evals = regret.shape[-1]
            max_evals_found = max(n_evals, max_evals_found)
            X = np.array(["".join(x) for x in d["x"]])
            n = len(X)
            cumregret = np.minimum.accumulate(regret)
            s, e = _round_inds(trainsize, batchsize, n)
            cumregrets[m].append([])
            bdiversity[m].append([])
            for i, j in zip(s, e):
                bdiversity[m][-1].append(diversity(X[i:j]))
                cumregrets[m][-1].append(cumregret[j - 1])
            cumregrets[m][-1] = np.array(cumregrets[m][-1])
            bdiversity[m][-1] = np.array(bdiversity[m][-1])

    max_iter_found = 1 + (max_evals_found - trainsize) // batchsize

    # Sort by performance
    means, dmeans = {}, {}
    stds, dstds = {}, {}
    regrets = []
    methods = []
    for m, ys in cumregrets.items():
        try:
            ys = np.vstack(
                [np.pad(y, (0, max_iter_found - len(y)), mode="edge") for y in ys]
            )[..., :max_iter_found]
            bdiversity[m] = np.vstack(
                [
                    np.pad(b, (0, max_iter_found - len(b)), mode="edge")
                    for b in bdiversity[m]
                ]
            )[..., :max_iter_found]
            means[m] = np.nanmean(ys, axis=0)
            dmeans[m] = np.nanmean(bdiversity[m], axis=0)
            stds[m] = np.nanstd(ys, axis=0)
            dstds[m] = np.nanstd(bdiversity[m], axis=0)
            methods.append(m)
            regrets.append(means[m][-1])
        except Exception:
            warnings.warn(f"Unable to process: {m}")
    methods = np.array(methods)[np.argsort(regrets)[::-1]]

    if topk is not None:
        for method in ("genbo-PI", "genbo-EI", "genbo-sEI", "vsd", "cbas"):
            methods = keep_bottom_k_matches(methods, base=method, K=topk)

    cycler = plt.cycler(
        color=plt.cm.viridis_r(np.linspace(0.05, 0.95, len(methods)))
    ) + plt.cycler(linestyle=[next(LINECYCLE) for _ in range(len(methods))])
    plt.rc("axes", prop_cycle=cycler)
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    # Plot by measure
    alpha = 0.1 if len(methods) > 10 else 0.15
    for p, f in (("Simple regret", "regret"), ("Diversity", "diversity")):
        fig, ax = plt.subplots(dpi=200, figsize=(figwidth, figheight))
        for m in methods:
            if f == "regret":
                mean, std = means[m], stds[m]
                if plot_maximum:
                    mean *= -1
                    p = "Maximum outcome"
            else:
                mean, std = dmeans[m], dstds[m]
            x = np.arange(len(mean))
            ax.fill_between(x, mean - std, mean + std, alpha=alpha)
            for old, new in STR_REPS.items():
                m = m.replace(old, new)
            ax.plot(x, mean, label=m, linewidth=2)
            ax.set_xlabel("Round", fontsize=14)
        ax.set_ylabel(p, fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(loc="best", frameon=True, framealpha=0.5, ncols=legend_cols)
        fig.tight_layout()
        fname = f
        if fileprefix != "":
            fname = f"{fileprefix}-{fname}"
        if filesuffix != "":
            fname += "-" + filesuffix
        fname += ".png"
        plt.savefig(basedir / fname)
        plt.close()
        print(f"Plot saved to: {basedir / fname}")

    suffix = f"-{filesuffix}" if filesuffix else ""
    with open(basedir / f"{fileprefix}-final{suffix}.json", "w") as f:
        json.dump(
            {m: means[m][-1].item() for m in methods}, f, indent=4, sort_keys=True
        )
    with open(basedir / f"{fileprefix}-final-std{suffix}.json", "w") as f:
        json.dump({m: stds[m][-1].item() for m in methods}, f, indent=4, sort_keys=True)
    with open(basedir / f"{fileprefix}-rank{suffix}.json", "w") as f:
        json.dump(methods.tolist(), f, indent=4, sort_keys=True)


def _round_inds(trainsize, batchsize, n):
    starts = [0] + list(range(trainsize, n, batchsize))
    ends = [trainsize] + [min(s + batchsize, n) for s in range(trainsize, n, batchsize)]
    return starts, ends


def keep_bottom_k_matches(arr: np.ndarray, base: str, K: int) -> np.ndarray:
    mask = np.char.startswith(arr, base)
    indices = np.where(mask)[0]

    # Indices to keep for matching entries: the last K matching indices
    keep_match_indices = set(indices[-K:]) if len(indices) > 0 else set()

    # Build a boolean mask to keep elements:
    # - Keep all non-matching elements
    # - Keep matching elements only if in keep_match_indices
    keep_mask = np.array(
        [(not m) or (i in keep_match_indices) for i, m in enumerate(mask)]
    )

    return arr[keep_mask]


if __name__ == "__main__":
    plot_results()
