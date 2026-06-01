#
# Adapted from VSD implementation at: https://github.com/csiro-funml/variationalsearch/blob/main/experiments/scripts/ehrlich.py
#
"""BBO using the Ehrlich function.

This experiment is based off:
https://github.com/MachineLearningLifeScience/poli-baselines/blob/main/examples/07_running_lambo2_on_ehrlich/run.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import click
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from poli.core.black_box_information import BlackBoxInformation
from poli.core.util.abstract_observer import AbstractObserver
from poli.objective_repository import (
    EhrlichHoloProblemFactory,
    EhrlichProblemFactory,
)
from vsd.augmentation import TransitionAugmenter
from vsd.proposals import DTransformerProposal, MultiCategoricalProposal
from vsd.thresholds import BudgetAnnealedThreshold


from genbo.losses import loss_registry
from genbo.solvers import GenerativeSolver
from genbo.utility import utilities_registry


SEQLEN_SETTING = {
    15: dict(motif_length=4, n_motifs=2, quantization=4),
    32: dict(motif_length=4, n_motifs=2, quantization=4),
    64: dict(motif_length=4, n_motifs=8, quantization=4),
    256: dict(motif_length=10, n_motifs=16, quantization=10),
}
DNETWORKS = {15: 32, 32: 64, 64: 128, 256: 256}
NHEADS = {15: 1, 32: 2, 64: 4, 256: 6}


# Code from https://github.com/MachineLearningLifeScience/poli-baselines/blob/
#   main/examples/07_running_lambo2_on_ehrlich/simple_observer.py
class SimpleObserver(AbstractObserver):
    def __init__(self) -> None:
        self.x_s = []
        self.y_s = []
        super().__init__()

    def initialize_observer(
        self,
        problem_setup_info: BlackBoxInformation,
        caller_info: object,
        seed: int,
    ) -> object: ...

    def observe(self, x: np.ndarray, y: np.ndarray, context=None) -> None:
        self.x_s.append(x)
        self.y_s.append(y)


def plot_best_y(obs: SimpleObserver, ax: plt.Axes, start_from: int = 0):
    best_y = np.maximum.accumulate(np.vstack(obs.y_s).flatten())
    ax.plot(best_y.flatten()[start_from:])
    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel("Best value found")


def plot_regret(ystar: float, obs: SimpleObserver, ax: plt.Axes, start_from: int = 0):
    diff = ystar - np.vstack(obs.y_s)
    regret = np.minimum.accumulate(diff.flatten())
    ax.plot(regret[start_from:])
    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel("Simple regret")


def plot_all_y(obs: SimpleObserver, ax: plt.Axes, start_from: int = 0):
    ax.plot(np.vstack(obs.y_s).flatten()[start_from:], ".")
    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel("Values found")


@click.command()
@click.option(
    "--proposal",
    type=click.Choice(
        [
            "tfm",
            "mf",
        ]
    ),
    default="mf",
    help="proposal to use.",
)
@click.option(
    "--utility",
    type=click.Choice(utilities_registry.keys()),
    default="EI",
    help="utility function to use.",
)
@click.option(
    "--sequence-length",
    type=click.Choice([str(k) for k in SEQLEN_SETTING.keys()]),
    default="32",
    help="sequence length for the Ehrlich function.",
)
@click.option("--max-iter", type=int, default=5, help="Maximum iterations to run.")
@click.option(
    "--bsize",
    type=int,
    default=128,
    help="Batch size for black box evaluation.",
)
@click.option(
    "--logdir",
    type=click.Path(file_okay=False),
    default="ehrlich",
    help="log and results directory.",
)
@click.option("--device", type=str, default="cpu", help="device to use for solver.")
@click.option("--seed", type=int, default=42, help="random seed.")
@click.option(
    "--poli",
    is_flag=True,
    help="use the poli-inbuilt ehrlich function implementation.",
)
@click.option("--use-prior", is_flag=True, help="use prior to weigh utilities.")
@click.option(
    "--use-logits",
    is_flag=True,
    help="use proposal log probabilities for importance weights.",
)
@click.option("--reg-factor", type=float, default=0.0, help="regularization factor.")
@click.option("--use-exp", is_flag=True, help="Use exponential L2 regularization")
@click.option("--weight-decay", type=float, default=0.0, help="weight decay factor.")
@click.option(
    "--warm-start", is_flag=True, help="warm start the solver from a previous solution."
)
@click.option(
    "--loss",
    type=click.Choice(loss_registry.keys()),
    default="ForwardKL",
    help="loss function to use for optimization.",
)
@click.option(
    "--lr", type=float, default=1e-2, help="learning rate for proposal training"
)
@click.option(
    "--dropout",
    type=float,
    default=0.0,
    help="Dropout probability (for transformers only). Default: 0",
)
@click.option(
    "--suffix", type=str, default="", help="additional suffix to the output file names"
)
def ehrlich(
    proposal,
    utility,
    sequence_length,
    max_iter,
    bsize,
    logdir,
    device,
    seed,
    poli,
    use_prior,
    use_logits,
    reg_factor,
    use_exp,
    weight_decay,
    warm_start,
    loss,
    lr,
    dropout,
    suffix,
):
    args_dict = locals()
    # Setup logging
    basename = f"genbo-{proposal}-{utility}-{loss}"
    if reg_factor > 0:
        basename += f"-reg{reg_factor:0.4f}".replace(".", "p")
    if use_prior:
        basename += "-prior"
    else:
        basename += "-noprior"
    if use_exp:
        basename += "-exp"
    if suffix:
        basename += f"-{suffix}"
    basename += f"_{seed}"

    logdir = Path(logdir) / sequence_length
    logdir.mkdir(exist_ok=True, parents=True)
    logfile = logdir / Path(basename + ".log")
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(logfile, mode="w"),
            logging.StreamHandler(),
        ],
        format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    )
    log = logging.getLogger(name=__name__)

    with open(logdir / f"{basename}.json", "w") as f:
        json.dump(args_dict, f, indent=4)

    settings_text = ""
    for k, val in args_dict.items():
        settings_text += f"- {k}: {val}\n"
    log.info("Running GenBO with settings:\n%s", settings_text)
    log.info("Logging to: %s", logfile)

    # Fix seeds
    torch.manual_seed(seed)

    # Setup problem
    sequence_length = int(sequence_length)
    log.info(
        f"Creating Ehrlich function: length {sequence_length} with"
        f" {SEQLEN_SETTING[sequence_length]}"
    )
    if poli:
        log.info("Using built-in poli Ehrlich implementation.")
        black_box, ystar, x0 = _erhlich_poli(sequence_length, seed, bsize)
    else:
        log.info("Using Holo bench Ehrlich implementation.")
        black_box, ystar, x0 = _erhlich_holo(sequence_length, seed, bsize)
    y0 = black_box(x0)

    # Set up a history logger
    observer = SimpleObserver()
    black_box.set_observer(observer)
    observer.x_s.append(x0)
    observer.y_s.append(y0)

    alpha_len = len(black_box.alphabet)
    if proposal == "tfm":
        vdistribution_class = DTransformerProposal
        vdistribution_kwargs = dict(
            d_features=sequence_length,
            k_categories=alpha_len,
            nhead=NHEADS[sequence_length],
            num_layers=2,
            dim_feedforward=DNETWORKS[sequence_length],
            clip_gradients=1.0,
            dropout=dropout,
        )
        vdist_options = dict(
            stop_options=dict(maxiter=20000, n_window=1200),
            scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
            scheduler_options=dict(T_0=200, T_mult=2),
        )
    elif proposal == "mf":
        vdistribution_class = MultiCategoricalProposal
        vdistribution_kwargs = dict(
            d_features=sequence_length,
            k_categories=alpha_len,
        )
        vdist_options = dict(
            stop_options=dict(maxiter=20000, n_window=600),
            scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
            scheduler_options=dict(T_0=100, T_mult=2),
        )
    else:
        log.error("Unknown proposal type: %s", proposal)
        sys.exit(1)

    vdist_options["optimizer_options"] = dict(lr=lr, weight_decay=weight_decay)
    vdist_options["verbose"] = True

    augmenter = TransitionAugmenter(max_mutations=5)
    prior_options = dict(
        augmenter=augmenter,
        augmentation_p=0.2,
        stop_options=dict(maxiter=20000, n_window=1000),
    )
    utility_fn = utilities_registry[utility](
        torch.from_numpy(y0).to(torch.get_default_dtype()), percentile=0.8
    )
    threshold = BudgetAnnealedThreshold(p0=0.5, pT=0.99, T=max_iter)
    loss_class = loss_registry[loss]
    loss_kwargs = dict(reg_factor=reg_factor, exp_reg=use_exp)

    # Create GenBO solver
    optim = GenerativeSolver(
        black_box=black_box,
        x0=x0,
        y0=y0,
        utility_fn=utility_fn,
        vdistribution_factory=vdistribution_class,
        vdistribution_kwargs=vdistribution_kwargs,
        loss_factory=loss_class,
        loss_kwargs=loss_kwargs,
        threshold=threshold,
        bsize=bsize,
        prior_options=prior_options,
        vdist_options=vdist_options,
        use_logits=use_logits,
        use_prior=use_prior,
        device=device,
        warm_start=warm_start,
    )
    solver_kwargs = dict(seed=seed)

    def post_step_cb(s: GenerativeSolver):
        np.savetxt(logdir / f"{basename}-losses.csv", s.last_losses.cpu().numpy())
        plt.plot(s.last_losses.cpu().numpy(), label=f"t = {s.iteration+1}")
        plt.ylabel("Loss")
        plt.xlabel("Step")
        plt.yscale("symlog")
        plt.legend(ncols=2)
        plt.savefig(logdir / f"{basename}-losses.png")

    # Solve
    log.info("Running solver...")
    start_t = time.time()
    optim.solve(max_iter=max_iter, post_step_callbacks=[post_step_cb], **solver_kwargs)
    end_t = time.time()
    log.info("Solver done")
    log.info("Elapsed time: %d s", end_t - start_t)

    fig, (ax1, ax2) = plt.subplots(1, 2, dpi=150, figsize=(12, 5))
    plot_regret(ystar, observer, ax1)
    plot_all_y(observer, ax2)
    for ax in (ax1, ax2):
        ax.axvline(len(x0), color="red", label="training cuttoff")
        ax.legend()
    fig.tight_layout()
    figurefile = logdir / f"{basename}.png"
    plt.savefig(figurefile)
    plt.close()

    # Save results
    try:
        resultsfile = logdir / f"{basename}.npz"
        np.savez(resultsfile, x=np.vstack(observer.x_s), y=np.vstack(observer.y_s))
    except Exception as e:
        log.error(f"Issue saving results: {e}")

    black_box.terminate()


def _erhlich_poli(sequence_length, seed, bsize):
    problem = EhrlichProblemFactory().create(
        sequence_length=sequence_length,
        return_value_on_unfeasible=-1,
        seed=seed,
        **SEQLEN_SETTING[sequence_length],
    )
    black_box = problem.black_box
    rs = np.random.RandomState(seed=seed)
    x0 = np.array(
        [list(black_box._sample_random_sequence(random_state=rs)) for _ in range(bsize)]
    )

    # Optimum is 1 for Ehrlich functions -- but just in case
    xstar = black_box.construct_optimal_solution()
    ystar = black_box(xstar)

    return black_box, ystar, x0


def _erhlich_holo(sequence_length, seed, bsize):
    problem = EhrlichHoloProblemFactory().create(
        sequence_length=sequence_length,
        return_value_on_unfeasible=-1,
        seed=seed,
        **SEQLEN_SETTING[sequence_length],
    )
    black_box = problem.black_box
    x0 = black_box.initial_solution(n_samples=bsize)
    x0 = np.array([list(x) for x in x0])

    # Optimum is 1 for Ehrlich functions -- but just in case
    xstar = black_box.optimal_solution()
    ystar = black_box(xstar)

    return black_box, ystar, x0


if __name__ == "__main__":
    ehrlich()
