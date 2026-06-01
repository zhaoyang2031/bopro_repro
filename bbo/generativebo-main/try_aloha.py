#
# Adapted from VSD implementation at: https://github.com/csiro-funml/variationalsearch/blob/main/experiments/scripts/ehrlich.py
#
"""BBO using the Ehrlich function.

This experiment is based off:
https://github.com/MachineLearningLifeScience/poli-baselines/blob/main/examples/07_running_lambo2_on_ehrlich/run.py
"""

import json
import logging
from pathlib import Path

import click
import matplotlib as mpl
import warnings

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from poli.core.black_box_information import BlackBoxInformation
from poli.core.util.abstract_observer import AbstractObserver
from poli.objective_repository import AlohaProblemFactory
from poli_baselines.solvers.simple.random_mutation import RandomMutation

try:
    from vsd.proposals import MultiCategoricalProposal, SequenceUninformativePrior
    from vsd.thresholds import BudgetAnnealedThreshold
    from vsd.surrogates import NNClassProbability
    from vsd.solvers import VSDSolver, CbASSolver
except ImportError:
    warnings.warn("Could not import VSD modules.")

from genbo.losses import loss_registry
from genbo.utility import utilities_registry

try:
    from genbo.solvers import GenerativeSolver
except (ImportError, NameError):
    warnings.warn("Could not import GenBO solver.")

NUM_MUTATIONS = 3


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
    "--solver",
    type=click.Choice(["genbo", "randmut", "vsd", "cbas"]),
    default="genbo",
    help="Solver method.",
)
@click.option(
    "--utility",
    type=click.Choice(utilities_registry.keys()),
    default="EI",
    help="utility function to use.",
)
@click.option("--max-iter", type=int, default=10, help="Maximum iterations to run.")
@click.option(
    "--bsize",
    type=int,
    default=8,
    help="Batch size for black box evaluation.",
)
@click.option(
    "--tsize",
    type=int,
    default=64,
    help="Initial training dataset.",
)
@click.option(
    "--logdir",
    type=click.Path(file_okay=False),
    default="aloha",
    help="log and results directory.",
)
@click.option("--device", type=str, default="cpu", help="device to use for solver.")
@click.option("--seed", type=int, default=42, help="random seed.")
@click.option("--use-prior", is_flag=True, help="use prior to weigh utilities.")
@click.option(
    "--use-logits",
    is_flag=True,
    help="use proposal log probabilities for importance weights.",
)
@click.option("--reg-factor", type=float, default=0.0, help="regularization factor.")
@click.option("--weight-decay", type=float, default=0.0, help="weight decay factor.")
@click.option(
    "--warm-start",
    is_flag=True,
    help="warm start the solver from a previous solution.",
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
@click.option("--percentile-min", type=float, default=0.5, help="minimum percentile.")
@click.option("--percentile-max", type=float, default=0.99, help="maximum percentile.")
@click.option(
    "--suffix",
    type=str,
    default="",
    help="additional suffix to the output file names",
)
def aloha(
    solver,
    utility,
    max_iter,
    bsize,
    tsize,
    logdir,
    device,
    seed,
    use_prior,
    use_logits,
    reg_factor,
    weight_decay,
    warm_start,
    loss,
    lr,
    percentile_min,
    percentile_max,
    suffix,
):
    args_dict = locals()

    basename = solver
    if solver == "genbo":
        basename += f"-{utility}-{loss}"
        if reg_factor > 0:
            basename += f"-reg{reg_factor:0.4f}".replace(".", "p")
        if use_prior:
            basename += "-prior"
        else:
            basename += "-noprior"
        basename += f"-lr{lr:0.4f}".replace(".", "p")
    if solver in ("vsd", "cbas", "genbo"):
        basename += f"-pmin{percentile_min:0.2f}-pmax{percentile_max:0.2f}".replace(
            ".", "p"
        )
    if suffix:
        basename += f"-{suffix}"
    basename += f"_{seed}"

    # Setup logging
    logdir = Path(logdir)
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
    log.info("Creating ALOHA function")
    problem = AlohaProblemFactory().create(seed=seed)
    black_box = problem.black_box
    sequence_length = black_box.info.max_sequence_length
    black_box.alphabet = black_box.info.alphabet

    # Make initial low fitness training data
    x0 = []
    while len(x0) < tsize:
        x = np.random.choice(black_box.alphabet, replace=True, size=sequence_length)
        if black_box(np.array([x])) > 1:
            continue
        x0.append(x)
    x0 = np.array([list(x) for x in x0])
    y0 = black_box(x0)

    # Set up a history logger
    observer = SimpleObserver()
    black_box.set_observer(observer)
    observer.x_s.extend(x0)
    observer.y_s.extend(y0)

    alpha_len = len(black_box.alphabet)

    def post_step_cb(s): ...

    solver_kwargs = dict(seed=seed)
    if solver in ("genbo", "vsd", "cbas"):
        threshold = BudgetAnnealedThreshold(
            p0=percentile_min, pT=percentile_max, T=max_iter
        )
        prior = SequenceUninformativePrior(
            d_features=sequence_length, k_categories=alpha_len
        )
        vdistribution_class = MultiCategoricalProposal
        vdistribution_kwargs = dict(
            d_features=sequence_length,
            k_categories=alpha_len,
        )

    if solver == "genbo":
        vdist_options = dict(
            stop_options=dict(maxiter=20000, n_window=600),
            scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
            scheduler_options=dict(T_0=100, T_mult=2),
        )

        vdist_options["optimizer_options"] = dict(lr=lr, weight_decay=weight_decay)
        vdist_options["verbose"] = True

        utility_fn = utilities_registry[utility](
            torch.from_numpy(y0).to(torch.get_default_dtype()), percentile=0.8
        )
        loss_class = loss_registry[loss]
        loss_kwargs = dict(reg_factor=reg_factor)

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
            prior=prior,
            vdist_options=vdist_options,
            use_logits=use_logits,
            use_prior=use_prior,
            device=device,
            warm_start=warm_start,
        )

        def post_step_cb(s: GenerativeSolver):
            np.savetxt(logdir / f"{basename}-losses.csv", s.last_losses.cpu().numpy())
            plt.plot(s.last_losses.cpu().numpy(), label=f"t = {s.iteration+1}")
            plt.ylabel("Loss")
            plt.xlabel("Step")
            plt.yscale("symlog")
            plt.legend(ncols=2)
            plt.savefig(logdir / f"{basename}-losses.png")

    elif solver in ("vsd", "cbas"):
        vdistribution = vdistribution_class(**vdistribution_kwargs)
        cpe = NNClassProbability(
            seq_len=sequence_length,
            alpha_len=alpha_len,
            embedding_dim=16,
            dropoutp=0.1,
            hlsize=64,
        )
        vdist_options = dict(
            gradient_samples=128,
            stop_options=dict(maxiter=10000, n_window=2000),
            optimizer_options=dict(lr=1e-3),
        )
        cpe_options = dict(
            stop_options=dict(maxiter=10000, n_window=1000),
            optimizer_options=dict(lr=1e-3),
        )
        optim = (VSDSolver if solver == "vsd" else CbASSolver)(
            black_box=black_box,
            x0=x0,
            y0=y0,
            threshold=threshold,
            cpe=cpe,
            vdistribution=vdistribution,
            prior=prior,
            bsize=bsize,
            device=device,
            vdist_options=vdist_options,
            cpe_options=cpe_options,
            seed=seed,
        )

    elif solver == "randmut":
        optim = RandomMutation(
            black_box=black_box,
            x0=x0,
            y0=y0,
            n_mutations=NUM_MUTATIONS,
            batch_size=bsize,
            top_k=10 * bsize,
        )

    else:
        print(f"Unknown solver {solver}!")
        exit()

    # Save results
    def save_results(s):
        post_step_cb(s)
        try:
            resultsfile = logdir / f"{basename}.npz"
            np.savez(
                resultsfile,
                x=np.vstack(observer.x_s),
                y=np.vstack(observer.y_s),
            )
        except Exception as e:
            log.error(f"Issue saving results: {e}")

    # Solve
    log.info("Running solver...")
    optim.solve(max_iter=max_iter, post_step_callbacks=[save_results], **solver_kwargs)
    save_results(optim)
    log.info("Solver done")

    fig, (ax1, ax2) = plt.subplots(1, 2, dpi=150, figsize=(12, 5))
    plot_regret(5, observer, ax1)
    plot_all_y(observer, ax2)
    for ax in (ax1, ax2):
        ax.axvline(len(x0), color="red", label="training cuttoff")
        ax.legend()
    fig.tight_layout()
    figurefile = logdir / f"{basename}.png"
    plt.savefig(figurefile)
    plt.close()

    black_box.terminate()


if __name__ == "__main__":
    aloha()
