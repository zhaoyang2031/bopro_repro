"""Solver interfaces for VSD and other methods for poli compatibility

See https://machinelearninglifescience.github.io/poli-docs/contributing/a_new_solver.html
"""

import time
import importlib.util
import logging
import math
import typing as T
from abc import ABC

import numpy as np
import torch
from botorch.optim.stopping import ExpMAStoppingCriterion
from numpy import ndarray
from poli.core.abstract_black_box import AbstractBlackBox
from poli_baselines.core.step_by_step_solver import StepByStepSolver
from torch import Tensor
from torch.optim import Optimizer
from tqdm import tqdm


# Optionally import VSD and VSD functionality for experiments
if importlib.util.find_spec("vsd"):
    from vsd.proposals import (
        SearchDistribution,
        SequenceSearchDistribution,
        clip_gradients,
        fit_ml,
    )
    from vsd.thresholds import Threshold
    from vsd.utils import batch_indices
    from vsd.acquisition import (
        AcquisitionFunction,
        MarginalAcquisition,
        LogPIClassiferAcquisition,
        CbASAcquisition,
        VariationalSearchAcquisition,
    )
    from vsd.surrogates import ClassProbabilityModel, fit_cpe
    from vsd.generation import generate_candidates_reinforce, _adapt_acquisition

from genbo.proposals import MaskedSearchDistribution
from genbo.utility import Utility, ThresholdedUtility
from genbo.losses import Loss

LOG = logging.getLogger(name=__name__)


class GenerativeSolver(ABC, StepByStepSolver):
    """
    Generative solver interface.

    Adapted from: https://github.com/csiro-funml/variationalsearch/blob/main/vsd/solvers.py
    """

    name = "GenBO"

    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: ndarray,
        y0: T.Optional[ndarray],
        utility_fn: Utility,
        vdistribution_factory: T.Callable[[T.Any], SequenceSearchDistribution],
        loss_factory: T.Callable[[T.Any], Loss],
        threshold: T.Optional[Threshold] = None,
        vdistribution_kwargs: T.Optional[T.Dict[str, T.Any]] = None,
        loss_kwargs: T.Optional[T.Dict[str, T.Any]] = None,
        warm_start: bool = False,
        use_logits: bool = True,
        use_prior: bool = True,
        prior: T.Optional[SequenceSearchDistribution] = None,
        bsize: int = 128,
        device: str | torch.device = "cpu",
        prior_options: T.Optional[None] = None,
        vdist_options: T.Optional[None] = None,
        prior_validation_prop: float = 0,
        seed: T.Optional[int] = None,
    ):
        super().__init__(black_box, x0, y0 if y0 is not None else black_box(x0))
        self.utility_fn = utility_fn
        self.bsize = bsize
        self.prior = prior
        self.device = device
        self.prior_validation_prop = prior_validation_prop
        self.seed = seed
        self.logits = torch.zeros(y0.shape).squeeze(-1)
        self.use_logits = use_logits
        self.use_prior = use_prior
        self.threshold = threshold
        self.proposal_factory = vdistribution_factory
        if vdistribution_kwargs is None:
            vdistribution_kwargs = dict()
        self.proposal_kwargs = vdistribution_kwargs
        self.loss_factory = loss_factory
        if loss_kwargs is None:
            loss_kwargs = dict()
        self.loss_kwargs = loss_kwargs
        self.warm_start = warm_start

        # Prior -- don't over fit, need some probability mass everywhere
        self.prior_options = setdefaults(
            prior_options,
            dict(
                optimizer_options=dict(lr=1e-3, weight_decay=1e-4),
                stop_options=dict(maxiter=20000, n_window=200),
                batch_size=64,
            ),
        )
        # Posterior -- regularized, take as many iterations as needed
        self.vdist_options = setdefaults(
            vdist_options,
            dict(
                optimizer_options=dict(lr=1e-3),
                stop_options=dict(maxiter=20000, n_window=3000),
            ),
        )

        # Tokenize/de-tokenize
        self._s_to_i = {s: i for i, s in enumerate(black_box.info.alphabet)}
        self._i_to_s = {i: s for i, s in enumerate(black_box.info.alphabet)}

    def next_candidate(self) -> ndarray:
        # Tokenize and convert to tensors
        x = seq2int(np.concatenate(self.history["x"], axis=0), self._s_to_i)
        y = (
            torch.as_tensor(np.concatenate(self.history["y"], axis=0))
            .squeeze(-1)
            .to(torch.get_default_dtype())
        )
        maxy = max(y)

        # Updates
        if self.threshold is not None:
            if isinstance(self.utility_fn, ThresholdedUtility):
                self.utility_fn.threshold = self.threshold(y.cpu())
        else:
            self.utility_fn.update(y)
        u = self.utility_fn(y)

        if isinstance(self.utility_fn, ThresholdedUtility):
            thresh = self.utility_fn.threshold
            thresh_info = f"threshold = {thresh:.3f}, "
        else:
            thresh_info = ""
        LOG.info(f"Round {self.iteration}, " + thresh_info + f"max y = {maxy:.3f}.")
        if self.prior is None and self.use_prior:
            LOG.info("Fitting prior and initial variational distribution ...")
            self._fit_prior(x)

        # Build proposal
        if self.warm_start:
            if self.iteration > 0:
                proposal = self.proposal
            else:
                proposal = self.proposal_factory(**self.proposal_kwargs)
                if self.prior is not None:
                    proposal.load_state_dict(self.prior.state_dict())
        else:
            proposal = self.proposal_factory(**self.proposal_kwargs)

        # Build loss function
        loss_fn = self.loss_factory(
            model=proposal.to(self.device),
            prior=self.prior if self.use_prior else None,
            **self.loss_kwargs,
        )
        n_obs = x.shape[-2]
        loss_fn.reg_factor *= math.log(n_obs) ** 2 / n_obs

        if self.use_logits:
            logits = self.logits
        else:
            logits = None

        # Update best sequences to mutate
        if isinstance(proposal, MaskedSearchDistribution):
            LOG.info("Update seed sequences in proposal ...")
            xb = x[(u.max() - u) < u.std() / 10]
            proposal.set_seeds(xb)
            if isinstance(self.prior, MaskedSearchDistribution):
                LOG.info("Update seed sequences in prior ...")
                self.prior.set_seeds(xb)

        LOG.info(f"Fitting {self.name} ...")
        losses = fit_proposal(
            proposal,
            x,
            u,
            loss_fn,
            logits=logits,
            device=self.device,
            **self.vdist_options,
        )
        self.last_losses = losses
        LOG.info(f"Final {self.name} loss = {losses[-1]:.3g}")

        # Generate candidates
        LOG.info(f"Generating {self.bsize} candidates ...")
        self.proposal = proposal
        with torch.no_grad():
            xcand = self.proposal.sample(torch.Size([self.bsize]))
            if self.use_logits:
                new_logits = self.proposal.log_prob(xcand).squeeze(-1)
                self.logits = torch.cat([self.logits, new_logits.to(self.logits)])
        LOG.info("Generation done.")

        LOG.info("Generation done.")
        seqs = int2seq(xcand.cpu(), self._i_to_s)
        return seqs

    def _fit_prior(self, x: Tensor):
        # Make a validation set for early stopping
        x_val = None
        callback = _callback
        if self.prior_validation_prop > 0:
            nval = max(1, round(len(x) * self.prior_validation_prop))
            indices = torch.randperm(len(x))
            x_val = x[indices[:nval]]
            x = x[indices[nval:]]
            callback = _val_callback

        prior = self.proposal_factory(**self.proposal_kwargs)

        fit_ml(
            prior,
            x,
            X_val=x_val,
            callback=callback,
            device=self.device,
            seed=self.seed,
            **self.prior_options,
        )
        prior.eval()
        prior.requires_grad_(False)

        # Make sure we are not learning the prior from now
        # prior = deepcopy(self.vdistribution)
        # for p in prior.parameters():
        #     p.requires_grad = False

        self.prior = prior


class _CPEGenerativeSolver(ABC, StepByStepSolver):
    """
    Fits a CPE guided guided generative solver, e.g. VSD, CbAS etc.

    optim: callable = None
    vacquisition: VariationalSearchAcquisition = None

        This class implements a step-by-step solver that:

    1. Fits a class probability estimator (CPE) to distinguish "positive"
        examples (above threshold).
    2. Fits a variational distribution to approximate the posterior over good
        solutions.
    3. Optionally fits a prior distribution to avoid overfitting.
    4. Optimizes an acquisition function (LogPIClassifierAcquisition) to propose
        new candidates.

    Parameters
    ----------
    black_box : AbstractBlackBox
        Black-box function interface.
    x0 : ndarray
        Initial input samples.
    y0 : ndarray or None
        Initial observed outputs.
    threshold : Threshold
        Trheshold labeller function for classifying positive samples.
    cpe : ClassProbabilityModel
        Class probability estimator for positive/negative classification.
    vdistribution : SequenceSearchDistribution or AutoRegressiveSearchDistribution
        Variational search distribution to be optimized.
    prior : SequenceSearchDistribution or AutoRegressiveSearchDistribution
        Prior search distribution
    bsize : int, optional
        Batch size for candidate generation. Default is 128.
    device : str or torch.device, optional
        Computation device. Default is "cpu".
    cpe_options : dict or None, optional
        Options for CPE fitting.
    vdist_options : dict or None, optional
        Options for variational distribution optimization.
    seed : int or None, optional
        Random seed.
    acq_fn_kwargs : dict or None, optional
        Additional keyword arguments for acquisition function.

    """

    name: str
    optim: T.Callable
    vacquisition: type[MarginalAcquisition]

    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: ndarray,
        y0: T.Optional[ndarray],
        cpe: ClassProbabilityModel,
        vdistribution: SequenceSearchDistribution,
        prior: SequenceSearchDistribution,
        threshold: T.Optional[Threshold] = None,
        bsize: int = 128,
        device: str | torch.device = "cpu",
        cpe_options: T.Optional[T.Dict[str, T.Any]] = None,
        vdist_options: T.Optional[T.Dict[str, T.Any]] = None,
        seed: T.Optional[int] = None,
        acq_fn_kwargs: T.Optional[dict] = None,
    ):
        super().__init__(black_box, x0, y0 if y0 is not None else black_box(x0))
        self.labeller = threshold
        self.cpe = cpe.to(device)
        self.vdistribution = vdistribution.to(device)
        self.prior = prior.to(device) if prior is not None else prior
        self.bsize = bsize
        self.device = device
        self.seed = seed
        self.acq_fn_kwargs = {} if acq_fn_kwargs is None else acq_fn_kwargs

        self.fit = False  # 0th round fitting flag
        self.acq = LogPIClassiferAcquisition(model=self.cpe)

        self.cpe_options = setdefaults(
            cpe_options,
            dict(
                optimizer_options=dict(lr=1e-3, weight_decay=1e-6),
                stop_options=dict(maxiter=20000, n_window=1000),
                batch_size=32,
            ),
        )
        self.vdist_options = setdefaults(
            vdist_options,
            dict(
                optimizer_options=dict(lr=1e-4),
                gradient_samples=256,
            ),
        )

        # Tokenize/de-tokenize
        self._s_to_i = {s: i for i, s in enumerate(black_box.info.alphabet)}
        self._i_to_s = {i: s for i, s in enumerate(black_box.info.alphabet)}

    def next_candidate(self) -> ndarray:
        # Tokenize and convert to tensors
        x = seq2int(self.history["x"], self._s_to_i).squeeze()
        y = torch.tensor(np.concatenate(self.history["y"], axis=0)).squeeze()

        x = x.to(self.device)
        y = y.float().to(self.device)

        # Updates
        thresh = self.labeller(y.cpu())
        z = self.labeller.labels(y.cpu())
        LOG.info(
            f"Round {self.iteration}, threshold = {thresh:.3f}, "
            f"# pos = {sum(z)}, max y = {max(y):.3f}."
        )

        LOG.info("Fitting CPE ...")
        fit_cpe(
            self.cpe,
            X=x,
            y=y,
            best_f=self.labeller,
            device=self.device,
            callback=_callback,
            seed=self.seed,
            **self.cpe_options,
        )

        # Update best sequences to mutate
        if isinstance(self.vdistribution, MaskedSearchDistribution):
            LOG.info("Update seed sequences in proposal ...")
            xb = x[z == 1]
            self.vdistribution.set_seeds(xb)
            if isinstance(self.prior, MaskedSearchDistribution):
                LOG.info("Update seed sequences in prior ...")
                self.prior.set_seeds(xb)

        LOG.info(f"Optimizing {self.name} acquisition function ...")
        vacq = self.vacquisition(self.acq, self.prior, **self.acq_fn_kwargs)
        vacq = vacq.to(self.device)
        xcand, _ = type(self).optim(
            acquisition_function=vacq,
            proposal_distribution=self.vdistribution,
            candidate_samples=self.bsize,
            callback=_grad_callback,
            **self.vdist_options,
        )

        assert len(xcand) == self.bsize, "Wrong candidate size"
        return int2seq(xcand, self._i_to_s)


def fit_proposal(
    proposal: SearchDistribution,
    X: Tensor,
    utilities: Tensor,
    loss_fn: Loss,
    logits: T.Optional[Tensor] = None,
    batch_size: int = 512,
    optimizer: Optimizer = torch.optim.AdamW,
    optimizer_options: T.Optional[T.Dict[str, T.Any]] = None,
    scheduler: T.Optional[type[torch.optim.lr_scheduler.LRScheduler]] = None,
    scheduler_options: T.Optional[T.Dict[str, T.Any]] = None,
    stop_options: T.Optional[T.Dict[str, T.Any]] = None,
    device: str = "cpu",
    seed: T.Optional[int] = None,
    verbose: bool = False,
    loss_eps: float = 1e-15,
):
    """Fit a proposal distribution using ML."""

    proposal.to(device)
    optimizer_options = {} if optimizer_options is None else optimizer_options
    stop_options = {} if stop_options is None else stop_options
    scheduler_options = {} if scheduler_options is None else scheduler_options

    clip_gradients(proposal)
    optim = optimizer(proposal.parameters(), **optimizer_options)
    if scheduler is not None:
        sched = scheduler(optim, **scheduler_options)
    stopping_criterion = ExpMAStoppingCriterion(**stop_options)  # type: ignore

    losses = []

    proposal.train()

    # TEST
    def compute_ntk(points: torch.Tensor) -> torch.Tensor:
        grads = []
        for x in tqdm(points, total=len(points), desc="NTK"):
            optim.zero_grad()
            lp = proposal.log_prob(x.to(device).unsqueeze(0))
            lp.backward()
            grad = torch.cat([p.grad.ravel() for p in proposal.parameters()])
            grads.append(grad)
        grads = torch.stack(grads)
        ntk = grads @ grads.t()
        return ntk

    if verbose:
        iterator = tqdm(
            batch_indices(len(X), batch_size, seed),
            desc="Training proposal",
            total=stop_options["maxiter"],
        )
    else:
        iterator = batch_indices(len(X), batch_size, seed)
    for bi in iterator:
        Xb = X[bi]
        Ub = utilities[bi].to(device)
        Xb = Xb.to(device)

        if logits is not None:
            Lb = logits[bi].to(device)
            Lb = Lb + torch.logsumexp(-Lb, -1) - math.log(Lb.shape[-1])
        else:
            Lb = None

        # Log likelihood
        log_probs = proposal.log_prob(Xb)
        loss = loss_fn(log_probs, Xb, Ub, Lb)

        optim.zero_grad()
        loss.backward()
        optim.step()
        if scheduler is not None:
            sched.step()

        sloss = loss.detach()

        if verbose:
            iterator.set_postfix(loss=sloss.cpu().item())
            losses.append(sloss)

        if stopping_criterion(fvals=sloss + loss_eps):
            break

    proposal.eval()
    losses = torch.stack(losses)
    return losses


def generate_candidates_eda(
    acquisition_function: AcquisitionFunction,
    proposal_distribution: SearchDistribution,
    optimizer: type[Optimizer] = torch.optim.Adam,
    optimizer_options: T.Optional[T.Dict[str, T.Union[float, str]]] = None,
    stop_options: T.Optional[T.Dict[str, T.Union[float, str]]] = None,
    callback: T.Optional[
        T.Callable[[int, Tensor, T.Tuple[Tensor, ...]], T.NoReturn]
    ] = None,
    timeout_sec: T.Optional[float] = None,
    gradient_samples: T.Optional[int] = None,
    candidate_samples: T.Optional[int] = None,
) -> T.Tuple[Tensor, Tensor]:
    r"""Generate a set of candidates using Estimation of Distribution (EDA).

    Args:
        acquisition_function: Acquisition/black box function to be used.
        proposal_distribution: a SearchDistribution to optimise using
            REINFORCE, this will be used to generate candidates from.
        optimizer (Optimizer): The pytorch optimizer to use to perform
            candidate search.
        opti_options: Options used to control the optimization. Includes
            maxiter: Maximum number of iterations
        stop_options: Options used to control the stopping criterion. Includes
            maxiter: Maximum number of iterations
        callback: A callback function accepting the current iteration, loss,
            and gradients as arguments. This function is executed after
            computing the loss and gradients, but before calling the optimizer.
        timeout_sec: Timeout (in seconds) for optimization. If provided,
            `gen_candidates_torch` will stop after this many seconds and return
            the best solution found so far.
        gradient_samples: Number of samples to draw from the posterior
            distribution for estimating the maximum likelihood gradient.
        candidate_samples: Number of final candidate samples to return from the
            proposal distribution.

    Returns:
        2-element tuple containing

        - A set of generated candidates from the proposal distribution.
        - The acquisition value for each candidate.
    """
    acquisition_function = _adapt_acquisition(acquisition_function)
    start_time = time.monotonic()
    optimizer_options = optimizer_options or {}
    stop_options = stop_options or {}

    # Set up the optimiser
    clip_gradients(proposal_distribution)
    params = list(proposal_distribution.parameters())
    _optimizer = optimizer(params=params, **optimizer_options)  # type: ignore

    # Draw samples once to optimise against
    with torch.no_grad():
        Xs, logpX = proposal_distribution(samples=gradient_samples)

        # EDA does not differentiate through acq
        wght = acquisition_function(Xs, logpX)

    i = 0
    stop = False
    stopping_criterion = ExpMAStoppingCriterion(**stop_options)  # type: ignore
    proposal_distribution.train()
    while not stop:
        logqX = proposal_distribution.log_prob(Xs)

        if wght.ndim != logqX.ndim:
            raise RuntimeError(
                f"acquisition dim, {wght.ndim} != logp dim, " f"{logqX.ndim}"
            )
        loss = -(wght * logqX).mean()  # EDA maximum likelihood

        loss.backward()
        if callback:
            callback(i, loss.detach(), [p.grad for p in params])
        _optimizer.step()
        _optimizer.zero_grad()

        stop = stopping_criterion.evaluate(fvals=loss.detach())
        i += 1

        if timeout_sec is not None:
            runtime = time.monotonic() - start_time
            if runtime > timeout_sec:
                stop = True
                LOG.info(f"Optimization timed out after {runtime} seconds.")

    # Sample candidates
    proposal_distribution.eval()
    with torch.no_grad():
        Xcand, logqX = proposal_distribution(samples=candidate_samples)
        Xcand_acq = acquisition_function(Xcand, logqX)
    return Xcand, Xcand_acq


class VSDSolver(_CPEGenerativeSolver):

    name = "VSD"
    optim = generate_candidates_reinforce
    vacquisition = VariationalSearchAcquisition


class CbASSolver(_CPEGenerativeSolver):

    name = "CbAS"
    optim = generate_candidates_eda
    vacquisition = CbASAcquisition


def seq2int(S: ndarray, mapping: T.Dict[str, int]) -> Tensor:
    Xi = np.vectorize(mapping.__getitem__)(S)
    return torch.as_tensor(Xi).long()


def int2seq(X: Tensor, mapping: T.Dict[int, str]) -> ndarray:
    S = np.vectorize(mapping.__getitem__)(X.detach().cpu().numpy())
    return S


def setdefaults(opt: dict | None, defaults: dict) -> None:
    if opt is not None:
        defaults.update(opt)
    return defaults


def _callback(it, loss, *args, log_iters=100):
    if (it % log_iters) == 0:
        LOG.info(f"  It: {it}, Loss = {loss:.3f}")


def _val_callback(it, loss, vloss, log_iters=100):
    if (it % log_iters) == 0:
        LOG.info(f"  It: {it}, Loss = {loss:.3f}, Valid. loss = {vloss:.3f}")


def _grad_callback(it, loss, grad, log_iters=100):
    if (it % log_iters) == 0:
        mgrad = np.mean([g.detach().to("cpu").mean() for g in grad])
        LOG.info(f"  It: {it}, Loss = {loss:.3f}, Mean gradient = {mgrad:.3f}")
