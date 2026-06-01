import inspect
import math
import sys
from abc import ABC, abstractmethod
from functools import cache

import torch
import torch.nn as nn
from torch.distributions import Distribution


class Loss(nn.Module, ABC):
    def __init__(
        self,
        reg_factor: float | torch.Tensor = 0.0,
        model: nn.Module | None = None,
        prior: Distribution | None = None,
        exp_reg: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not torch.is_tensor(reg_factor):
            reg_factor = torch.as_tensor(reg_factor)
        self.register_buffer("reg_factor", reg_factor)
        self.prior = prior
        self.model = model
        self.exp_reg = exp_reg
        if reg_factor > 0:
            if model is None:
                raise TypeError(
                    "Proposal model instance must be provided for regularization"
                )
            parameters = nn.utils.parameters_to_vector(model.parameters()).detach()
            self.register_buffer("initial_parameters", parameters)

    @cache
    def _prior_logits(self, x: torch.Tensor) -> torch.Tensor:
        if self.prior is None:
            return torch.zeros(x.shape[:-1]).to(x)
        else:
            return self.prior.to(device=x.device).log_prob(x)

    @abstractmethod
    def _loss(
        self,
        log_probs: torch.Tensor,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pointwise losses based on log probabilities."""
        pass

    def forward(
        self,
        log_probs: torch.Tensor,
        x: torch.Tensor,
        u: torch.Tensor,
        logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the loss.
        Args:
            log_probs (torch.Tensor): Log probabilities of x under the proposal.
            x (torch.Tensor): Input samples.
            u (torch.Tensor): Utility or target values.
            logits (torch.Tensor | None): Optional logits for weighting.
        Returns:
            torch.Tensor: The computed loss.
        """
        if logits is None:
            logits = torch.zeros_like(u)
        point_losses = self._loss(log_probs, x, u)
        weighted_losses = point_losses / logits.exp()
        loss = weighted_losses.mean(-1)
        if self.reg_factor > 0:
            parameters = nn.utils.parameters_to_vector(self.model.parameters())
            squared_distance = (
                (parameters - self.initial_parameters.to(parameters)).pow(2).mean()
            )
            if self.exp_reg:
                loss += self.reg_factor * squared_distance.exp()
            else:
                loss += self.reg_factor * squared_distance
        return loss


class ForwardKL(Loss):
    def _loss(
        self,
        log_probs: torch.Tensor,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the forward KL loss from log probabilities.
        """
        log_ps_loss = (
            -log_probs
            * u
            * (
                self._prior_logits(x)
                - torch.logsumexp(self._prior_logits(x), -1)
                + math.log(x.shape[-2])
            ).exp()
        )
        return log_ps_loss


class BalancedForwardKL(ForwardKL):
    def _loss(
        self,
        log_probs: torch.Tensor,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the balanced forward KL loss from log probabilities.
        """
        # loss = -log_probs * u + log_probs.exp()
        loss = super()._loss(log_probs, x, u)
        total_loss = loss + log_probs.exp()
        return total_loss


def make_pairs(x: torch.Tensor) -> torch.Tensor:
    """Make pairs of the input data

    Args:
        x (torch.Tensor): (..., N, D)-shaped tensor

    Returns:
        torch.Tensor: (..., 2, N/2, D)-shaped tensor
    """
    shape = x.shape
    if x.shape[-2] % 2 != 0:
        raise AssertionError("There must be an even number of data points")
    return x.reshape(*shape[:-2], 2, shape[-2] // 2, shape[-1])


class PairedLoss(Loss):
    def forward(
        self,
        log_probs: torch.Tensor,
        x: torch.Tensor,
        u: torch.Tensor,
        logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for paired losses, operating on pairs of log probabilities.
        """
        log_prob_pairs = make_pairs(log_probs.unsqueeze(-1)).squeeze(-1)
        log_prior = self._prior_logits(x)
        log_prior_pairs = make_pairs(log_prior.unsqueeze(-1)).squeeze(-1)
        u_pairs = make_pairs(u.unsqueeze(-1)).squeeze(-1)
        if logits is not None:
            logit_pairs = make_pairs(logits.unsqueeze(-1)).squeeze(-1).sum(-2)
        else:
            logit_pairs = torch.zeros(u_pairs.shape[:-2] + u_pairs.shape[-1:]).to(
                u_pairs
            )
        return super().forward(log_prob_pairs, log_prior_pairs, u_pairs, logit_pairs)


class PreferenceLoss(PairedLoss):
    def __init__(
        self, *args, temperature: float = 1.0, nugget: float = 1e-15, **kwargs
    ):
        """Preference loss with an optional nugget for numerical stability."""
        super().__init__(*args, **kwargs)
        self.nugget = nugget
        self.temperature = temperature

    def _pairs_loss(
        self,
        log_q_pairs: torch.Tensor,
        log_prior_pairs: torch.Tensor,
        u_pairs: torch.Tensor,
    ) -> torch.Tensor:
        log_qs_diff = log_q_pairs[..., 0, :] - log_q_pairs[..., 1, :]
        log_ps_diff = log_prior_pairs[..., 0, :] - log_prior_pairs[..., 1, :]
        us_diff = u_pairs[..., 0, :] - u_pairs[..., 1, :]
        log_ps = self.temperature * us_diff.sign() * (log_qs_diff - log_ps_diff)
        loss = -(torch.sigmoid(log_ps) + self.nugget).log()
        return loss

    def _loss(
        self, log_probs: torch.Tensor, x: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute preference loss from log probabilities.
        """
        loss = self._pairs_loss(log_probs, x, u)
        return loss


class RobustPreferenceLoss(PreferenceLoss):
    """
    Unbiased preference loss function from Provably Robust DPO (Chowdhury et al., 2024)
    """

    def __init__(
        self,
        *args,
        flip_prob: float | torch.Tensor = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if (flip_prob > 0.5) or (flip_prob < 0):
            raise ValueError("Flip probability must be between 0 and 0.5")
        if not torch.is_tensor(flip_prob):
            flip_prob = torch.as_tensor(flip_prob)
        self.register_buffer("flip_prob", flip_prob)

    def _loss(
        self, log_probs: torch.Tensor, x: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the robust preference loss with label flip adjustment.
        """
        loss_orig = self._pairs_loss(log_probs, x, u)
        loss_flip = self._pairs_loss(log_probs, x, u.flip(dims=[-2]))
        loss = (1 - self.flip_prob) * loss_orig - self.flip_prob * loss_flip
        return loss


def loss_classes():
    """List available loss classes from genbo.losses"""
    loss_names = []
    for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass):
        if issubclass(obj, Loss) and obj is not Loss and not inspect.isabstract(obj):
            loss_names.append(name)
    return loss_names


loss_registry = {k: getattr(sys.modules[__name__], k) for k in loss_classes()}
