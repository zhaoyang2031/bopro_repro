"""Variational distributions and priors.

This will require the `experiments` optional dependencies
"""

import importlib.util
import warnings
from abc import abstractmethod, ABC
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributions as td
import torch.nn.functional as fnn

from torch import Tensor

# Optionally import VSD and VSD functionality for experiments
if importlib.util.find_spec("vsd"):
    from vsd.utils import PositionalEncoding


class Transpose(torch.nn.Module):
    """Transpose two dimensions of a tensor.

    Parameters
    ----------
    dim0 : int, default: -1
        First dimension to swap.
    dim1 : int, default: -2
        Second dimension to swap.
    """

    def __init__(self, dim0=-1, dim1=-2) -> None:
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, X: Tensor) -> Tensor:
        """Return ``X`` with ``dim0`` and ``dim1`` swapped."""
        return X.transpose(dim0=self.dim0, dim1=self.dim1)


class Skip(torch.nn.Module):
    """Residual wrapper that adds input to module output.

    Parameters
    ----------
    nn : torch.nn.Module
        The module to wrap; must map ``x`` to a tensor broadcastable to ``x``.
    """

    def __init__(self, nn: torch.nn.Module):
        super().__init__()
        self.nn = nn

    def forward(self, x: Tensor) -> Tensor:
        return self.nn(x) + x


class FuseNorm(torch.nn.Module):
    """LayerNorm inputs and fuse: ``norm(a) + w * norm(b)``.

    Parameters
    ----------
    add_shape : int
        Feature dimension for both inputs.
    alpha0 : float, default: 1.0
        Initial value for the learnable fusion weights ``w``.
    """

    def __init__(self, add_shape: int, alpha0: float = 1.0) -> None:
        super().__init__()
        self.norm_a = torch.nn.LayerNorm(add_shape)
        self.norm_b = torch.nn.LayerNorm(add_shape)
        self.w = torch.nn.Parameter(torch.ones([add_shape]) * alpha0)

    def forward(self, ab: Tuple[Tensor, Tensor]) -> Tensor:
        a, b = ab
        add = self.norm_a(a) + self.w * self.norm_b(b)
        return add


class _TestMixin:
    """Sample consistency unit test context manager"""

    def __init__(self):
        # Test-only recording hooks
        self._test_sample_consistency: bool = False
        self._last_sample_log_prob: Tensor | float = 0.0

    @contextmanager
    def record_sample_log_prob(self):
        """Context manager: record the last sample and its log-prob.

        For unit testing.

        When enabled, `sample` will compute and store the log-probability of the
        returned samples using `self.log_prob` and make it available via
        `last_sample_log_prob()` for unit tests.
        """
        prev = self._test_sample_consistency
        self._test_sample_consistency = True
        try:
            yield self
        finally:
            self._test_sample_consistency = prev


class _TransformerBackbone(torch.nn.Module):
    """Shared embedding + positional + encoder (+ optional token head)."""

    def __init__(
        self,
        k_categories: int,
        embedding_dim: Optional[int] = None,
        nhead: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
        num_layers: int = 1,
        add_mask_token: bool = False,  # Add a mask token to the embedding of k+1
    ):
        super().__init__()

        # Embedding
        if embedding_dim is None:
            embedding_dim = max(8, k_categories // 2) * nhead
        self.e = embedding_dim
        self.k = k_categories
        self.emb = torch.nn.Embedding(
            k_categories + 1 if add_mask_token else k_categories, embedding_dim
        )

        # Positional encoding
        self.pos = PositionalEncoding(emb_size=embedding_dim)

        # Encoder
        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.tfm = torch.nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.dec = torch.nn.Linear(embedding_dim, k_categories)

    def _change_embeddings(self, e: Tensor) -> Tensor:
        return e

    def encode(self, X: Tensor) -> torch.Tensor:
        """Return encoded hidden states (B, L, e).

        This implements bi-directional masking.
        """
        return self.tfm(self.pos(self._change_embeddings(self.emb(X))))

    def load(self, other: "_TransformerBackbone"):
        for attr in ("emb", "pos", "tfm", "dec"):
            self_attr = getattr(self, attr)
            other_attr = getattr(other, attr, None)
            if self_attr is not None and other_attr is not None:
                self_attr.load_state_dict(other_attr.state_dict())

    def set_dropout_p(self, p: float):
        """Reset dropout p -- useful for multiple training steps."""
        for m in self.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = p
            # MultiheadAttention also has its own dropout parameter:
            if isinstance(m, torch.nn.MultiheadAttention):
                m.dropout = p


#
# Sequence data -- masking and mutation
#


class MaskedSearchDistribution(torch.nn.Module, ABC):
    """Abstract base class for masked/mutation search distributions, q(X|X0)."""

    def __init__(
        self,
        d_features: int,
        k_categories: int,
        X0: Optional[Tensor] = None,
        samples: int = 100,
        clip_gradients: Optional[float] = None,
    ):
        torch.nn.Module.__init__(self)
        self.d = d_features
        self.k = k_categories
        self.samples = samples
        self.clip_gradients = clip_gradients
        self.X0 = X0
        self.X0s = None

    @abstractmethod
    def log_prob(self, X: Tensor) -> Tensor: ...

    @abstractmethod
    def sample(self, sample_shape: torch.Size = torch.Size([1])) -> Tensor: ...

    def forward(self, samples: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        samples = self.samples if samples is None else samples
        with torch.no_grad():
            Xs = self.sample(torch.Size([samples]))
        logqX = self.log_prob(Xs)
        return Xs, logqX

    def set_seeds(self, X0: Tensor):
        self.X0 = X0
        self.X0s = None

    def clear_seeds(self):
        self.X0 = None
        self.X0s = None

    @torch.no_grad()
    def _sample_seeds(self, samples: int) -> Tensor:
        X0 = self._check_seeds()
        N0 = len(X0)
        if N0 < samples:  # With replacement
            self.X0s = X0[torch.randint(high=N0, size=torch.Size([samples]))]
        else:  # Without replacement
            self.X0s = X0[torch.randperm(N0)[:samples]]
        return self.X0s

    def _check_seeds(self) -> Tensor:
        if self.X0 is None:
            raise ValueError(
                "Property X0 required, instantiate the object with this "
                "property, or assign it using obj.set_seeds(X0)."
            )
        return self.X0

    def _construct_q(self, X0: Tensor) -> td.Distribution:
        raise NotImplementedError


class _TransformerMLMBackbone(_TransformerBackbone):

    def __init__(
        self,
        k_categories: int,
        embedding_dim: int | None = None,
        nhead: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0,
        num_layers: int = 1,
        pad_token: Optional[int] = None,
    ):
        _TransformerBackbone.__init__(
            self,
            k_categories=k_categories,
            embedding_dim=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_layers=num_layers,
            add_mask_token=True,
        )
        self.pad_token = pad_token
        self.mask_token = self.k

    @torch.no_grad()
    def _get_diffs(self, X0: Tensor, X: Tensor) -> Tensor:
        if X0.shape != X.shape:
            raise ValueError("X0 and X have to be the same shape.")
        mask = X != X0
        return mask

    def _construct_token_q(
        self,
        X: Tensor,
        mask: Tensor,
        replacement: bool = True,
    ) -> torch.distributions.Categorical:
        # Mask out tokens to sample
        X_mask = X.clone().masked_fill(mask, self.mask_token)

        # Get logits
        logits = self.dec(self.encode(X_mask))

        # Don't allow padding token generation
        if self.pad_token is not None:
            logits[..., self.pad_token] = float("-inf")

        # Don't re-sample existing token
        if not replacement:
            forbidden = fnn.one_hot(X, num_classes=self.k).bool()
            forbidden = forbidden & mask.unsqueeze(-1)
            logits[forbidden] = float("-inf")

        q = torch.distributions.Categorical(logits=logits)
        return q

    def _log_probx_pad_aware(
        self,
        q: td.Distribution,
        mask: Tensor,
        X: Tensor,
        X0: Optional[Tensor] = None,
    ) -> Tensor:
        if self.pad_token is not None:
            Xp = X if X0 is None else X0
            mask = mask & (Xp != self.pad_token)
        logq = q.log_prob(X).masked_fill_(~mask, 0).sum(dim=1)
        return logq

    def _log_probm_pad_aware(
        self, logits: Tensor, mask: Tensor, X: Tensor
    ) -> Tensor:
        if self.pad_token is not None:
            mask = mask & (X != self.pad_token)
        logq = torch.log_softmax(logits, dim=1).masked_fill(~mask, 0).sum(dim=1)
        return logq


class _TransformerMLMMixin(_TransformerMLMBackbone, _TestMixin):

    def __init__(
        self,
        k_categories: int,
        embedding_dim: int | None = None,
        nhead: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0,
        num_layers: int = 1,
        mask_p: float = 0.15,
        pad_token: Optional[int] = None,
    ):
        _TransformerMLMBackbone.__init__(
            self,
            k_categories=k_categories,
            embedding_dim=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_layers=num_layers,
            pad_token=pad_token,
        )
        _TestMixin.__init__(self)
        self.mask_p = mask_p

    @torch.no_grad()
    def _sample_mask(self, X: Tensor) -> Tensor:
        probs = torch.full(X.shape, self.mask_p, device=X.device)

        # Don't change padded token positions
        if self.pad_token is not None:
            probs = probs.masked_fill(X == self.pad_token, 0.0)

        mask = torch.bernoulli(input=probs).bool()
        return mask

    def _log_prob(self, X: Tensor, X0: Optional[Tensor] = None) -> Tensor:
        if X0 is None:
            mask = self._sample_mask(X)
            q = self._construct_token_q(X, mask, replacement=True)  # self mask
        else:
            mask = self._get_diffs(X0, X)
            if self._test_sample_consistency:
                mask = self._last_mask
            q = self._construct_token_q(X0, mask, replacement=True)  # seed mask

        # Token prob
        logqX = self._log_probx_pad_aware(q, mask, X, X0)
        return logqX

    @torch.no_grad()
    def _sample(
        self,
        X0: Tensor,
        gibbs_steps: int = 20,
    ) -> Tensor:
        Xs = X0.clone()
        for _ in range(gibbs_steps):
            mask = self._sample_mask(Xs)
            q = self._construct_token_q(Xs, mask, replacement=True)
            Xs[mask] = q.sample()[mask]

        # Testing only
        if self._test_sample_consistency:
            logqX = self._log_probx_pad_aware(q, mask, Xs, X0)
            self._last_mask = mask  # Consistent masking for tests
            self._last_sample_log_prob = logqX

        return Xs


class TransformerMLMProposal(MaskedSearchDistribution, _TransformerMLMMixin):
    """Masked Transformer model that randomly mutates.

    Good for using as a prior generative model.
    """

    def __init__(
        self,
        d_features: int,
        k_categories: int,
        mask_p: float = 0.15,
        X0: Optional[Tensor] = None,
        embedding_dim: Optional[int] = None,
        nhead: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
        num_layers: int = 1,
        pad_token: Optional[int] = None,
        gibbs_steps: int = 20,
        samples: int = 100,
        clip_gradients: Optional[float] = None,
    ):
        MaskedSearchDistribution.__init__(
            self,
            d_features=d_features,
            k_categories=k_categories,
            X0=X0,
            samples=samples,
            clip_gradients=clip_gradients,
        )
        _TransformerMLMMixin.__init__(
            self,
            k_categories=k_categories,
            mask_p=mask_p,
            embedding_dim=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_layers=num_layers,
            pad_token=pad_token,
        )
        self.gibbs_steps = gibbs_steps

    @torch.no_grad()
    def sample(
        self,
        sample_shape: torch.Size = torch.Size([1]),
    ) -> Tensor:
        if len(sample_shape) > 1:
            raise ValueError("Sample shapes of dim > 1 not implemented.")
        samples = int(sample_shape[0])
        X0s = self._sample_seeds(samples=samples)
        Xs = self._sample(X0s, gibbs_steps=self.gibbs_steps)
        return Xs

    def log_prob(self, X: Tensor) -> Tensor:
        X0 = None  # Evaluate against masked X0
        if self.X0s is not None:  # Evaluate against previous samples
            if X.shape != self.X0s.shape:
                raise ValueError(
                    "X must be the same shape as the last sample, or call "
                    "`self.clear_seeds()` first."
                )
            X0 = self.X0s
        return self._log_prob(X, X0)


class _TransformerMutationMixin(_TransformerMLMBackbone, _TestMixin):

    def __init__(
        self,
        k_categories: int,
        num_mutations: int = 10,
        embedding_dim: Optional[int] = None,
        nhead: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
        num_layers: int = 1,
        mask_cnn_kernel: int = 5,
        pad_token: Optional[int] = None,  # immutable
        replacement: bool = True,  # allow original token replacement
    ):
        _TransformerMLMBackbone.__init__(
            self,
            k_categories=k_categories,
            embedding_dim=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_layers=num_layers,
            pad_token=pad_token,
        )
        _TestMixin.__init__(self)
        self.num_mutations = num_mutations
        self.replacement = replacement
        if mask_cnn_kernel % 2 == 0:
            raise ValueError("mask_cnn_kernel must be odd.")

        out_logit = torch.nn.Linear(in_features=self.e, out_features=1)
        with torch.no_grad():
            out_logit.bias.data.copy_(torch.tensor([-2.0]))  # start mask p low

        # Mask model
        self.mask_dec = torch.nn.Sequential(
            FuseNorm(self.e, alpha0=1e-5),
            Skip(
                torch.nn.Sequential(
                    Transpose(),
                    torch.nn.Conv1d(
                        in_channels=self.e,
                        out_channels=self.e,
                        kernel_size=mask_cnn_kernel,
                        dilation=1,
                        padding=mask_cnn_kernel // 2,
                    ),
                    torch.nn.SiLU(),
                    Transpose(),
                )
            ),
            torch.nn.LayerNorm(normalized_shape=self.e),
            torch.nn.LeakyReLU(),  # "Crisper" logits
            out_logit,
        )

    def load(self, other: _TransformerMLMBackbone):
        _TransformerMLMBackbone.load(self, other)
        if hasattr(other, "mask_dec"):
            self.mask_dec.load_state_dict(other.mask_dec.state_dict())

    def _mask_logits_pad_aware(self, X0: Tensor) -> Tensor:
        # Skip around the backbone transformer if needed
        emb = self.pos(self._change_embeddings(self.emb(X0)))
        enc = self.tfm(emb)
        logits = self.mask_dec((emb, enc)).squeeze(-1)

        # Don't change padded token positions
        if self.pad_token is not None:
            logits = logits.masked_fill(X0 == self.pad_token, float("-inf"))

        return logits

    def _log_prob(self, X: Tensor, X0: Tensor) -> Tensor:
        mask = self._get_diffs(X0, X)

        with torch.no_grad():
            dmuts = mask.sum(dim=1)
            if (dmuts > self.num_mutations).any():
                warnings.warn(
                    f"More than {self.num_mutations} encountered!",
                    RuntimeWarning,
                )

        # Mask prob
        logits = self._mask_logits_pad_aware(X0)
        logqm = self._log_probm_pad_aware(logits, mask, X0)

        # Token prob
        q = self._construct_token_q(X0, mask, replacement=self.replacement)
        logqX = self._log_probx_pad_aware(q, mask, X, X0)

        return logqX + logqm

    @torch.no_grad()
    def _sample(
        self,
        X0: Tensor,
    ) -> Tensor:
        Xs = X0.clone()

        # Sample mask/mutations
        logits = self._mask_logits_pad_aware(X0)

        pos = torch.multinomial(
            fnn.softmax(logits, dim=-1),
            num_samples=self.num_mutations,
            replacement=False,  # require num_mutations count in sample
        )
        mask = torch.zeros_like(Xs).bool()
        mask.scatter_(dim=1, index=pos, value=True)

        # Sample tokens
        q = self._construct_token_q(X0, mask, replacement=self.replacement)
        Xs[mask] = q.sample()[mask]

        # Testing only
        if self._test_sample_consistency:
            logqX = self._log_probx_pad_aware(q, mask, Xs, X0)
            logqm = self._log_probm_pad_aware(logits, mask, X0)
            self._last_sample_log_prob = logqX + logqm

        return Xs


class TransformerMutationProposal(
    MaskedSearchDistribution, _TransformerMutationMixin
):
    """Masked Transformer model that learns how to mutate."""

    prior_same_class = False  # Use the MLM as a prior

    def __init__(
        self,
        d_features: int,
        k_categories: int,
        num_mutations: int = 10,
        X0: Optional[Tensor] = None,
        embedding_dim: Optional[int] = None,
        nhead: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
        num_layers: int = 1,
        mask_cnn_kernel: int = 5,
        pad_token: Optional[int] = None,
        replacement: bool = False,  # allow original token replacement
        samples: int = 100,
        clip_gradients: Optional[float] = None,
    ):
        self._save_constructor_args(locals())
        MaskedSearchDistribution.__init__(
            self,
            d_features=d_features,
            k_categories=k_categories,
            X0=X0,
            samples=samples,
            clip_gradients=clip_gradients,
        )
        _TransformerMutationMixin.__init__(
            self,
            k_categories=k_categories,
            num_mutations=num_mutations,
            embedding_dim=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_layers=num_layers,
            mask_cnn_kernel=mask_cnn_kernel,
            pad_token=pad_token,
            replacement=replacement,
        )

    @torch.no_grad()
    def sample(
        self,
        sample_shape: torch.Size = torch.Size([1]),
    ) -> Tensor:
        if len(sample_shape) > 1:
            raise ValueError("Sample shapes of dim > 1 not implemented.")
        samples = int(sample_shape[0])
        X0s = self._sample_seeds(samples=samples)
        Xs = self._sample(X0s)
        return Xs

    def log_prob(self, X: Tensor) -> Tensor:
        if self.X0s is not None:  # Evaluate against previous samples
            if X.shape != self.X0s.shape:
                raise ValueError(
                    "X must be the same shape as the last sample, or call "
                    "`obj.clear_seeds()` first."
                )
            X0 = self.X0s
        else:  # Evaluate against internal seeds
            X0 = self._check_seeds()
            if X.shape != X0.shape:
                raise ValueError(
                    "X must be the same shape as the internal seeds, obj.X0!"
                )
        return self._log_prob(X, X0)

    def _save_constructor_args(self, local_vars: Dict[str, Any]):
        self._constructor_args = {
            k: v
            for k, v in local_vars.items()
            if k not in ("self", "__class__")
        }

    def get_constructor_args(self) -> Dict[str, Any]:
        if not hasattr(self, "_constructor_args"):
            raise ValueError("Consturctor arguments not saved!")
        return self._constructor_args.copy()

    def get_compatible_prior(self) -> TransformerMLMProposal:
        kwargs = self.get_constructor_args()
        # Pop irrelevant items
        for i in ("mask_cnn_kernel", "num_mutations", "replacement"):
            kwargs.pop(i)
        return TransformerMLMProposal(**kwargs)
