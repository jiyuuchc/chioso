from __future__ import annotations

from typing import Sequence, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp

from jax.typing import ArrayLike
from jax import Array

from ..data.sgdata import SGData2D


class FFN(nn.Module):
    """A feed-forward block commonly used in transformer"""

    dim: int
    dropout_rate: float = 0.0
    deterministic: bool = False

    @nn.compact
    def __call__(self, x, *, deterministic=None):
        deterministic = deterministic or self.deterministic

        shortcut = x

        x = nn.Dense(self.dim)(x)
        x = jax.nn.gelu(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)
        x = nn.Dense(shortcut.shape[-1])(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)
        x = shortcut + x
        x = nn.LayerNorm()(x)

        return x


class SGAttention(nn.Module):
    n_heads: int
    dims: int | None = None
    dropout_rate: float = 0
    derterministic: bool = False

    @nn.compact
    def __call__(self, token, ctx, segms, rpts, entry_mask, *, deterministic=None):
        EPS = jnp.finfo("float32").eps

        deterministic = deterministic or self.derterministic
        dims = self.dims or ctx.shape[-1]

        n_entries = ctx.shape[0]
        n_pixles = token.shape[0]

        shortcut = token

        query = nn.Dense(dims)(token)

        query = jnp.repeat(query, rpts, axis=0, total_repeat_length=n_entries)
        query = query.reshape(n_entries, self.n_heads, dims // self.n_heads)
        query /= jnp.sqrt(query.shape[-1])

        keys = nn.Dense(dims)(ctx).reshape(
            n_entries, self.n_heads, dims // self.n_heads
        )
        vals = nn.Dense(dims)(ctx).reshape(
            n_entries, self.n_heads, dims // self.n_heads
        )

        weights = (query * keys).sum(axis=-1, keepdims=True)
        offsets = jax.ops.segment_max(weights, segms, num_segments=n_pixles)
        offsets = jnp.repeat(offsets, rpts, axis=0, total_repeat_length=n_entries)
        weights = jnp.exp(weights - offsets)  # [N, n_heads, 1]
        weights = jnp.where(entry_mask[:, None, None], weights, 0)

        norm_factor = jax.ops.segment_sum(weights, segms, num_segments=n_pixles)
        norm_factor = jnp.repeat(
            norm_factor, rpts, axis=0, total_repeat_length=n_entries
        )
        weights = weights / (norm_factor + EPS)

        if not deterministic and self.dropout_rate > 0:
            rng = self.make_rng("dropout")
            mask = (
                jax.random.uniform(rng, shape=weights.shape) + self.dropout_rate
            ) >= 1
            weights = jnp.where(mask, 0, weights) / (1 - self.dropout_rate)

        results = jax.ops.segment_sum(weights * vals, segms, num_segments=n_pixles)
        results = results.reshape(n_pixles, dims)

        results = nn.Dense(dims)(results)

        token = nn.LayerNorm()(shortcut + results)

        return token


class SGAttentionStem(nn.Module):
    dims: int = 64
    ffn_dims: int = 192
    n_layers: int = 3
    n_heads: int = 4
    att_dropout_rate: float = 0
    dropout_rate: float = 0
    deterministic: bool = False
    counts_bin: Sequence[int] = (1, 2, 4, 8, 16, 32, 64)

    @nn.compact
    def __call__(self, sgdata, *, training=False):
        indices = sgdata.indices
        indptr = sgdata.indptr
        cnts = sgdata.data

        n_pixels = sgdata.shape[0] * sgdata.shape[1]
        n_reads = indptr[-1]
        n_entries = indices.shape[0]

        if cnts.shape[0] != n_entries:
            raise ValueError(
                "invalid SGData: lengths of cnts and indices array are not equal"
            )

        rpt = jnp.diff(indptr)
        segms = jnp.repeat(
            jnp.arange(len(rpt)),
            rpt,
            total_repeat_length=n_entries,
        )

        entry_mask = jnp.arange(cnts.shape[0]) < n_reads

        ctx = nn.Embed(sgdata.n_genes, self.dims)(indices)
        digitized_cnts = jnp.digitize(cnts, jnp.asarray(self.counts_bin))
        ctx += nn.Embed(len(self.counts_bin) + 1, self.dims)(digitized_cnts)
        ctx = nn.Dense(self.dims)(ctx)

        x = jnp.zeros([n_pixels, self.dims])
        for _ in range(self.n_layers):
            x = SGAttention(
                self.n_heads,
                self.dims,
                dropout_rate=self.att_dropout_rate,
            )(x, ctx, segms, rpt, entry_mask, deterministic=not training)

            x = FFN(
                self.ffn_dims,
                dropout_rate=self.dropout_rate,
            )(x, deterministic=not training)

        x = x.reshape([sgdata.shape[0], sgdata.shape[1], self.dims])

        return x


def dummy_stem(sg, tokens, normalize=False, gamma=None):
    indices = sg.indices
    cnts = sg.data
    indptr = sg.indptr

    n_entries = indices.shape[0]
    n_pixels = sg.shape[0] * sg.shape[1]
    masking = jnp.arange(n_entries) < indptr[-1]

    cnts = jnp.where(masking, cnts, 0)
    if gamma is not None:
        cnts = cnts * jnp.exp(gamma[indices])

    rpt = jnp.diff(indptr)
    segms = jnp.repeat(
        jnp.arange(len(rpt)),
        rpt,
        total_repeat_length=n_entries,
    )

    embeddings = tokens[indices] * cnts[:, None]

    sum_cnts = jax.ops.segment_sum(
        cnts,
        segms,
        num_segments=n_pixels,
    )
    embedding_sum = jax.ops.segment_sum(
        embeddings,
        segms,
        num_segments=n_pixels,
    )

    if normalize:
        out = embedding_sum / (sum_cnts[:, None] + 1e-8)
    else:
        out = embedding_sum

    out = out.reshape(*sg.shape, tokens.shape[-1])

    return out


class SGStem(nn.Module):
    feature_dim: int
    normalize: bool = False
    rescale: bool = True
    token_init: Callable = nn.initializers.variance_scaling(1.0, 'fan_in', 'normal', out_axis=0)

    @nn.compact
    def __call__(self, sg:SGData2D, *, rescale:bool=None, normalize:bool=None, training:bool=False):
        if rescale is None:
            rescale = self.rescale

        if normalize is None:
            normalize = self.normalize

        gamma = self.param(
            "gamma", lambda rng, shape: jnp.zeros(shape), (sg.n_genes)
        )

        tokens = self.param("tokens", self.token_init, (sg.n_genes, self.feature_dim))

        if rescale:
            out = dummy_stem(sg, tokens, normalize=normalize, gamma=gamma)
        else:
            out = dummy_stem(sg, tokens, normalize=normalize)

        return out

