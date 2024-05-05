from typing import Callable

import flax.linen as nn
import jax.numpy as jnp

class SCEmbed(nn.Module):
    n_genes: int 
    dim: int = 256
    normalize: bool = False
    log_transform: bool = False

    @nn.compact
    def __call__(self, gids, cnts):
        mask = gids >= 0
        cnts = jnp.where(mask, cnts, 0)

        if self.log_transform:
            cnts = jnp.log1p(cnts)

        x = nn.Embed(self.n_genes, self.dim)(gids)
        if self.normalize:
            cnts = cnts / cnts.sum(axis=-1, keepdims=True)

        x = cnts @ x

        return x


class MLP(nn.Module):
    out_dim: int = 256
    hidden_layers: int = 3
    hidden_dim: int = -1
    dropout: float = 0.2
    deterministic: bool = False

    @nn.compact
    def __call__(self, x, *, deterministic=None):
        deterministic = deterministic or self.deterministic

        hidden_dim = self.hidden_dim
        if hidden_dim < 0:
            hidden_dim = x.shape[-1]

        for _ in range(self.hidden_layers):
            x = nn.Dense(hidden_dim, use_bias=False)(x)
            x = nn.Dropout(self.dropout)(x, deterministic=deterministic)
            x = nn.LayerNorm(use_scale=False)(x)
            x = nn.gelu(x)

        x = nn.Dense(self.out_dim)(x)
        return x


class LinearPredictor(nn.Module):
    n_genes: int
    dim_out: int
    dim_hidden: int = 256
    n_layers: int = 6
    dropout: float =0.2    
    normalize: bool = True
    log_transform: bool = False
    
    def setup(self):
        self.embed = SCEmbed(self.n_genes, self.dim_hidden, self.normalize, self.log_transform)
        self.mlp = MLP(self.dim_out, self.n_layers, dropout=self.dropout)

    def __call__(self, gids, cnts, *, training=False):
        x = self.embed(gids, cnts)
        self.sow("intermediates", "feature", x)
        x = self.mlp(x, deterministic=not training)
        return x
