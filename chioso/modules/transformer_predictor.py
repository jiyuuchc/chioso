import flax.linen as nn
import jax.numpy as jnp
import jax

class TransformerBlock(nn.Module):
    n_heads: int
    dim: int
    dropout: float
    ff_dropout: float
    deterministic: bool=False
    
    @nn.compact
    def __call__(self, x, *, deterministic=None):
        if deterministic is None:
            deterministic = self.deterministic
        
        x, x_pos, mask = x
        if x_pos is None:
            x_pos = jnp.zeros_like(x)
        
        shortcut = x

        x = nn.MultiHeadDotProductAttention(
            self.n_heads, dropout_rate=self.dropout
        )(x + x_pos, x + x_pos, x, mask=mask, deterministic=deterministic)

        x = nn.LayerNorm()(x+shortcut)
        
        shortcut = x

        x = nn.Dense(self.dim)(x)
        x = jax.nn.gelu(x)
        x = nn.Dropout(self.ff_dropout)(x, deterministic=deterministic)
        x = nn.Dense(shortcut.shape[-1])(x)
        x = nn.Dropout(self.ff_dropout)(x, deterministic=deterministic)
        x = nn.LayerNorm()(x + shortcut)
        
        x = nn.LayerNorm()(x+shortcut)
        
        return x


class SCTransformer(nn.Module):
    n_genes: int
    dim_out: int = 68

    dim_hidden: int = 256
    n_heads: int = 4
    n_layers: int = 6
    dropout: float = 0.1
    cnts_binning: tuple[int] = (1,2,4,8,16,32,64,128,256)

    n_added_tokens: int = 1
    
    @nn.compact
    def __call__(self, gids, cnts, *, training=False):
        # cnts = jnp.clip(cnts, 0, 511)
        mask = gids >= 0
        x = nn.Embed(self.n_genes, self.dim_hidden)(gids)
        cnts = jnp.digitize(cnts, jnp.asarray(self.cnts_binning))
        x = x + nn.Embed(len(self.cnts_binning)+1, self.dim_hidden)(cnts)

        # add additional token
        x = jnp.r_[x, jnp.zeros([self.n_added_tokens, self.dim_hidden], dtype=x.dtype)]
        mask = jnp.r_[mask, jnp.asarray([True] * self.n_added_tokens)]

        for _ in range(self.n_layers):
            x = TransformerBlock(
                self.n_heads,
                self.dim_hidden,
                self.dropout,
                self.dropout,
            )((x, None, mask), deterministic=not training)

        self.sow("intermediates", "latent", x[-self.n_added_tokens:])

        out = nn.Dense(self.dim_out)(x[-self.n_added_tokens:])
        
        return out
