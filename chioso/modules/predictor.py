import flax.linen as nn
import jax.numpy as jnp

class SCLinear(nn.Module):
    n_genes: int 
    n_out: int 

    n_layers: int = 6
    hidden_dim: int = 256
    dropout: float = 0.2
    
    @nn.compact
    def __call__(self, gids, cnts, *, training=False):
        mask = gids >= 0
        cnts = jnp.where(mask, cnts, 0)
        x = nn.Embed(self.n_genes, self.hidden_dim)(gids)
        cnts = cnts / cnts.sum()
        x = cnts @ x
        
        for _ in range(self.n_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.Dropout(self.dropout)(x, deterministic=not training)
            x = nn.relu(x)
            x = nn.LayerNorm(use_scale=False)(x)

        out = nn.Dense(self.n_out)(x)
        
        return out
