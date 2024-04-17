import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from .attention import CellAnnotator

@jax.custom_vjp
def gradient_reversal(x):
    return x

def _gr_fwd(x):
    return x, None

def _gr_bwd(_, g):
    return (jax.tree_util.tree_map(lambda v: -v, g),)

gradient_reversal.defvjp(_gr_fwd, _gr_bwd)
""" A gradient reveral op. 

    This is a no-op during inference. During training, it does nothing
    in the forward pass, but reverse the sign of gradient in backward
    pass. Typically placed right before a discriminator in adversal 
    training.
"""

class Discriminator(nn.Module):
    n_layers: int = 4

    @nn.compact
    def __call__(self, x):
        dim = x.shape[-1]
        for _ in range(self.n_layers):
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
            x = nn.LayerNorm()(x)
        x = nn.Dense(1)(x)
        return x

class AdvModel(nn.Module):
    predictor: CellAnnotator
    dsc: Discriminator

    def __call__(self, mosta_data, tome_data):
        sg, mask = mosta_data

        x, binary_pred = self.predictor(sg)
        
        x = x.reshape(-1, x.shape[-1])
        x = gradient_reversal(x)
        pred_x = self.dsc(x)

        if mask is not None:
            mask = (mask >= 0.5).astype("float32").reshape(-1, 1)
        else:
            mask = (binary_pred >= 0).astype("float32").reshape(-1, 1)

        dsc_loss_x = optax.sigmoid_binary_cross_entropy(
            pred_x,
            jnp.ones_like(pred_x),
        ).sum(where=mask)

        pred_y = self.dsc(tome_data)
        dsc_loss_y = optax.sigmoid_binary_cross_entropy(
            pred_y,
            jnp.zeros_like(pred_y)
        ).sum()
        
        return dict(
            dsc_loss_x=dsc_loss_x, 
            dsc_loss_y=dsc_loss_y, 
            prediction=binary_pred,
        )
