import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from .attention import CellAnnotator
from .predictor import MLP
from ..data import SGData2D

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
        return MLP(1, self.n_layers)(x)

class AdvModel(nn.Module):
    embedding: jax.Array
    dsc_n_layers: int = 4
    patch_size: tuple[int] = (128, 128)
    momentum: float = 0.9

    def setup(self):
        self.predictor = CellAnnotator(self.embedding)
        self.dsc = Discriminator(self.dsc_n_layers)
        self.fg_norm = self.variable('stats', "fg_norm", jnp.zeros, [1])

    def __call__(self, mosta_data, tome_data, *, training=False):
        (data, indices, indptr), mask = mosta_data
        sg = SGData2D(data, indices, indptr, self.patch_size, self.embedding.shape[0])

        x, fg = self.predictor(sg, training=training)

        if mask is not None:
            mask_loss = optax.sigmoid_binary_cross_entropy(fg, mask).mean()

        else:
            is_initialized = self.has_variable('stats', 'fg_norm')
            if is_initialized:
                self.fg_norm.value = self.fg_norm.value * self.momentum + fg.mean() * (1-self.momentum)
                fg = fg - self.fg_norm.value
            mask = nn.sigmoid(fg)
            mask_loss = None

        x = gradient_reversal(x)
        pred_x = self.dsc(x)[..., 0]

        dsc_loss_x = optax.sigmoid_binary_cross_entropy(
            pred_x,
            jnp.ones_like(pred_x),
        ).sum(where=mask>=.5)/128/128
        # dsc_loss_x *= mask
        # dsc_loss_x = dsc_loss_x.mean() 

        pred_y = self.dsc(tome_data)
        dsc_loss_y = optax.sigmoid_binary_cross_entropy(
            pred_y,
            jnp.zeros_like(pred_y)
        ).mean()
        
        return dict(dsc_loss_x=dsc_loss_x, dsc_loss_y=dsc_loss_y, mask_loss=mask_loss)
