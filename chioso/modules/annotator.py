from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from jax import Array
from jax.typing import ArrayLike

from .stem import dummy_stem
from .predictor import MLP

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

class CellAnnotator(nn.Module):
    tokens: Array
    roi: int = 8
    att_ks: int = 8
    n_masks: int = 1
    dsc_n_layers: int = 4
    normalize: bool = False
    learned_scaling:bool = False
    
    def att(self, x0, weights):
        _, x0 = jax.lax.scan(
            lambda carry, s: (None, jax.lax.conv_general_dilated_local(
                s[None, :, :, None],
                weights[:,:,:,None],
                (1,1),
                "same",
                (self.roi, self.roi),
                dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            ).squeeze(0).squeeze(-1)), 
            None,
            x0.transpose(2, 0, 1),
        )
        x0 = x0.transpose(1,2,0)
        return x0

    def att_simple(self, x0, weights):
        x = jax.lax.conv(
            x0.transpose(2,0,1)[:, None, :, :],
            weights[None, None, :, :],
            [1,1],
            "SAME",
        )
        x = x.squeeze(1).transpose(1,2,0)

        return x
        
    @nn.compact
    def __call__(self, sg, ref_inputs=None, *, training=False):
        gamma = self.param(
            "gamma", lambda rng, shape: jnp.zeros(shape), (sg.n_genes)
        )

        sg = sg.replace(data = sg.data * jnp.exp(gamma[sg.indices]))

        x0 = dummy_stem(sg, self.tokens)

        weights = nn.sigmoid(
            self.param("weights", nn.initializers.truncated_normal(), (self.n_masks, self.att_ks, self.att_ks))
        )
        weights = weights.at[:, (self.att_ks-1)//2, (self.att_ks-1)//2].set(1.0) # fixed weight for the center location 
        weights = jax.image.resize(weights, (self.n_masks, self.roi, self.roi), "linear")

        if self.learned_scaling:
            x0 = x0 * jnp.exp(nn.Dense(1)(x))

        if self.normalize:
            cnts = dummy_stem(sg, jnp.ones([self.tokens.shape[0], 1]))            

        def scan_fn(_, w):
            result = self.att_simple(x0, w)

            if self.normalize:
                total_cnts = self.att_simple(cnts, w)
                result /= total_cnts + 1e-6

            return None, result

        _, x = jax.lax.scan(
            scan_fn,
            None,
            weights,
        )
        # x = self.att_simple(x0, weights)

        x = gradient_reversal(x)

        dsc = MLP(1, self.dsc_n_layers, deterministic=True)

        dsc_x = dsc(x)
        idx = jnp.argmin(dsc_x, axis=0, keepdims=True)
        dsc_x = jnp.take_along_axis(dsc_x, idx, 0).squeeze(0)
        x_out = jnp.take_along_axis(x, idx, 0).squeeze(0)
        dsc_loss_main = optax.sigmoid_binary_cross_entropy(dsc_x, jnp.ones_like(dsc_x))

        if ref_inputs is not None:
            dsc_y = dsc(ref_inputs)
            dsc_loss_ref = optax.sigmoid_binary_cross_entropy(dsc_y, jnp.zeros_like(dsc_y))
        else:
            dsc_loss_ref = None
            
        return dict(
            dsc_loss_main = dsc_loss_main,
            dsc_loss_ref = dsc_loss_ref,
            output = x_out,
        )
