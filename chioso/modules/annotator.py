from typing import Optional, Sequence

from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp

from jax import Array
from jax.typing import ArrayLike

from .stem import dummy_stem
from .predictor import MLP
from ..data import SGData2D

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    rate: float

    @nn.module.compact
    def __call__(
        self, inputs: ArrayLike, deterministic: Optional[bool] = True
    ) -> Array:
        if self.rate == 0.0:
            return inputs
        keep_prob = 1.0 - self.rate
        if deterministic:
            return inputs
        else:
            rng = self.make_rng("dropout")
            binary_factor = jnp.floor(
                keep_prob + jax.random.uniform(rng, dtype=inputs.dtype)
            )
            output = inputs / keep_prob * binary_factor
            return output


class _Block(nn.Module):
    """ConvNeXt Block.
    Args:
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    drop_rate: int = 0.4
    layer_scale_init_value: float = 1e-6
    kernel_size: int = 7

    @nn.compact
    def __call__(self, x: ArrayLike, *, training: Optional[bool] = None) -> Array:
        dim = x.shape[-1]
        ks = self.kernel_size
        scale = self.layer_scale_init_value

        shortcut = x

        x = nn.Conv(dim, (ks, ks), feature_group_count=dim)(x)
        x = nn.LayerNorm(epsilon=1e-6)(x)
        x = nn.Dense(dim * 4)(x)
        x = jax.nn.gelu(x)
        x = nn.Dense(dim)(x)

        if scale > 0:
            gamma = self.param(
                "gamma", lambda rng, shape: scale * jnp.ones(shape), (x.shape[-1])
            )
            x = x * gamma

        deterministic = training is None or not training
        x = DropPath(self.drop_rate)(x, deterministic=deterministic)

        x = x + shortcut

        return x


""" Implements the convnext encoder. Described in https://arxiv.org/abs/2201.03545
Original implementation: https://github.com/facebookresearch/ConvNeXt
"""
class ConvNeXt(nn.Module):
    """ConvNeXt CNN backbone

    Attributes:
        patch_size: Stem patch size
        depths: Number of blocks at each stage.
        dims: Feature dimension at each stage.
        drop_path_rate: Stochastic depth rate.
        layer_scale_init_value: Init value for Layer Scale.
    """

    patch_size: int = 4
    depths: Sequence[int] = (3, 3, 27, 3)
    dims: Sequence[int] = (96, 192, 384, 768)
    drop_path_rate: float = 0.0
    layer_scale_init_value: float = 1e-6

    @nn.compact
    def __call__(
        self, x: ArrayLike, *, training: Optional[bool] = None
    ) -> list[Array]:
        """
        Args:
            x: Image input.
            training: Whether run the network in training mode (i.e. with stochastic depth)

        Returns:
            A list of features at various scales
        """
        dp_rate = 0
        outputs = []
        for k in range(len(self.depths)):
            if k == 0:
                ps = self.patch_size
                x = nn.Conv(self.dims[k], (ps, ps), strides=(ps, ps))(x)
                x = nn.LayerNorm(epsilon=1e-6)(x)
            else:
                x = nn.LayerNorm(epsilon=1e-6)(x)
                x = nn.Conv(self.dims[k], (2, 2), strides=(2, 2))(x)

            for _ in range(self.depths[k]):
                x = _Block(dp_rate, self.layer_scale_init_value)(x, training=training)
                dp_rate += self.drop_path_rate / (sum(self.depths) - 1)

            outputs.append(x)

        # keys = [str(k + 1 if self.patch_size == 2 else k + 2) for k in range(4)]
        # encoder_out = dict(zip(keys, outputs))

        return outputs


class FPN(nn.Module):
    out_channels: int = 256

    @nn.compact
    def __call__(self, inputs: Sequence[ArrayLike]) -> Sequence[Array]:
        out_channels = self.out_channels

        outputs = [jax.nn.relu(nn.Dense(out_channels)(x)) for x in inputs]

        for k in range(len(outputs) - 1, 0, -1):
            x = jax.image.resize(outputs[k], outputs[k - 1].shape, "nearest")
            x += outputs[k - 1]
            x = nn.Conv(out_channels, (3, 3))(x)
            x = jax.nn.relu(x)
            outputs[k - 1] = x

        return outputs

class CellAnnotator(nn.Module):
    embed: jax.Array
    depths: tuple[int] = (3,9,3)
    dims: tuple[int] = (256, 384, 512)
    dropout: float = 0.4
    fpn_dim: int = 384
    roi: int = 8
    att_ks: int = 8
    normalize: bool = False
    training: bool = False
    
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

    @nn.compact
    def __call__(self, sg, *, training=None):
        if training is None:
            training = self.training
        # sg = SGData2D(cnts, gids, indptr, self.shape2d, self.embed.shape[0])
        gamma = self.param(
            "gamma", lambda rng, shape: jnp.zeros(shape), (sg.n_genes)
        )
        x0 = dummy_stem(sg, self.embed, gamma=gamma)

        x = ConvNeXt(1,  depths=self.depths, dims=self.dims, drop_path_rate=self.dropout)(x0, training=training)
        x = FPN(self.fpn_dim)(x)[0]

        self.sow("intermediates", "features", x)

        weights = nn.sigmoid(MLP(self.att_ks ** 2, 3)(x, deterministic=True))
        weights = weights.reshape(weights.shape[:-1] + (self.att_ks, self.att_ks))
        weights = jax.image.resize(weights, weights.shape[:2] + (self.roi, self.roi), "linear")
        weights = weights.reshape(weights.shape[:2] + (self.roi**2,))

        self.sow("intermediates", "att_weights", weights)

        out = self.att(x0, weights)

        if self.normalize:
            cnts = dummy_stem(sg, jnp.ones([self.embed.shape[0], 1]))
            total_cnts = self.att(cnts, weights)
            out = out / (total_cnts + 1e-6)

        return out
