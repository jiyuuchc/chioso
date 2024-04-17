from __future__ import annotations

from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from jax import Array
from jax.typing import ArrayLike


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
