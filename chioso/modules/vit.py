"""Vision Transformer.
Based on scenic implementation: https://github.com/google-research/scenic/
"""

from typing import Any, Callable, Optional, Sequence

from absl import logging
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

from . import attention_layers, nn_layers

Initializer = Callable[[jnp.ndarray, Sequence[int], jnp.dtype], jnp.ndarray]


class AddPositionEmbs(nn.Module):
    """Adds learned positional embeddings to the inputs.

    Attributes:
      posemb_init: Positional embedding initializer.

    Returns:
      Output in shape `[bs, timesteps, in_dim]`.
    """
    posemb_init: Initializer = nn.initializers.normal(
        stddev=0.02)  # From BERT.

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        # Inputs.shape is (batch_size, seq_len, emb_dim).
        assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                                  ' but it is: %d' % inputs.ndim)
        pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
        pe = self.param('pos_embedding', self.posemb_init, pos_emb_shape,
                        inputs.dtype)
        return inputs + pe


class MAPHead(nn.Module):
    """Multihead Attention Pooling."""
    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    num_heads: int = 12
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        n, _, d = x.shape
        probe = self.param('probe', nn.initializers.xavier_uniform(), (1, 1, d),
                           x.dtype)
        probe = jnp.tile(probe, [n, 1, 1])

        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, kernel_init=nn.initializers.xavier_uniform()
        )(probe, x)

        y = nn.LayerNorm()(x)
        x = x + attention_layers.MlpBlock(
            mlp_dim=self.mlp_dim,
            dtype=self.dtype,
            dropout_rate=0.0)(y, deterministic=True)
        return x[:, 0]


class Encoder1DBlock(nn.Module):
    """Transformer encoder layer.

    Attributes:
      mlp_dim: Dimension of the mlp on top of attention block.
      num_heads: Number of self-attention heads.
      dtype: The dtype of the computation (default: float32).
      dropout_rate: Dropout rate.
      attention_dropout_rate: Dropout for attention heads.
      stochastic_depth: probability of dropping a layer linearly grows
        from 0 to the provided value.

    Returns:
      output after transformer encoder block.
    """
    mlp_dim: int
    num_heads: int
    dtype: Any = jnp.float32
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    stochastic_depth: float = 0.0

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        """Applies Encoder1DBlock module.

        Args:
          inputs: Input data.
          deterministic: Deterministic or not (to apply dropout).

        Returns:
          Output after transformer encoder block.
        """
        # Attention block.
        assert inputs.ndim == 3
        x = nn.LayerNorm(dtype=self.dtype)(inputs)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            broadcast_dropout=False,
            deterministic=deterministic,
            dropout_rate=self.attention_dropout_rate)(x, x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)
        x = nn_layers.StochasticDepth(
            rate=self.stochastic_depth)(x, deterministic)
        x = x + inputs

        # MLP block.
        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = attention_layers.MlpBlock(
            mlp_dim=self.mlp_dim,
            dtype=self.dtype,
            dropout_rate=self.dropout_rate,
            activation_fn=nn.gelu,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6))(
                y, deterministic=deterministic)
        y = nn_layers.StochasticDepth(
            rate=self.stochastic_depth)(y, deterministic)
        return y + x


class Encoder(nn.Module):
    """Transformer Encoder.

    Attributes:
      num_layers: Number of layers.
      mlp_dim: Dimension of the mlp on top of attention block.
      num_heads: The number of heads for multi-head self-attention.
      positional_embedding: The type of positional embeddings to add to the
        input tokens. Options are {learned_1d, sinusoidal_2d, none}.
      dropout_rate: Dropout rate.
      stochastic_depth: probability of dropping a layer linearly grows
        from 0 to the provided value. Our implementation of stochastic depth
        follows timm library, which does per-example layer dropping and uses
        independent dropping patterns for each skip-connection.
      dtype: Dtype of activations.
      has_cls_token: Whether or not the sequence is prepended with a CLS token.
    """
    num_layers: int
    mlp_dim: int
    num_heads: int
    positional_embedding: str = 'learned_1d'
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    stochastic_depth: float = 0.0
    dtype: Any = jnp.float32
    has_cls_token: bool = False

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, *, train: bool = False):
        """Applies Transformer model on the inputs.

        Args:
          inputs: Input tokens of shape [batch, num_tokens, channels].
          train: If in training mode, dropout and stochastic depth is applied.

        Returns:
          Encoded tokens.
        """

        assert inputs.ndim == 3  # Shape is `[batch, len, emb]`.
        dtype = jax.dtypes.canonicalize_dtype(self.dtype)

        # Add positional embeddings to tokens.
        if self.positional_embedding == 'learned_1d':
            x = AddPositionEmbs(
                posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
                name='posembed_input')(
                    inputs)
        elif self.positional_embedding == 'sinusoidal_1d':
            x = attention_layers.Add1DPositionEmbedding(
                posemb_init=None)(inputs)
        elif self.positional_embedding == 'sinusoidal_2d':
            batch, num_tokens, hidden_dim = inputs.shape
            if self.has_cls_token:
                num_tokens -= 1
            height = width = int(np.sqrt(num_tokens))
            if height * width != num_tokens:
                raise ValueError(
                    'Input is assumed to be square for sinusoidal init.')
            if self.has_cls_token:
                inputs_reshape = inputs[:, 1:].reshape(
                    [batch, height, width, hidden_dim]
                )
                x = attention_layers.AddFixedSinCosPositionEmbedding()(inputs_reshape)
                x = x.reshape([batch, num_tokens, hidden_dim])
                x = jnp.concatenate([inputs[:, :1], x], axis=1)
            else:
                inputs_reshape = inputs.reshape(
                    [batch, height, width, hidden_dim])
                x = attention_layers.AddFixedSinCosPositionEmbedding()(inputs_reshape)
                x = x.reshape([batch, num_tokens, hidden_dim])
        elif self.positional_embedding == 'none':
            x = inputs
        else:
            raise ValueError('Unknown positional embedding: '
                             f'{self.positional_embedding}')
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        # Input Encoder.
        for lyr in range(self.num_layers):
            x = Encoder1DBlock(
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                stochastic_depth=(lyr / max(self.num_layers - 1, 1))
                * self.stochastic_depth,
                name=f'encoderblock_{lyr}',
                dtype=dtype,
            )(x, deterministic=not train)
        encoded = nn.LayerNorm(name='encoder_norm')(x)
        return encoded


class ViT(nn.Module):
    """Vision Transformer model.

      Attributes:
      num_classes: Number of output classes.
      mlp_dim: Dimension of the mlp on top of attention block.
      num_layers: Number of layers.
      num_heads: Number of self-attention heads.
      patch_size: specify the patch dimension for tokenization
      hidden_size: Size of the hidden state of the output of model's stem.
      positional_embedding: The type of positional embeddings to add to the
        tokens at the beginning of the transformer encoder. Options are
        {learned_1d, sinusoidal_2d, none}.
      representation_size: Size of the representation layer in the model's head.
        if None, we skip the extra projection + tanh activation at the end.
      dropout_rate: Dropout rate.
      attention_dropout_rate: Dropout for attention heads.
      classifier: type of the classifier layer. Options are 'gap', 'gmp', 'gsp',
        'token', 'none'.
      dtype: JAX data type for activations.
    """

    num_classes: int
    mlp_dim: int
    num_layers: int
    num_heads: int
    # patch_size: tuple[int,int]
    hidden_size: int
    positional_embedding: str = 'learned_1d'
    representation_size: Optional[int] = None
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    stochastic_depth: float = 0.0
    classifier: str = 'gap'
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool, debug: bool = False):

        # fh, fw = self.patch_size
        # Extracting patches and then embedding is in fact a single convolution.
        # x = nn.Conv(
        #     self.hidden_size, (fh, fw),
        #     strides=(fh, fw),
        #     padding='VALID',
        #     name='embedding')(
        #         x)
        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])

        # If we want to add a class token, add it here.
        if self.classifier == 'token':
            cls = self.param('cls', nn.initializers.zeros, (1, 1, c), x.dtype)
            cls = jnp.tile(cls, [n, 1, 1])
            x = jnp.concatenate([cls, x], axis=1)

        x = Encoder(
            mlp_dim=self.mlp_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            positional_embedding=self.positional_embedding,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            stochastic_depth=self.stochastic_depth,
            dtype=self.dtype,
            has_cls_token=self.classifier == 'token',
            name='Transformer',
        )(x, train=train)

        if self.classifier in ('token', '0'):
            x = x[:, 0]
        elif self.classifier in ('gap', 'gmp', 'gsp'):
            fn = {'gap': jnp.mean, 'gmp': jnp.max,
                  'gsp': jnp.sum}[self.classifier]
            x = fn(x, axis=1)
        elif self.classifier == 'map':
            x = MAPHead(
                num_heads=self.num_heads, mlp_dim=self.mlp_dim, dtype=self.dtype)(x)
        elif self.classifier == 'none':
            pass
        else:
            raise ValueError(f'Unknown classifier {self.classifier}')

        if self.representation_size is not None:
            x = nn.Dense(self.representation_size, name='pre_logits')(x)
            x = nn.tanh(x)
        else:
            x = nn_layers.IdentityLayer(name='pre_logits')(x)

        if self.num_classes > 0:
            # If self.num_classes <= 0, we just return the backbone features.
            x = nn.Dense(
                self.num_classes,
                kernel_init=nn.initializers.zeros,
                name='output_projection')(
                    x)
        return x
