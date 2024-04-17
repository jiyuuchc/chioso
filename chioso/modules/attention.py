import math

import flax.linen as nn
import jax

from chioso.modules import ConvNeXt, FPN, SGDummyStem

class CellAnnotator(nn.Module):
    embed: jax.Array
    encoder_depth: tuple[int] = (3,3,9)
    encoder_hidden_dim: int = 64
    roi: tuple[int, int] = (8, 8)
    n_decoder_layers: int = 4
    dropout: float = 0

    def att(self, x, weights):
        _, x0 = jax.lax.scan(
            lambda carry, s: (None, jax.lax.conv_general_dilated_local(
                s[None, :, :, None],
                weights[:,:,:,None],
                (1,1),
                "same",
                self.roi,
                dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            ).squeeze(0).squeeze(-1)), 
            None,
            x.transpose(2, 0, 1),
        )
        x0 = x0.transpose(1,2,0)
        return x0

    @nn.compact
    def __call__(self, sg, *, training=False):
        x = SGDummyStem(embed=self.embed)(sg)

        x = ConvNeXt(
            1, 
            depths=self.encoder_depth, 
            dims=(self.encoder_hidden_dim,) * 3,
            drop_path_rate = self.dropout,
        ) (x, training=training)

        x = FPN(math.prod(self.roi))(x)[0]

        # x = UNet((64,64,64),)(x)[0]
        
        weights = nn.sigmoid(x)

        x0 = self.att(x, weights)

        embed_dim = x0.shape[-1]
        for _ in range(self.n_decoder_layers):
            x0 = nn.Dense(embed_dim)(x0)
            x0 = nn.relu(x0)
            x0 = nn.LayerNorm()(x0)
            
        out = nn.Dense(1)(x0)

        return x0, out
