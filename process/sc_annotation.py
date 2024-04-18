#!/usr/bin/env python

from pathlib import Path
from functools import partial
import logging

import numpy as np
import xtrain
import jax
import optax
import tensorflow as tf
import orbax.checkpoint as ocp
import flax.linen as nn
import wandb
import tifffile
from tqdm import tqdm
from absl import flags, app
from ml_collections import config_flags
import pprint

from chioso.modules import MLP, gradient_reversal

print(jax.default_backend())

tf.config.set_visible_devices([], "GPU")

_CONFIG = config_flags.DEFINE_config_file("config")
_FLAGS = flags.FLAGS
flags.DEFINE_string("logpath", ".", "")

jnp = jax.numpy

def eval(pred_fn, ds_test, exp_gamma=None):
    config = _CONFIG.value
    label = config.label
    logits = []
    embeddings = []
    for gids, cnts in tqdm(ds_test.as_numpy_iterator()):
        if exp_gamma is not None:
            cnts = cnts * exp_gamma[gids]
        p, v = pred_fn(gids, cnts)

        if config.get("output_mask", None) is not None:
            p = np.where(config.output_mask, p, -1e8)

        logits.append(np.array(p))
        embeddings.append(np.array(v["intermediates"]["feature"][0]))

    logits = np.concatenate(logits)
    embeddings = np.concatenate(embeddings)

    cts = np.argmax(logits, axis=-1)
    cts = np.r_[0, cts+1]
    new_label = cts[label]

    return logits, embeddings, new_label.astype("uint8")

class PredictorGan(nn.Module):
    embedding: jax.Array
    dsc_n_layers: int = 4

    def setup(self):
        self.gamma = self.param(
            "gamma", lambda rng, shape: jnp.zeros(shape), (self.embedding.shape[0])
        )
        self.dsc = MLP(1, self.dsc_n_layers)

    def __call__(self, target_data, ref_data, *, training=False):
        gids_t, cnts_t = target_data
        cnts = cnts_t * jnp.exp(self.gamma[gids_t])
        cnts = cnts / (cnts.sum(axis=-1, keepdims=True) + 1e-8)
        x = self.embedding[gids_t]
        state = cnts @ x
        state = gradient_reversal(state)

        dsc_x = self.dsc(state, deterministic=not training)
        dsc_y = self.dsc(ref_data, deterministic=not training)

        dsc_loss_x = optax.sigmoid_binary_cross_entropy(
            dsc_x,
            jnp.ones_like(dsc_x),
        ).mean()

        dsc_loss_y = optax.sigmoid_binary_cross_entropy(
            dsc_y,
            jnp.zeros_like(dsc_y)
        ).mean()

        return dict(
            dsc_loss_x = dsc_loss_x,
            dsc_loss_y = dsc_loss_y,
        )

def main(_):
    logpath = Path(_FLAGS.logpath)  
    logpath.mkdir(parents=True, exist_ok=True)    

    config = _CONFIG.value
    pprint.pp(config, compact=True)

    run = wandb.init(project="mosta", group=config.name)
    # run.config.update(config.to_dict())

    target_ds = config.target_dataset
    checkpoint_path = Path(config.checkpoint_path)

    predictor = nn.vmap(
        config.predictor.model.type,
        variable_axes={'params': None, "intermediates":0}, 
        split_rngs={'params': False},
    )(** config.predictor.model.config)

    restored = ocp.StandardCheckpointer().restore(
        (checkpoint_path/"checkpoint").absolute()
    )

    pred_fn = jax.jit(partial(
        predictor.apply, 
        dict(params=restored["train_state"]["params"]),
        mutable="intermediates",
    ))

    # baseline
    padding = config.train.padding
    ds_test = target_ds.padded_batch(1024, padded_shapes=(([padding], [padding])))
    logits, features, label = eval(pred_fn, ds_test)
    np.savez(logpath/"baseline_predictions", features=features, logits=logits)
    tifffile.imwrite(logpath/"baseline_label.tif", label)

    # adversial training
    ref_ds = tf.data.Dataset.load(str(checkpoint_path/"ref_embedding.ds"))
    if config.get("ref_data_mask", None) is not None:
        ref_ds_mask = tf.data.Dataset.from_tensor_slices(config.ref_data_mask)
        ref_ds = (
            tf.data.Dataset.zip(ref_ds, ref_ds_mask)
            .filter(lambda x, y: y)
            .map(lambda x, y: x)
        )

    bs = config.train.batch_size
    padding = config.train.padding
    combined_ds = xtrain.TFDatasetAdapter(
        tf.data.Dataset.zip(target_ds.repeat(), ref_ds.repeat())
        .padded_batch(bs, padded_shapes=(([padding], [padding]), [predictor.dim_hidden],))
        .map(lambda x, y: dict(target_data=x, ref_data=y))
    )

    embedding = jax.numpy.asarray(restored["train_state"]["params"]["embed"]["Embed_0"]["embedding"])
    model = PredictorGan(
        embedding,
        config.get("dsc_n_layers", 4)
    )

    trainer = xtrain.Trainer(
        model = model,
        losses = ["dsc_loss_x", "dsc_loss_y"],
        optimizer = optax.adamw(1e-3, weight_decay=1e-2),
        strategy=xtrain.VMapped,
    )

    train_iter = trainer.train(combined_ds, training=True)

    for k in range(config.train.n_steps // config.train.val_interval):
        for _ in range(config.train.val_interval):
            next(train_iter)
        print(train_iter.loss_logs)
        wandb.log(train_iter.loss)

        gamma = train_iter.parameters["gamma"]
        run.log({"gamma": gamma})

        logits, features, label = eval(pred_fn, ds_test, exp_gamma = jnp.exp(gamma))        
        tifffile.imwrite(logpath/f"render-{k}.tif", label)

    np.savez(logpath/"final_predictions", features=features, logits=logits)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    wandb.login()
    app.run(main)
