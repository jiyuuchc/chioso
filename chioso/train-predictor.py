#!/usr/bin/env python

import logging
import random
import sys

from functools import partial
from pathlib import Path

import anndata as ad
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import orbax.checkpoint as ocp
import tensorflow as tf
import wandb

from absl import flags, app
from ml_collections import config_flags
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from xtrain import Trainer, TFDatasetAdapter, VMapped

backend = jax.default_backend()

_CONFIG = config_flags.DEFINE_config_file("config")
_FLAGS = flags.FLAGS
flags.DEFINE_string("logpath", ".", "")

def ce_loss(batch, prediction, cls_weight=None):
    inputs, label = batch
    y_true = batch[1]
    loss = optax.softmax_cross_entropy_with_integer_labels(prediction, y_true)
    if cls_weight is not None:
        loss = loss * cls_weight[y_true]
    loss = loss.mean()
    return loss

class BalancedAcc:
    def __init__(self, has_aux=False):
        self.y_pred = []
        self.y_true = []
        self.states = []
        self.has_aux = has_aux

    def update(self, batch, prediction):
        if self.has_aux:
            prediction, aux = prediction
            self.states.append(aux["intermediates"]["state"][0])
        self.y_pred.append(jnp.argmax(prediction, axis=-1))
        self.y_true.append(batch[1])

    def compute(self):
        y_true = np.concatenate(self.y_true)
        y_pred = np.concatenate(self.y_pred)

        return balanced_accuracy_score(y_true, y_pred)

def get_ds(config):
    balanced = config.train.get("balanced_loss", True)
    padding = config.dataset.get("padding", 4096)
    val_frac = config.train.get("val_split", 0.2)
    dropout = config.dataset.get("dropout", 0)

    ds = tf.data.Dataset.load(config.dataset.path)

    if balanced:
        y_true = list(ds.map(lambda a,b,c: c).as_numpy_iterator())
        y_true = np.array(y_true)
        cls_weight = jnp.asarray(compute_class_weight("balanced", classes=np.unique(y_true), y=y_true))
    else:
        cls_weight = None

    def _dropout(x, y):
        gids, cnts = x
        if dropout == 0:
            return (gids, cnts), y
        else:
            p = tf.random.uniform([], 0, dropout)
            sgms = tf.repeat(tf.range(len(cnts)), cnts)
            s = tf.random.categorical(tf.math.log([[p, 1-p]]), len(sgms))
            cnts_new = tf.math.segment_sum(s[0], sgms)
            return (gids, cnts_new), y

    train_split = (
        ds
        .enumerate()
        .filter(lambda i, _: int(float(i+1) * val_frac) == int(float(i) * val_frac))
        .map(lambda _, x: x)
        .map(lambda gid, cnt, gt: ((gid, cnt), gt))
        .repeat()
        .map(_dropout)
        .shuffle(12800)
        .padded_batch(
            config.train.batchsize,
            padded_shapes=(
                ([padding], [padding]), [],
            ),
        )    
        .prefetch(1)
        
    )

    val_split = (
        ds
        .enumerate()
        .filter(lambda i, _: int(float(i+1) * val_frac) > int(float(i) * val_frac))
        .map(lambda _, x: x)
        .map(lambda gid, cnt, gt: ((gid, cnt), gt))
        .padded_batch(
            config.train.batchsize,
            padded_shapes=(
                ([padding], [padding]), [],
            ),
            drop_remainder=True,
        )    
        .prefetch(1)
    )

    return train_split, val_split, cls_weight

def main(_):
    config = _CONFIG.value
    print(config)

    wandb.init(project="mosta", group=config.name)
    wandb.config.update(config.to_dict())
    
    logpath = Path(_FLAGS.logpath)
    seed = config.train.get("seed", 42)
    random.seed(seed)
    tf.random.set_seed(seed)

    run(config, logpath, random.randint(0, sys.maxsize))

def run(config, logpath, seed):
    logpath.mkdir(parents=True, exist_ok=True)

    logging.info(f"Logging to {logpath.resolve()}")

    ds_train, ds_val, cls_weight = get_ds(config)

    model = config.model.type(**config.model.config)
    balanced = config.train.get("balanced_loss", True)
    freeze_embedding = config.train.get("freeze_embedding", False)

    sc = optax.piecewise_constant_schedule(
        config.train.lr, 
        {int(config.train.train_steps * 0.8): 0.1},
    )
    tx = optax.adamw(sc, weight_decay = config.train.weight_decay)

    trainer = Trainer(
        model = model,
        optimizer = tx,
        losses = partial(ce_loss, cls_weight=cls_weight),
        strategy= VMapped,
        seed = seed,
    )    

    train_it = trainer.train(
        TFDatasetAdapter(ds_train), 
        rng_cols=["dropout"], 
        training=True, 
    )

    if freeze_embedding:
        frozen = jax.tree_util.tree_map_with_path(
            lambda p, _: jax.tree_util.DictKey("Embed_0") in p, train_it.parameters,
        )
        train_it = trainer.train(
            TFDatasetAdapter(ds_train), 
            rng_cols=["dropout"], 
            frozen=frozen,
            training=True,            
        )

    embed = config.model.get("embed", None)
    if embed is not None:
        if not isinstance(embed, np.ndarray):
            embed = np.load(embed)
        if not isinstance(embed, np.ndarray):
            try:
                embed = embed["data"]
            except KeyError:
                embed = embed["arr_0"]
        
        train_it.parameters["Embed_0"]["embedding"] = jnp.asarray(embed)

    cp = ocp.StandardCheckpointer()
    for step in range(config.train.train_steps):
        next(train_it)
        if (step + 1) % config.train.validation_interval == 0:
            print(f"step: {step+1}")
            print(train_it.loss_logs)

            vals = trainer.compute_metrics(
                TFDatasetAdapter(ds_val),
                BalancedAcc(),
                dict(params=train_it.parameters),
            )
            print(vals)

            wandb.log(train_it.loss)
            wandb.log(vals)

            train_it.reset_loss_logs()
    cp.save(
        (logpath/"checkpoint").absolute(),
        args=ocp.args.StandardSave(train_it),
    )

    logging.info("Record reference cell features")
    embedding = tf.constant(train_it.parameters["embed"]["Embed_0"]["embedding"])
    # FIXME implement using nn.apply and JIT
    def _agg(indices, cnts, gt):
        cnts = tf.cast(cnts, tf.float32)
        if model.log_transform:
            cnts = tf.math.log1p(cnts)
        if model.normalize:
            cnts = cnts / tf.reduce_sum(cnts)
        x = cnts[None,:] @ tf.gather(embedding, indices)
        return x[0]

    ds_export = (
        tf.data.Dataset.load(config.dataset.path)
        .map(_agg)
    )
    ds_export.save(str(logpath/config.dataset.outname))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    wandb.login()

    app.run(main)
