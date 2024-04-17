#!/usr/bin/env python

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import logging
import random
import sys

from functools import partial
from pathlib import Path

from absl import flags, app
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import orbax.checkpoint as ocp
import tensorflow as tf
import wandb

from ml_collections import config_flags
from pprint import pprint
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

from xtrain import Trainer, TFDatasetAdapter, VMapped

backend = jax.default_backend()

_CONFIG = config_flags.DEFINE_config_file("config")
_FLAGS = flags.FLAGS
flags.DEFINE_string("logpath", ".", "")

def ce_loss(batch, prediction, balanced=True):
    inputs, label = batch
    cls_weight = jnp.array([1.44054720e+00, 4.07123607e+00, 3.87952382e+00, 1.69368467e+01,
        1.13068053e+02, 3.15893404e+01, 1.93404125e+00, 1.08074276e+01,
        4.48241132e-01, 1.86840357e-01, 9.31950185e-01, 9.20956502e-01,
        1.39022027e+00, 2.48050044e-01, 5.22962285e-01, 8.78469038e-01,
        4.73429617e+00, 6.09021829e+00, 1.50411630e+02, 1.31054098e+01,
        2.99723357e+01, 1.51945020e+01, 8.66994587e+00, 1.84481463e+00,
        7.54689175e-01, 6.07217320e-01, 6.39425415e+00, 8.11707478e-01,
        1.68958290e-01, 2.43608732e+01, 1.18289088e+01, 1.31579997e+01,
        3.30995471e-01, 3.10156406e+00, 1.01015820e+01, 1.45228697e+00,
        4.37348085e-01, 8.25937917e-01, 1.08899818e+00, 1.42069910e+01,
        5.17759913e-01, 8.20974845e+00, 2.74345175e+00, 7.64685991e+00,
        4.82386431e-01, 2.52894039e-01, 7.30934804e+00, 4.20381222e+01,
        1.28285349e+01, 1.62486300e+01, 2.04068554e+00, 7.95674237e-01,
        1.11347919e+00, 6.55794706e+02, 1.16275657e+01, 7.00037047e+00,
        3.65141818e+01, 8.76730890e+00, 3.55251737e+00, 4.54151458e+01,
        1.20293988e+00, 3.02767639e+00, 3.57233356e-01, 7.19862465e-01,
        2.27091456e-01, 3.85625489e-01, 3.03828092e-01, 2.47881277e+00])
    y_true = batch[1]
    loss = optax.softmax_cross_entropy_with_integer_labels(prediction, y_true)
    if balanced:
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
    padding = config.dataset.get("padding", 4096)
    ds_location = config.dataset.get(
        "path",
        "../tome.ds"
    )

    with open(config.dataset.lut) as f:
        lut_ = json.load(f)

    n_tome_genes = config.dataset.get("n_tome_genes", 49585)

    lut = np.zeros([n_tome_genes], dtype=int) - 1
    lut[list(lut_.values())] = np.arange(len(lut_))
    lut = tf.constant(lut, dtype=tf.int32)

    def pre_process(idx, cnt, gt):
        idx = tf.gather(lut, idx)
        return (idx, cnt), gt

    ds = (
        tf.data.Dataset.load(ds_location)
        .map(pre_process)
    )

    ds_train = (
        ds.enumerate()
        .filter(lambda n,x: n%5 != 0)
        .map(lambda n,x: x)
        .repeat()
        .shuffle(12800)
        .padded_batch(
            config.train.batchsize,
            padded_shapes=(
                ([padding], [padding]), [],
            ),
            padding_values=-1,
        )    
        .prefetch(1)
    ) 

    ds_val = (
        ds.enumerate()
        .filter(lambda n,x: n%5 == 0)
        .map(lambda n, x: x)
        .padded_batch(
            config.train.batchsize,
            padded_shapes=(
                ([padding], [padding]), [],
            ),
            padding_values=-1,
            drop_remainder=True,
        )    
        .prefetch(1)
    ) 

    return ds_train, ds_val

def main(_):
    config = _CONFIG.value
    pprint(config)

    wandb.init(project="mosta", group=config.name)
    wandb.config.update(config.to_dict())
    
    logpath = Path(_FLAGS.logpath)
    seed = config.train.get("seed", 42)
    random.seed(seed)

    for i in range(config.get("num_runs", 1)):
        run(config, logpath/str(i), random.randint(0, sys.maxsize))

def run(config, logpath, seed):
    logpath.mkdir(parents=True, exist_ok=True)

    logging.info(f"Logging to {logpath.resolve()}")

    ds_train, ds_val = get_ds(config)

    model = config.model.type(**config.model.config)
    balanced = config.train.get("balanced_loss", True)
    freeze_embedding = config.train.get("freeze_embedding", False)

    sc = optax.piecewise_constant_schedule(
        config.train.lr, 
        {int(config.train.train_steps * 0.9): 0.1},
    )
    tx = optax.adamw(sc, weight_decay = config.train.weight_decay)

    trainer = Trainer(
        model = model,
        optimizer = tx,
        losses = partial(ce_loss, balanced=balanced),
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

    cp = ocp.StandardCheckpointer()
    cp.save(
        (logpath/"checkpoint").absolute(), 
        args=ocp.args.StandardSave(train_it),
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    wandb.login()

    app.run(main)
