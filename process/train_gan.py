#!/usr/bin/env python

import logging
import random

from functools import partial
from pathlib import Path

from absl import flags, app
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pprint
import tensorflow as tf
# import wandb
import tifffile

from ml_collections import config_flags

from xtrain import Trainer, TFDatasetAdapter, Adversal
from chioso.modules import CellAnnotator, MLP

backend = jax.default_backend()
tf.config.set_visible_devices([], "GPU")

_CONFIG = config_flags.DEFINE_config_file("config")
_FLAGS = flags.FLAGS
flags.DEFINE_string("logpath", ".", "")

def get_ds(config):
    main_ds = (
        tf.data.Dataset.load(str(config.dataset.train))
        .map(lambda x, y: x)
        .repeat()
    )
    # mask = tf.constant(config.dataset.mask)

    ref_ds = tf.data.Dataset.load(str(
        Path(config.predictor.checkpoint) / "ref_embedding.ds"    
    )).repeat().batch(config.train.ref_batch_size)

    combined_ds = tf.data.Dataset.zip(main_ds, ref_ds).map(lambda x,y: [(x,y)]).prefetch(1)

    logging.debug(combined_ds.element_spec)

    return combined_ds

def get_models(config):
    params = ocp.StandardCheckpointer().restore(
        (Path(config.predictor.checkpoint) / "checkpoint").absolute(),
    )["train_state"]["params"]

    predictor = config.predictor.model.type(**config.predictor.model.config)
    predictor = predictor.bind(dict(params=params))
    mlp, var_mlp = predictor.mlp.unbind()

    embedding = params["embed"]["Embed_0"]["embedding"]
    main_model = CellAnnotator(jnp.asarray(embedding), **config.model)

    def pred_fn(test_ds, variables):
        @jax.jit
        def _inner(data):
            pred = main_model.apply(variables, *data)
            logits = mlp.apply(
                var_mlp, pred, deterministic=True
            )
            ct = np.argmax(logits, axis=-1)
            score = jax.nn.softmax(logits, axis=-1).max(axis=-1)
            dsc_loss = pred["dsc_loss_main"]

            return dict(
                ct = ct,
                score = score,
                dsc_loss = dsc_loss
            )

        preds = []
        for sgdata, (y0, x0) in test_ds.as_numpy_iterator():
            pred = _inner(sgdata)
            pred = jax.tree_util.tree_map(lambda x: np.array(x), pred)
            pred.update(dict(
                y0 = y0,
                x0 = x0,
            ))
            preds.append(pred)

        preds = jax.tree_util.tree_map(
            lambda *x: tuple(x), *preds,
        )

        y_max, x_max = np.max(preds["y0"]), np.max(preds["x0"])
        ps_y, ps_x = config.model.get("shape2d", (128,128))
        full_img = np.zeros([y_max + ps_y, x_max + ps_x], dtype="uint8")
        score_img = np.zeros([y_max + ps_y, x_max + ps_x], dtype="float32")
        loss_img = np.zeros([y_max + ps_y, x_max + ps_x], dtype="float32")

        for y0, x0, ct, score, dsc_loss in zip(preds["y0"], preds["x0"], preds["ct"], preds["score"], pred["dsc_loss"]):
            full_img[y0:y0+ps_y, x0:x0+ps_x] = ct
            score_img[y0:y0+ps_y, x0:x0+ps_x] = score.reshape(ps_y, ps_x)
            loss_img[y0:y0+ps_y, x0:x0+ps_x] = dsc_loss.reshape(ps_y, ps_x)

        return full_img, score_img

    return main_model, pred_fn

def loss_fn(batch, prediction):
    loss_main = prediction["dsc_loss_main"]
    loss_ref = prediction["dsc_loss_ref"]

    # loss_main = jnp.where(
    #     loss_main > jnp.median(loss_main),
    #     loss_main,
    #     0,
    # )
    
    return loss_main.mean() + loss_ref.mean()
   
def run(config, logpath, seed):
    logpath.mkdir(parents=True, exist_ok=True)
    logging.info(f"Logging to {logpath.resolve()}")

    combined_ds = get_ds(config)

    main_model, pred_fn = get_models(config)

    trainer = Trainer(
        model = Adversal(main_model, MLP(1, 4), loss_reduction_fn=None),
        optimizer = optax.inject_hyperparams(optax.adamw)(config.train.lr, weight_decay=config.train.weight_decay),
        # losses = ("dsc_loss_main", "dsc_loss_ref"),
        losses = loss_fn,
        seed=seed,
    )

    train_it = trainer.train(TFDatasetAdapter(combined_ds), training=True)

    test_ds = tf.data.Dataset.load(str(config.dataset.train))

    for steps in range(config.train.train_steps):
        next(train_it)
        if (steps + 1) % config.train.validation_interval == 0:
            print(train_it.loss_logs)
            # print(train_it.variables)

            train_it.reset_loss_logs()

            # eval
            cp_step = (steps + 1) // config.train.validation_interval
            full_img, score_img = pred_fn(test_ds, dict(params=train_it.parameters["main_module"]))
            # full_img = (full_img + 1) * config.dataset.mask
            tifffile.imwrite(logpath/f"ct-{cp_step}.tiff", full_img)
            tifffile.imwrite(logpath/f"score-{cp_step}.tiff", score_img)

    ocp.StandardCheckpointer().save(
        (logpath/"checkpoint").absolute(),
        args=ocp.args.StandardSave(train_it),
    )

def main(_):
    config = _CONFIG.value
    pprint.pp(config)

    # wandb.init(project="mosta", group=config.name)
    # wandb.config.update(config.to_dict())

    logpath = Path(_FLAGS.logpath)
    seed = config.train.get("seed", 42)
    random.seed(seed)
    run(config, logpath, seed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # wandb.login()

    app.run(main)
