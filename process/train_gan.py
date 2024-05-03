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
import wandb
import tifffile

from ml_collections import config_flags
from skimage.transform import rescale
from scipy.stats import pearsonr

from xtrain import Trainer, GeneratorAdapter, Adversal
from chioso.modules import CellAnnotator, MLP
from chioso.data import SGData2D

backend = jax.default_backend()
tf.config.set_visible_devices([], "GPU")

_CONFIG = config_flags.DEFINE_config_file("config")
_FLAGS = flags.FLAGS
flags.DEFINE_string("logpath", ".", "")

def get_ds(config):
    ds_path = Path(config.dataset.path) / (config.dataset.name + ".ds")
    main_ds = (
        tf.data.Dataset.load(str(ds_path))
        .map(lambda x, y: x)
        .repeat()
    )

    ref_ds = tf.data.Dataset.load(str(
        Path(config.predictor.checkpoint) / config.dataset.ref_name
    )).repeat().batch(config.train.ref_batch_size)

    combined_ds = tf.data.Dataset.zip(main_ds, ref_ds).prefetch(1)

    logging.debug(combined_ds.element_spec)

    def _generator():
        for sg_data, ref_data in combined_ds.as_numpy_iterator():
            sg = SGData2D(*sg_data, shape=config.dataset.patch_shape, n_genes=config.dataset.n_genes)
            sg = sg.binning([config.dataset.binning, config.dataset.binning])
            sg = sg.pad_to_bucket_size(config.dataset.bucket_size)

            yield (sg, ref_data), None 

    return GeneratorAdapter(_generator)

def get_label(config):
    import h5py
    h5file = Path(config.dataset.path) / (config.dataset.name + ".h5")
    try:
        with h5py.File(h5file, 'r') as data:
            label = data["uns/dapi_segm"][...]

        label = rescale((label>0).astype(float), 1/config.dataset.binning)

        return label
    except:
        return None 
  
def get_models(config):
    params = ocp.StandardCheckpointer().restore(
        (Path(config.predictor.checkpoint) / "checkpoint").absolute(),
    )["train_state"]["params"]

    predictor = config.predictor.model.type(**config.predictor.model.config)
    predictor = predictor.bind(dict(params=params))
    mlp, var_mlp = predictor.mlp.unbind()

    embedding = params["embed"]["Embed_0"]["embedding"]
    inner_model = CellAnnotator(jnp.asarray(embedding), **config.model)
    model = Adversal(inner_model, MLP(1, 4, deterministic=True), loss_reduction_fn=None)

    def pred_fn(test_ds, variables):
        @jax.jit
        def _inner(data):
            pred = model.apply(variables, *data, training=False)
            logits = mlp.apply(
                var_mlp, pred["output"], deterministic=True
            )
            ct = np.argmax(logits, axis=-1)
            dsc_loss = pred["dsc_loss_main"]

            return dict(
                ct = ct,
                dsc_loss = dsc_loss
            )

        preds = []
        fake_ref_data = np.zeros([1, 256])
        binning = config.dataset.binning
        bs = config.dataset.border_size // binning

        for sg_data, (y0, x0) in test_ds:
            sg = SGData2D(*sg_data, shape=config.dataset.patch_shape, n_genes=config.dataset.n_genes)
            sg = sg.binning([binning, binning])
            sg = sg.pad_to_bucket_size(config.dataset.bucket_size)

            pred = _inner((sg, fake_ref_data))
            pred = jax.tree_util.tree_map(lambda x: np.array(x), pred)
            pred.update(dict(
                y0 = y0 // binning,
                x0 = x0 // binning,
            ))
            preds.append(pred)

        y_max, x_max = y0 // binning, x0 // binning
        ps_y, ps_x = config.dataset.patch_shape
        ps_y = ps_y // binning
        ps_x = ps_x // binning
        full_img = np.zeros([y_max + ps_y, x_max + ps_x], dtype="uint8")
        loss_img = np.zeros([y_max + ps_y, x_max + ps_x], dtype="float32")

        for pred in preds:
            y0, x0 = pred["y0"], pred["x0"]
            full_img[y0+bs:y0+ps_y-bs, x0+bs:x0+ps_x-bs] = pred["ct"].reshape(ps_y, ps_x)[bs:-bs,bs:-bs]
            loss_img[y0+bs:y0+ps_y-bs, x0+bs:x0+ps_x-bs] = pred["dsc_loss"].reshape(ps_y, ps_x)[bs:-bs,bs:-bs]

        return full_img, loss_img

    return model, pred_fn

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

    model, pred_fn = get_models(config)

    gt_fg = get_label(config)

    trainer = Trainer(
        model = model,
        optimizer = optax.adamw(config.train.lr, weight_decay=config.train.weight_decay),
        # losses = ("dsc_loss_main", "dsc_loss_ref"),
        losses = loss_fn,
        seed=seed,
    )

    train_it = trainer.train(combined_ds, training=True)

    test_ds = tf.data.Dataset.load(str(Path(config.dataset.path) / (config.dataset.name + ".ds")))

    # full_img, loss_img = pred_fn(test_ds, dict(params=train_it.parameters))
    # tifffile.imwrite(logpath/"ct-0.tiff", full_img)
    # tifffile.imwrite(logpath/"loss-0.tiff", loss_img)

    for steps in range(config.train.train_steps):
        next(train_it)
        if (steps + 1) % config.train.validation_interval == 0:
            print(train_it.loss_logs)
            # print(train_it.variables)

            train_it.reset_loss_logs()

            # eval
            cp_step = (steps + 1) // config.train.validation_interval
            full_img, loss_img = pred_fn(test_ds, dict(params=train_it.parameters))
            tifffile.imwrite(logpath/f"ct-{cp_step}.tiff", full_img)
            tifffile.imwrite(logpath/f"loss-{cp_step}.tiff", loss_img)

            if gt_fg is not None:
                h,w = gt_fg.shape
                pred_fg = loss_img[:h, :w]
                corr = pearsonr(pred_fg.reshape(-1), gt_fg.reshape(-1))
                print(f"corr = {corr}")
                wandb.log(dict(corr=corr))

    ocp.StandardCheckpointer().save(
        (logpath/f"cp_{cp_step}").absolute(),
        args=ocp.args.StandardSave(train_it),
    )

def main(_):
    config = _CONFIG.value
    pprint.pp(config)

    wandb.init(project="mosta", group=config.name)
    wandb.config.update(config.to_dict())

    logpath = Path(_FLAGS.logpath)

    seed = config.train.get("seed", 42)
    random.seed(seed)

    run(config, logpath, seed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    wandb.login()

    app.run(main)
