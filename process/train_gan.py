#!/usr/bin/env python

import logging
import random

from functools import partial
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import tensorflow as tf
import wandb
import tifffile

from absl import flags, app
from tqdm import tqdm
from ml_collections import config_flags
from skimage.transform import rescale
from scipy.stats import pearsonr

from xtrain import Trainer, GeneratorAdapter, Adversal
from chioso.modules import CellAnnotator, MLP
from chioso.data import SGData2D, SGDataset2D

backend = jax.default_backend()
tf.config.set_visible_devices([], "GPU")

_CONFIG = config_flags.DEFINE_config_file("config")
_FLAGS = flags.FLAGS
flags.DEFINE_string("logpath", ".", "")

def train_data_gen(config):
    ps = config.dataset.patch_size
    binning = config.dataset.binning
    bucketsize = config.dataset.bucket_size

    ref_ds = tf.data.Dataset.load(str(
        Path(config.predictor.checkpoint) / config.predictor.dataset.outname
    )).repeat().batch(config.train.ref_batch_size)
    ref_ds_iter = ref_ds.as_numpy_iterator()

    ref_iter = ref_ds.as_numpy_iterator()
    train_files = list(Path(config.dataset.path).glob(config.dataset.train_files))

    while True:
        random.shuffle(train_files)
        for train_file in train_files:
            with h5py.File(train_file, "r") as f:
                sgdataset = SGDataset2D(f["X"], np.dtype("int32"))
                h, w = sgdataset.shape
                for y0 in range(0, h-ps, ps):
                    for x0 in range(0, w-ps, ps):
                        sgc = sgdataset[y0:y0+ps, x0:x0+ps]
                        if binning != 1:
                            sgc = sgc.binning([binning, binning])
                        sgc = sgc.pad_to_bucket_size(bucket_size=bucketsize)

                        ref_data = next(ref_iter)

                        yield (sgc, ref_data), None


def test_data_gen(config):
    ps = config.dataset.patch_size
    gs = config.dataset.grid_size
    binning = config.dataset.binning
    bucketsize = config.dataset.bucket_size

    h5file = Path(config.dataset.path) / (config.dataset.test_file)
    with h5py.File(h5file, "r") as f:
        sgdataset = SGDataset2D(f["X"], np.dtype("int32"))
        h, w = sgdataset.shape
        for y0 in range(0, h, gs):
            for x0 in range(0, w, gs):
                sgc = sgdataset[y0:y0+ps, x0:x0+ps]

                if sgc.shape[0] != ps or sgc.shape[1] != ps:
                    sgc = sgc.pad([[0, ps-sgc.shape[0]],[0, ps-sgc.shape[1]]])

                if binning != 1:
                    sgc = sgc.binning([binning, binning])

                sgc = sgc.pad_to_bucket_size(bucket_size=bucketsize)

                yield sgc, y0 // binning, x0 // binning

def get_ds(config):
    train_ds = GeneratorAdapter(partial(train_data_gen, config))
    if "test_file" in config.dataset:
        test_ds = GeneratorAdapter(partial(test_data_gen, config))
    else:
        test_ds = None

    return train_ds, test_ds

def get_label(config):
    try:
        h5file = Path(config.dataset.path) / (config.dataset.test_file)
        with h5py.File(h5file, 'r') as data:
            label = data["segmentation"][...]

        label = rescale((label>0).astype(float), 1/config.dataset.binning)

        return label
    except:
        return None 
  

def loss_fn(batch, prediction):
    loss_main = prediction["dsc_loss_main"]
    loss_ref = prediction["dsc_loss_ref"]

    # loss_main = jnp.where(
    #     loss_main > jnp.median(loss_main),
    #     loss_main,
    #     0,
    # )
    
    return loss_main.mean() + loss_ref.mean()


def get_model(config):
    params = ocp.StandardCheckpointer().restore(
        (Path(config.predictor.checkpoint) / "checkpoint").absolute(),
    )["train_state"]["params"]

    predictor = config.predictor.model.type(**config.predictor.model.config)

    inner_model = CellAnnotator(predictor, **config.model)
    model = Adversal(
        inner_model, 
        MLP(1, 4, deterministic=True), 
        collection_name="adversal", 
        loss_reduction_fn=None,
    )

    return model, params

def run(config, logpath):
    logpath.mkdir(parents=True, exist_ok=True)

    logging.info(f"Logging to {logpath.resolve()}")

    train_ds, test_ds = get_ds(config)

    gt_fg = get_label(config)

    model,params = get_model(config)
    
    trainer = Trainer(
        model = model,
        optimizer = optax.adamw(config.train.lr, weight_decay=config.train.weight_decay),
        # losses = ("dsc_loss_main", "dsc_loss_ref"),
        losses = loss_fn,
        seed=config.train.seed,
        mutable="adversal",
    )

    train_it = trainer.train(train_ds, training=True)
    train_it.parameters["main_module"]["predictor"] = params
    train_it.freeze("main_module/predictor")

    def pred_fn(test_ds):
        model_apply = jax.jit(partial(model.apply, training=False, mutable="adversal"))

        preds = []
        fake_ref_data = np.zeros([1, 256])
        for sg, y0, x0 in test_ds:
            pred, _ = model_apply(dict(params=train_it.parameters), sg, fake_ref_data)
            preds.append(dict(
                ct = np.argmax(pred["output"], axis=-1),
                dsc_loss = np.array(pred["dsc_loss_main"]),
                y0 = y0,
                x0 = x0
            ))

        y_max, x_max = y0, x0
        binning = config.dataset.binning
        bs = (config.dataset.patch_size - config.dataset.grid_size) // binning // 2
        ps_y =  ps_x = config.dataset.patch_size // binning

        full_img = np.zeros([y_max + ps_y, x_max + ps_x], dtype="uint8")
        loss_img = np.zeros([y_max + ps_y, x_max + ps_x], dtype="float32")

        for pred in preds:
            y0, x0 = pred["y0"], pred["x0"]
            full_img[y0+bs:y0+ps_y-bs, x0+bs:x0+ps_x-bs] = pred["ct"].reshape(ps_y, ps_x)[bs:-bs,bs:-bs]
            loss_img[y0+bs:y0+ps_y-bs, x0+bs:x0+ps_x-bs] = pred["dsc_loss"].reshape(ps_y, ps_x)[bs:-bs,bs:-bs]

        return full_img, loss_img

    # full_img, loss_img = pred_fn(test_ds)
    # tifffile.imwrite(logpath/"ct-0.tiff", full_img)
    # tifffile.imwrite(logpath/"loss-0.tiff", loss_img)

    for steps in tqdm(range(config.train.train_steps)):
        next(train_it)

        if (steps + 1) % config.train.validation_interval == 0:
            print(train_it.loss_logs)
            # print(train_it.variables)

            train_it.reset_loss_logs()

            # eval
            cp_step = (steps + 1) // config.train.validation_interval
            if test_ds is not None:
                full_img, loss_img = pred_fn(test_ds)
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

    print(config)

    wandb.init(project="mosta", group=config.name)
    wandb.config.update(config.to_dict())

    logpath = Path(_FLAGS.logpath)

    seed = config.train.get("seed", 42)
    random.seed(seed)

    run(config, logpath)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    wandb.login()

    app.run(main)
