#!/usr/bin/env python

import logging
import random
import pickle

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
import pydensecrf.densecrf as dcrf

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

def format_patch(config, sgc):
    ps = config.dataset.patch_size
    binning = config.dataset.binning
    bucketsize = config.dataset.bucket_size

    if sgc.shape[0] != ps or sgc.shape[1] != ps:
        sgc = sgc.pad([[0, ps-sgc.shape[0]],[0, ps-sgc.shape[1]]])

    if binning != 1:
        sgc = sgc.binning([binning, binning])

    sgc = sgc.pad_to_bucket_size(bucket_size=bucketsize)

    return sgc

def get_ds(config):
    def train_data_gen():
        ps = config.dataset.patch_size

        ref_ds = tf.data.Dataset.load(str(
            Path(config.predictor.checkpoint) / config.predictor.dataset.outname
        )).repeat().batch(config.train.ref_batch_size)
        ref_ds_iter = ref_ds.as_numpy_iterator()

        ref_iter = ref_ds.as_numpy_iterator()
        train_files = list(Path(config.dataset.path).glob(config.dataset.train_files))

        logging.info(f"Found {len(train_files)} trainng datasets:")
        logging.info([fn.name for fn in train_files])

        while True:
            random.shuffle(train_files)

            for train_file in train_files:
                f = h5py.File(train_file, "r")
                sgdataset = SGDataset2D(f["X"], np.dtype("int32"))
                h, w = sgdataset.shape

                for y0 in range(0, h-ps, ps):
                    for x0 in range(0, w-ps, ps):
                        sgc = format_patch(config, sgdataset[y0:y0+ps, x0:x0+ps])
                        
                        ref_data = next(ref_iter)

                        yield (sgc, ref_data), None

                f.close()

    return GeneratorAdapter(train_data_gen)


def checkpoint(config, train_it, dst):
    dst.mkdir(parents=True, exist_ok=True)

    ps = config.dataset.patch_size
    gs = config.dataset.grid_size
    binning = config.dataset.binning
    bs = (config.dataset.patch_size - config.dataset.grid_size) // binning // 2
    n_cls = config.predictor.model.config.dim_out
    hidden_dim = config.predictor.model.config.get("dim_hidden", 256)

    trainer = train_it.ctx

    with open(dst/"model_def.pickle", "wb") as f:
        pickle.dump(trainer.model, f)

    ocp.StandardCheckpointer().save(
        (dst/"model").absolute(),
        args=ocp.args.StandardSave(train_it),
    )

    def _model_apply(mdl, sgc, y0, x0):
        pred = mdl(sgc, jnp.zeros([1, hidden_dim]), training=False)
        return pred, y0, x0

    def _gen(sgdataset):
        h, w = sgdataset.shape
        for y0 in range(0, h, gs):
            for x0 in range(0, w, gs):
                sgc = format_patch(config, sgdataset[y0:y0+ps, x0:x0+ps])
                yield sgc, y0//binning, x0//binning

    cp_file = h5py.File(dst / "predicted_logits.h5", "w")

    for input_path in Path(config.dataset.path).glob(config.dataset.train_files):
        cp_grp = cp_file.create_group(input_path.name)

        with h5py.File(input_path, "r") as src_data:
            sgdataset = SGDataset2D(src_data["X"], np.dtype("int32"))
            h, w = sgdataset.shape

            imgh = ((h - 1) // gs * gs + ps) // binning
            imgw = ((w - 1) // gs * gs + ps) // binning

            cls_predict = cp_grp.create_dataset("cls_predict", [imgh, imgw], dtype="int32")
            fg_logits = cp_grp.create_dataset("fg_logits", [imgh, imgw], dtype="float32")

            predict_iter = trainer.predict(
                GeneratorAdapter(partial(_gen, sgdataset), prefetch=1),
                dict(params=train_it.parameters), 
                method=_model_apply,
            )

            for (pred, y0, x0), _ in predict_iter:
                y0, x0 = y0 + bs, x0 + bs
                y1, x1 = y0 + gs//binning, x0 + gs//binning

                cls_predict[y0 : y1, x0 : x1] = np.argmax(pred["output"][bs:-bs, bs:-bs], axis=-1)
                fg_logits[y0 : y1, x0 : x1] = pred["dsc_loss_main"][bs:-bs, bs:-bs, 0]

        logging.info(f"Saved perdictions for {input_path.name}")

        # some input data have a segmentaion label (e.g. DAPI)
        # report the accuracy of the prediction in this case
        if "segmentation" in src_data:
            gt_seg = src_data["segmentation"][...]
            gt_seg = rescale((label>0).astype(float), 1/binning)
            
            minh = min(gt_seg.shape[0], fg_logits.shape[0])
            minw = min(gt_seg.shape[1], fg_logits.shape[1])
            corr = pearsonr(fg_logits[:minh, :minw].reshape(-1), gt_seg[:minh, :minw].reshape(-1))

            print(f"{input_path.stem} : corr = {corr}")
            wandb.log(dict(corr = {input_path.stem: corr}))

        src_data.close()

    cp_file.close()                    


def get_model(config):
    params = ocp.StandardCheckpointer().restore(
        (Path(config.predictor.checkpoint) / "checkpoint").absolute(),
    )["train_state"]["params"]

    predictor = config.predictor.model.type(**config.predictor.model.config)

    inner_model = CellAnnotator(predictor, **config.model)
    model = Adversal(
        inner_model, 
        MLP(1, config.discriminator.n_layers, deterministic=True), 
        collection_name="adversal", 
        loss_reduction_fn=None,
    )

    return model, params


def run(config, logpath):
    logpath.mkdir(parents=True, exist_ok=True)

    logging.info(f"Logging to {logpath.resolve()}")

    train_ds = get_ds(config)

    model,params = get_model(config)
    
    def loss_fn(batch, prediction):
        loss_main = prediction["dsc_loss_main"]
        loss_ref = prediction["dsc_loss_ref"]

        # loss_main = jnp.where(
        #     loss_main > jnp.median(loss_main),
        #     loss_main,
        #     0,
        # )
    
        return loss_main.mean() + loss_ref.mean()

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

    # checkpoint(config, train_it, logpath / "checkpoint_0")

    for steps in tqdm(range(config.train.train_steps)):
        next(train_it)

        if (steps + 1) % config.train.checkpoint_interval == 0:
            print(train_it.loss_logs)
            # print(train_it.variables)

            train_it.reset_loss_logs()

            # eval
            cp_step = (steps + 1) // config.train.checkpoint_interval
            cp_dir = logpath / f"checkpoint_{cp_step}"

            checkpoint(config, train_it, cp_dir)


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
