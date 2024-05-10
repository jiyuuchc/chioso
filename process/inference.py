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
import wandb
import tifffile
import pydensecrf.densecrf as dcrf

from absl import flags, app
from tqdm import tqdm
from ml_collections import config_flags

from xtrain import Trainer, GeneratorAdapter, Adversal
from chioso.modules import CellAnnotator, MLP
from chioso.data import SGData2D, SGDataset2D

backend = jax.default_backend()

_CONFIG = config_flags.DEFINE_config_file("config")
_FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint", None, "")
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


def inference(config):
    logpath = Path(_FLAGS.logpath)
    logpath.mkdir(parents=True, exist_ok=True)

    ps = config.dataset.patch_size
    gs = config.dataset.grid_size
    binning = config.dataset.binning
    bs = (config.dataset.patch_size - config.dataset.grid_size) // binning // 2
    n_cls = config.predictor.model.config.dim_out
    hidden_dim = config.predictor.model.config.get("dim_hidden", 256)

    with open(Path(_FLAGS.checkpoint)/"model_def.pkl", "rb") as f:
        model = pickle.load(f)

    params = ocp.StandardCheckpointer().restore(
        Path(_FLAGS.checkpoint/"model").absolute(),
    )["train_state"]["params"]

    def _method(mdl, sgc, y0, x0):
        pred = mdl(sgc, jnp.zeros([1, hidden_dim]), training=False)
        return pred, y0, x0
    
    apply_fn=partial(
        model.apply,
        mutable="adversal",
        method=_method,
    )

    def _gen(sgdataset):
        h, w = sgdataset.shape
        for y0 in range(0, h, gs):
            for x0 in range(0, w, gs):
                sgc = format_patch(config, sgdataset[y0:y0+ps, x0:x0+ps])
                yield sgc, y0//binning, x0//binning

    def _predict_label(pred):
        logits = np.concatenate([
            # - jnp.log(jnp.exp(pred["dsc_loss_main"])-1+1e-8),
            - pred["dsc_loss_main"],
            jax.nn.log_softmax(pred['output']),
        ], axis=-1)
        u = -logits.transpose(2,0,1).reshape(n_cls + 1, -1)
        d = dcrf.DenseCRF2D(ps//binning, ps//binning, n_cls + 1)
        d.setUnaryEnergy(np.ascontiguousarray(u))
        d.addPairwiseGaussian(sxy=config.inference.sxy, compat=config.inference.compat)
        q = np.argmax(d.inference(config.inference.iters), axis=0).reshape(ps//binning, ps//binning)

        return q 

    for input_path in Path(config.dataset.path).glob(config.dataset.train_files):
        with h5py.File(input_path, "r") as src_data:
            sgdataset = SGDataset2D(src_data["X"], np.dtype("int32"))
            h, w = sgdataset.shape

            imgh = ((h - 1) // gs * gs + ps) // binning
            imgw = ((w - 1) // gs * gs + ps) // binning
            label = np.zeros([imgh, imgw], dtype="uint8")

            for inputs in GeneratorAdapter(partial(_gen, sgdataset), prefetch=1):
                (pred, x0, y0), _ = xtrain.JIT.predict(apply_fn, dict(params=params), inputs)
                patch_label = _predict_label(pred)

                y0, x0 = y0 + bs, x0 + bs
                y1, x1 = y0 + gs//binning, x0 + gs//binning

                label[y0 : y1, x0 : x1] = patch_label[bs:-bs,bs:-bs]

            tifffile.imwrite(f"{input_path.stem}_prediction.tif", label)

        logging.info(f"Saved perdictions for {input_path.name}")


def main(_):
    config = _CONFIG.value

    print(config)

    wandb.init(project="mosta", group="inference")
    wandb.config.update(config.to_dict())

    inference(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    wandb.login()
    app.run(main)
    