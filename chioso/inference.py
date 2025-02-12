#!/usr/bin/env python

import logging
import pickle

from functools import partial
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pydensecrf.densecrf as dcrf
import tifffile

from absl import flags, app
from ml_collections import config_flags
from xtrain import GeneratorAdapter, JIT
from chioso.data import SGDataset2D
from chioso.utils import predict, crf_scan

_CONFIG = config_flags.DEFINE_config_file("config")
_FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint", None, "")
flags.DEFINE_string("logpath", ".", "")

def inference(config):
    logpath = Path(_FLAGS.logpath)
    logpath.mkdir(parents=True, exist_ok=True)

    with open(Path(_FLAGS.checkpoint)/"model_def.pickle", "rb") as f:
        model = pickle.load(f)

    params = ocp.StandardCheckpointer().restore(
        (Path(_FLAGS.checkpoint)/"model").absolute(),
    )["train_state"]["params"]

    for input_path in Path(config.dataset.path).glob(config.dataset.train_files):
        with h5py.File(input_path, "r") as src_data:
            sgdataset = SGDataset2D(src_data["X"], np.dtype("int32"))
            label, scores, cts, offsets = predict(
                model, params, sgdataset,
                ps = config.dataset.patch_size, 
                gs = config.dataset.grid_size, 
                binning = config.dataset.binning,
                bucketsize = config.dataset.bucket_size,
                sxy = config.inference.sxy,
                compat = config.inference.compat,
                n_iter = config.inference.iters,
            )

            cts = np.log(cts + .5)
            # scores = np.clip(scores - 0.5, 0, np.inf)
            # mask = cts + scores

            # src_data["label"] = label
            # src_data["mask"] = mask

        name = input_path.stem
        tifffile.imwrite(logpath/f"{name}_score.tif", scores.astype("float32"))
        tifffile.imwrite(logpath/f"{name}_cts.tif", cts.astype("float32"))
        tifffile.imwrite(logpath/f"{name}_label.tif", label.astype("uint16"))
        tifffile.imwrite(logpath/f"{name}_offsets.tif", offsets.astype("float32"))

        logging.info(f"Saved perdictions for {input_path.name}")


def main(_):
    config = _CONFIG.value

    print(config)

    inference(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info(f"JAX backend is { jax.default_backend()}")  

    app.run(main)
    
