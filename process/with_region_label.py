#!/usr/bin/env python

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import logging

import h5py
import jax
import tifffile
import numpy as np
import orbax.checkpoint as ocp

from absl import flags, app
from scipy.sparse import csr_array
from pprint import pprint
from tqdm import tqdm
from pathlib import Path

from chioso.modules import MLP
from chioso.data import SGData2D

backend = jax.default_backend()

_FLAGS = flags.FLAGS
flags.DEFINE_string("data", None, "")
flags.DEFINE_string("checkpoint", None, "")
flags.DEFINE_string("logpath", ".", "")

def main(_):
    logpath = Path(_FLAGS.logpath)  
    logpath.mkdir(parents=True, exist_ok=True)    

    checkpoint_path = Path(_FLAGS.checkpoint)
    data_path = Path(_FLAGS.data)

    params = ocp.StandardCheckpointer().restore(
        checkpoint_path.absolute(),
    )["train_state"]["params"]
    embedding = params["embed"]["Embed_0"]["embedding"]
    params_mlp = params["mlp"]

    logging.info("Load SG data...")
    with h5py.File(data_path, mode="r", ) as data:
        cnts = data["uns/sc/data"][...]
        gids = data["uns/sc/indices"][...]
        indptr = data["uns/sc/indptr"][...]
        genes = list(data["var/index"])        
        label = np.asarray(data["uns/dapi_segm"]).astype(int)

        assert label.max() == data["uns/sc"].attrs["shape"][0]
        assert len(genes) == data["uns/sc"].attrs["shape"][1]

        csr = csr_array((cnts, gids, indptr), data["uns/sc"].attrs["shape"])

    logging.info("Compute cell embedding...")
    embed = []
    chunk = 1024
    for k in tqdm(range(0, csr.shape[0], chunk)):
        k1 = min(k + chunk, csr.shape[0])
        cnts = jax.numpy.asarray(csr[k:k1].todense())
        cnts = cnts / cnts.sum(axis=-1, keepdims=True)
        embed.append( cnts @ embedding)
    embed = jax.numpy.concatenate(embed, axis=0)

    logging.info("Predicting ...")
    predictions = MLP(68, 6, deterministic=True).apply(dict(params=params_mlp), embed)
    cell_type = jax.numpy.argmax(predictions, axis=-1)
    scores = jax.nn.softmax(predictions, axis=-1)

    ct_a = (np.r_[0, np.asarray(cell_type) + 1]).astype("uint8")
    ct_label = ct_a[label]

    np.savez(logpath/"predictions", embed=embed, cell_type=cell_type, scores=scores)
    tifffile.imwrite("lable.tif", ct_label)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    app.run(main)
