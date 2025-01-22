#!/usr/bin/env python

from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
import logging
import shutil

from gzip import BadGzipFile
from functools import partial

import h5py
import numpy as np

from absl import flags, app
from scipy.sparse import csr_array
from pathlib import Path

from chioso.data import SGData2D, SGDataset2D

_FLAGS = flags.FLAGS
flags.DEFINE_string("data", None, "")
flags.DEFINE_string("genes", None, "")
flags.DEFINE_string("outdir", ".", "")
flags.DEFINE_multi_string("comments", ["#"], "")
flags.DEFINE_integer("skiprows", 0, "")
flags.DEFINE_string("label", None, "")

def load_gem():
    gemfile = Path(_FLAGS.data)
    genefile = Path(_FLAGS.genes)
    logging.info(f"Read gene names from {genefile}")

    with open(genefile) as f:
        genes = json.load(f)
        if isinstance(genes, dict):
            genes = list(genes.keys())
    lut = dict(zip(genes, range(len(genes))))
    
    def conv_f(x):
        try:
            x = x.decode()
        except:
            pass
        if x in lut:
            return lut[x]
        else:
            return -1

    kwargs = dict(
        skiprows = _FLAGS.skiprows,
        converters = {0: conv_f},
        unpack = True,
        dtype = "int32",
        comments = _FLAGS.comments,
    )

    logging.info("Reading SG data ... ")    
    try:
        gids, x, y, c = np.loadtxt(gemfile, **kwargs)
    except BadGzipFile:
        gemfile = gemfile.rename(gemfile.name.replace(".gz", ""))
        gids, x, y, c = np.loadtxt(gemfile, **kwargs)

    c = np.where(gids>=0, c, 0)
    gids = np.where(gids >= 0, gids, 0)

    x -= x.min()
    y -= y.min()
    h, w = y.max() + 1, x.max() + 1

    rows = np.asarray(y * w + x).astype("int64")
    cols = np.asarray(gids).astype("int64")

    sa = csr_array((c, (rows, cols)), shape=(h * w, len(genes)))

    sg = SGData2D.from_csr(sa, (h,w))

    return sg, genes

def write_h5(sg, genes, label=None):
    logging.info("Writing h5 file ... ")

    gemfile = Path(_FLAGS.data)
    outdir = Path(_FLAGS.outdir)
    outfile = outdir / (gemfile.stem + ".h5")
    if outfile.exists():
        logging.warn(f"{outfile} already exists. The new data will overwrite existing data.")
        shutil.rmtree(outfile)

    with h5py.File(outfile, mode="w", ) as data:
        SGDataset2D.create_from_sgdata(data, "X", sg)
        data["genes"] = genes
        if label is not None:
            data["segmentation"] = label

def main(_):
    outdir = Path(_FLAGS.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if _FLAGS.label is not None:
        import tifffile
        label = tifffile.imread(_FLAGS.label)
    else:
        label = None

    sg, genes = load_gem()

    write_h5(sg, genes, label)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    app.run(main)

