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
import tensorflow as tf

from absl import flags, app
from scipy.sparse import csr_array
from pathlib import Path

from chioso.data import SGData2D

_FLAGS = flags.FLAGS
flags.DEFINE_string("data", None, "")
flags.DEFINE_string("genes", None, "")
flags.DEFINE_integer("binning", 1, "")
flags.DEFINE_string("outdir", ".", "")
flags.DEFINE_multi_string("comments", ["#"], "")
flags.DEFINE_integer("skiprows", 0, "")
flags.DEFINE_bool("writeh5", True, "")
flags.DEFINE_integer("patchsize", 1024, "")
flags.DEFINE_integer("gridsize", 768, "")
flags.DEFINE_string("label", None, "")

def load_gem():
    gemfile = Path(_FLAGS.data)

    if gemfile.suffix == ".h5":
        logging.info("Assuming H5 inputs. Skipping gid lookup")
        sg = SGData2D.from_h5ad(gemfile)
        return sg        

    genefile = Path(_FLAGS.genes)
    logging.info("Read gene names from {genefile}")

    with open(genefile) as f:
        genes = json.load(f)
    lut = dict(zip(genes, range(len(genes))))
    
    def conv_f(x):
        x = x.decode()
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
    
    try:
        gids, x, y, c = np.loadtxt(gemfile, **kwargs)
    except BadGzipFile:
        gemfile = gemfile.rename(gemfile.name.replace(".gz", ""))
        gids, x, y, c = np.loadtxt(gemfile, **kwargs)

    c = np.where(gids>=0, c, 0)
    gids = np.where(gids >= 0, gids, 0)

    # x = x // _FLAGS.binning
    # y = y // _FLAGS.binning
    x -= x.min()
    y -= y.min()
    h, w = y.max() + 1, x.max() + 1

    rows = np.asarray(y * w + x).astype("int64")
    cols = np.asarray(gids).astype("int64")

    sa = csr_array((c, (rows, cols)), shape=(h * w, len(genes)))

    sg = SGData2D.from_csr(sa, (h,w))

    return sg

def write_h5(sg, label=None):
    gemfile = Path(_FLAGS.data)
    outdir = Path(_FLAGS.outdir)
    outfile = outdir / (gemfile.stem + ".h5")
    if outfile.exists():
        shutil.rmtree(outfile)

    with h5py.File(outfile, mode="w", ) as data:
        data.create_group("X")
        data["X/data"] = sg.data
        data["X/indices"] = sg.indices
        data["X/indptr"] = sg.indptr
        data["X"].attrs["shape"] = (sg.shape[0] * sg.shape[1], sg.n_genes)
        data["X"].attrs["2D_dimension"] = sg.shape

def write_ds(sg):
    gemfile = Path(_FLAGS.data)
    outdir = Path(_FLAGS.outdir)
    outfile = outdir / (gemfile.stem + ".ds")
    ps, gs = _FLAGS.patchsize, _FLAGS.gridsize
    # bucketsize = _FLAGS.bucketsize
    binning = _FLAGS.binning

    if outfile.exists():
        shutil.rmtree(outfile)

    def mosta_ds_gen():
        h, w = sg.shape
        h_pad = (h-ps-1) // gs * gs + gs + ps 
        w_pad = (w-ps-1) // gs * gs + gs + ps
        sg_pad = sg.pad(((0, h_pad-h), (0, w_pad-w)))
        for y0 in range(0, h_pad, gs):
            for x0 in range(0, w_pad, gs):
                sgc = sg_pad[y0:y0+ps, x0:x0+ps]
                if binning != 1:
                    sgc = sgc.binning([binning, binning])
                # sgc = sgc.pad_to_bucket_size(bucket_size=bucketsize)

                yield ((sgc.data, sgc.indices, sgc.indptr), (y0 // binning, x0 // binning))


    def make_ds():
        return tf.data.Dataset.from_generator(
            mosta_ds_gen, 
            output_signature=(
                (
                    tf.TensorSpec([None],tf.int32),
                    tf.TensorSpec([None],tf.int32),
                    tf.TensorSpec([ps*ps//binning//binning+1],tf.int32)
                ) ,
                (
                    tf.TensorSpec([],tf.int32),
                    tf.TensorSpec([],tf.int32),
                ),
            ),
        )
    make_ds().save(str(outfile))
    
    
def main(_):
    outdir = Path(_FLAGS.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if _FLAGS.writeh5 and _FLAGS.label is not None:
        import tifffile
        label = tifffile.imread(_FLAGS.label)
    else:
        label = None

    logging.info("Reading sg data ... ")
    sg = load_gem()

    if _FLAGS.writeh5 and Path(_FLAGS.data).suffix != ".h5":
        logging.info("Writing h5 file ... ")
        write_h5(sg, label)

    logging.info("write tfrecords ...")
    write_ds(sg)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    app.run(main)

