#!/usr/bin/env python

from __future__ import annotations

import json
import logging
import shutil

from gzip import BadGzipFile
from functools import partial

import h5py
import numpy as np
import tensorflow as tf
import pandas as pd

from absl import flags, app
from scipy.sparse import csr_array
from pathlib import Path

from chioso.data import SGData2D

_FLAGS = flags.FLAGS
flags.DEFINE_string("data", None, "")
flags.DEFINE_string("genes", None, "")
flags.DEFINE_string("label", "", "")
flags.DEFINE_string("outdir", ".", "")

def process_loom():
    src = Path(_FLAGS.data)
    label = pd.read_csv(_FLAGS.label)
    outdir = Path(_FLAGS.outdir)
    outfile = outdir/(src.stem + ".ds")

    if outfile.exists():
        shutil.rmtree(outfile)

    all_cell_types_dict = {}

    def gen():
        import loompy

        with loompy.connect(src) as data:
            n_rows, n_cols = data.shape

            for ix, cols, view in data.scan(axis=1, ):
                for k in cols:
                    celltype = label.iloc[k]["celltype"]

                    if celltype is not np.nan:
                        if not celltype in all_cell_types_dict:
                            all_cell_types_dict[celltype] = len(all_cell_types_dict)

                        c = view[:, k-ix].astype("int32")
                        gids = np.where(c)[0]
                        cnts = c[gids]

                        yield gids, cnts, all_cell_types_dict[celltype]

    def get_ds():
        return tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec([None], dtype=tf.int32),
                tf.TensorSpec([None], dtype=tf.int32),
                tf.TensorSpec([], dtype=tf.int32),
            ),
        )

    get_ds().save(str(outfile))

    with open(outfile/"metadata", "w") as f:
        json.dump(all_cell_types_dict, f)


def main(_):
    process_loom()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    app.run(main)

