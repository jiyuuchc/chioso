#!/usr/bin/env python

from __future__ import annotations

import json
import logging
import numpy as np
import pprint
import tensorflow as tf

from anndata import read_h5ad
from absl import flags, app
from pathlib import Path

_FLAGS = flags.FLAGS
flags.DEFINE_string("data", None, "path to h5ad file.")
flags.DEFINE_string("genes", None, "path to gene list file.")
flags.DEFINE_string("outdir", ".", "output dir")
flags.DEFINE_string("col", "celltype", "the column of adata.obs indicating cell type")

def process_h5ad():
    src = Path(_FLAGS.data)
    genefile = Path(_FLAGS.genes)
    outdir = Path(_FLAGS.outdir)
    label_col = _FLAGS.col

    outfile = outdir/(src.stem + ".ds")

    adata = read_h5ad(src, backed='r')
    adata.var_names_make_unique()

    logging.info(f"Read gene names from {genefile}")
    with open(genefile) as f:
        genes = json.load(f)
        if isinstance(genes, dict):
            genes = list(genes.keys())
        gene_to_id = dict(zip(genes, range(len(genes))))

    orig_genes = adata.var_names
    lut = np.array([gene_to_id[g] if g in gene_to_id else -1 for g in orig_genes])

    all_cell_types_dict = {}

    def gen():
        for row, obs in zip(adata.X, adata.obs.iterrows()):
            celltype = obs[1][label_col]

            if celltype is not np.nan and not str(celltype) in ("nan", "NA"):
                if not celltype in all_cell_types_dict:
                    all_cell_types_dict[celltype] = len(all_cell_types_dict)

                gids = lut[row.indices]
                cnts = row.data
                sel = gids >= 0
                cnts = cnts[sel]
                gids = gids[sel]

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

    logging.info(f"Writing dataset to {outfile}")
    get_ds().save(str(outfile))

    print("Done creating dataset for these cell types:")

    pprint.pp(all_cell_types_dict)

def main(_):
    process_h5ad()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    app.run(main)
