import logging
from pathlib import Path
from functools import partial

import ml_collections
import pandas as pd
import numpy as np
import h5py
import tensorflow as tf

import cfg_predictor

def region_count(data_path, label):
    from skimage.measure import regionprops
    from chioso.data import SGData2D
    
    sg = SGData2D.from_h5ad(data_path)

    if label.shape != sg.shape:
        raise ValueError(
            f"label has the shape: {label.shape}, which is different from that of the SGData {sg.shape}"
        )
    csr = sg.to_csr()

    for rp in regionprops(label):
        coords = rp["coords"]
        coords = coords[:, 0] * sg.shape[1] + coords[:, 1]
        s = csr[coords, :].sum(axis=0)
        gids = np.where(s)[0]
        cnts = s[gids]
        
        yield gids, cnts

def get_config():
    # we don't need to match the whole ref dataset
    # only use the 13.5 samples
    df=pd.read_csv("/home/FCAM/jyu/datasets/tome.ds/source/cell_annotate.csv").dropna()
    possible_outputs = df["celltype"].to_numpy()
    unique_outputs = np.unique(possible_outputs)
    output_dict = dict(zip(unique_outputs, range(len(unique_outputs))))

    selection = np.asarray(df["development_stage"] == 13.5)
    output_mask = np.zeros([len(unique_outputs)], dtype=bool)
    for ct in np.unique(possible_outputs[selection]):
        output_mask[output_dict[ct]] = True

    data_path = Path("../mosta/sg_data/E14.5_E1S3_labeled.h5")
    with h5py.File(data_path, mode="r", ) as data:
        label = np.asarray(data["uns/dapi_segm"]).astype(int)

    cache_name = data_path.stem + "_sc"
    cache_path = Path(data_path.parent/cache_name)
    if Path(str(cache_path) + ".index").exists():
        logging.warn(f"Found existing data cache file {cache_path}, which will be used. If this is not the correct cache, delete it and rerun the program")

    target_ds = tf.data.Dataset.from_generator(
        partial(region_count, data_path, label),
        output_signature=(tf.TensorSpec([None], tf.int32), tf.TensorSpec([None], tf.int32)),
    ).cache(str(cache_path))

    # prepare config dict
    config = ml_collections.ConfigDict()
    config.name = "single cell annotation"

    config.data_src = data_path.absolute()

    predictor_cfg = cfg_predictor.get_config()
    config.predictor = ml_collections.ConfigDict()
    config.predictor.model = predictor_cfg.model

    config.checkpoint_path = "train_predictor/240424-1102/1/"
    config.target_dataset = target_ds
    config.label = label
    config.ref_data_mask = selection
    config.output_mask = output_mask

    config.train = ml_collections.ConfigDict()
    config.train.batch_size = 4096 * 4
    config.train.padding = 4096
    config.train.n_steps = 10 * 1000
    config.train.val_interval = 1000

    return config
