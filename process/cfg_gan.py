from __future__ import annotations

from pathlib import Path

import ml_collections

import cfg_predictor

def get_mask(h5file):
    import h5py
    from skimage.transform import rescale

    with h5py.File(h5file, 'r') as data:
        label = data["uns/dapi_segm"][...]
    mask = label != 0
    mask = rescale(mask, 0.25)
    return (mask >= 0.5).astype("uint8")

def get_config():
    config = ml_collections.ConfigDict()
    config.name = "gan model"

    predictor_cfg = cfg_predictor.get_config()
    config.predictor = ml_collections.ConfigDict()
    config.predictor.model = predictor_cfg.model
    config.predictor.checkpoint = str(Path("./train_predictor/240424-1102/1/").absolute())

    config.dataset = ml_collections.ConfigDict()
    config.dataset.train = Path("../mosta/sg_data/E16.5_E1S3_labeled.ds")
    # config.dataset.mask = get_mask("../mosta/sg_data/E16.5_E1S3_labeled.h5")

    config.train = ml_collections.ConfigDict()
    config.train.seed = 4242
    config.train.train_steps = 25000
    config.train.validation_interval = 2500
    config.train.lr = 1e-4
    config.train.weight_decay = 1e-3
    config.train.ref_batch_size = 16384

    config.model = ml_collections.ConfigDict()
    config.model.shape2d = (256, 256)
    config.model.normalize = False

    config.num_runs = 1

    return config
