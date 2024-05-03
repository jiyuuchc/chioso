from __future__ import annotations

from pathlib import Path

import ml_collections

import cfg_predictor

def get_config():
    config = ml_collections.ConfigDict()
    config.name = "adversal_model"

    predictor_cfg = cfg_predictor.get_config()
    config.predictor = ml_collections.ConfigDict()
    config.predictor.model = predictor_cfg.model
    config.predictor.checkpoint = str(Path("./train_predictor/240423-2258/2").absolute()) # need to be absolute path

    config.dataset = ml_collections.ConfigDict()
    config.dataset.path = "../mosta/sg_data/"
    config.dataset.name = "E12.5_E1S3_labeled"
    config.dataset.ref_name = "ref_embedding.ds"
    config.dataset.patch_shape = (1024, 1024)
    config.dataset.border_size = (1024-768) // 2
    config.dataset.binning = 4
    config.dataset.n_genes = 27504
    config.dataset.bucket_size = 524288

    config.train = ml_collections.ConfigDict()
    config.train.seed = 4242
    config.train.train_steps = 25000
    config.train.validation_interval = 2500
    config.train.lr = 1e-4
    config.train.weight_decay = 1e-3
    config.train.ref_batch_size = 16384

    config.model = ml_collections.ConfigDict()
    config.model.normalize = False
    config.model.roi = 8
    config.model.depths = (3,9,3)
    config.model.dims = (256, 384, 512)
    config.model.dropout = 0.0
    config.model.fpn_dim = 384
    config.model.att_ks = 8

    return config
