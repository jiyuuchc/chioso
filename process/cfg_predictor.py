from __future__ import annotations

import ml_collections

from chioso.modules import LinearPredictor

def get_config():
    config = ml_collections.ConfigDict()
    config.name = "celltype_linear_model"

    config.dataset = ml_collections.ConfigDict()
    config.dataset.train = "../mosta/ref_data/tome_train.ds"
    config.dataset.val = "../mosta/ref_data/tome_val.ds"

    config.train = ml_collections.ConfigDict()
    config.train.seed = 1234
    config.train.batchsize = 128
    config.train.train_steps = 100000
    config.train.validation_interval = 10000
    config.train.lr = 1e-4
    config.train.weight_decay = 1e-3

    config.model = ml_collections.ConfigDict()
    config.model.type = LinearPredictor
    config.model.config = ml_collections.ConfigDict()
    config.model.config.n_genes = 27504
    config.model.config.dim_out = 68
    config.model.config.dropout = 0.2
    config.model.normalize = True
    config.model.log_transform = False

    config.num_runs = 1

    return config
