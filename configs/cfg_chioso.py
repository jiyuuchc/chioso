import ml_collections

import cfg_predictor

def get_config():
    config = ml_collections.ConfigDict()
    config.name = "adversal_model"

    # import configs from predictor
    predictor_cfg = cfg_predictor.get_config()
    config.predictor = ml_collections.ConfigDict()
    config.predictor.model = predictor_cfg.model
    config.predictor.dataset = predictor_cfg.dataset
    config.predictor.checkpoint = "./checkpoint/"

    # overrides
    # config.predictor.model.config.normalize=False
    # config.predictor.model.config.log_transform=True

    config.dataset = ml_collections.ConfigDict()
    config.dataset.path = "./"
    config.dataset.train_files = "*.h5"
    config.dataset.patch_size = 1024
    config.dataset.grid_size = 1000
    config.dataset.binning = 4
    config.dataset.bucket_size = 524288

    config.train = ml_collections.ConfigDict()
    config.train.seed = 4242
    config.train.train_steps = 30000
    config.train.checkpoint_interval = 3000
    config.train.lr = 1e-3
    config.train.weight_decay = 1e-3
    config.train.ref_batch_size = 16384

    config.model = ml_collections.ConfigDict(dict(
        depths = (3,9,3),
        dims = (256, 384, 512),
        dropout = 0.3,
        fpn_dim = 384,
        roi = 9,
        att_ks = 5,
        learned_scaling = False,
    ))

    config.discriminator = ml_collections.ConfigDict()
    config.discriminator.n_layers = 4

    config.inference = ml_collections.ConfigDict()
    config.inference.sxy = 1.5
    config.inference.compat = 20
    config.inference.iters = 0

    return config
