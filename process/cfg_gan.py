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
    config.predictor.checkpoint = "/home/FCAM/jyu/work/chioso/runs/train_predictor/240507-1904"

    # overrides
    # config.predictor.model.config.normalize=False
    # config.predictor.model.config.log_transform=True

    config.dataset = ml_collections.ConfigDict()
    config.dataset.path = "/home/FCAM/jyu/work/chioso/mosta/sg_data"
    config.dataset.train_files = "E16.5_*.h5"
    config.dataset.test_file = "E16.5_E1S3_labeled.h5"
    config.dataset.patch_size = 1024
    config.dataset.grid_size = 768
    config.dataset.binning = 4
    config.dataset.bucket_size = 524288

    config.train = ml_collections.ConfigDict()
    config.train.seed = 4242
    config.train.train_steps = 25000
    config.train.validation_interval = 2500
    config.train.lr = 1e-3
    config.train.weight_decay = 1e-3
    config.train.ref_batch_size = 16384

    config.model = ml_collections.ConfigDict()
    config.model.roi = 8
    config.model.depths = (3,9,3)
    config.model.dims = (256, 384, 512)
    config.model.dropout = 0.3
    config.model.fpn_dim = 384
    config.model.att_ks = 4

    return config
