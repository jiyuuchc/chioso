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

    # The checkpoint directory used by chioso.train-predictor
    config.predictor.checkpoint = "."

    config.dataset = ml_collections.ConfigDict()

    # The training data directory, where all the h5 files are
    config.dataset.path = "./sg_data/"

    # A filter for the files use for training.
    config.dataset.train_files = "*.h5"

    # The spatial dimension of each training sample
    config.dataset.patch_size = 1024

    # Should be slightly smaller than patch_size to allow stitching patches together during inference
    config.dataset.grid_size = 1000

    # Pre-binning the ST data if necessary
    config.dataset.binning = 4

    # Padding the input data to avoid ragged input data sizes
    config.dataset.bucket_size = 524288

    config.train = ml_collections.ConfigDict()

    # A seed for random number generator
    config.train.seed = 4242

    # Number of training steps
    config.train.train_steps = 30000

    # How frequently should we save checkpoints
    config.train.checkpoint_interval = 3000

    # Learning rate
    config.train.lr = 1e-3

    # Model weight L2 regularozation factor
    config.train.weight_decay = 1e-3

    # Batch size for the reference training data
    config.train.ref_batch_size = 16384

    # Model hyperparameters
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

    # Number of iterations of CRF filtering during inference. Set to 0 to disable
    config.inference.iters = 0

    # The distance correlation of the CRF filter. Bigger value enfores a spatially smoother output
    config.inference.sxy = 1.5

    # The energy factor of the CRF filter. Bigger value enfores a spatially smoother output
    config.inference.compat = 20


    return config
