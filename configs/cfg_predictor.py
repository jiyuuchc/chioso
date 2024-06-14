import ml_collections

from chioso.modules import LinearPredictor

def get_config():
    config = ml_collections.ConfigDict()
    config.name = "celltype_linear_model"

    config.dataset = ml_collections.ConfigDict()
    config.dataset.path = "ref_data/tome.ds"
    config.dataset.outname = "ref_embedding.ds"
    config.dataset.dropout = 0.5

    config.train = ml_collections.ConfigDict()
    config.train.seed = 1234
    config.train.batchsize = 128
    config.train.train_steps = 100000
    config.train.validation_interval = 10000
    config.train.lr = 1e-4
    config.train.weight_decay = 1e-3
    config.train.val_split = 0.2
    config.train.balanced_loss = True

    config.model = ml_collections.ConfigDict()
    config.model.type = LinearPredictor
    config.model.config = ml_collections.ConfigDict()
    config.model.config.n_genes = 27504
    config.model.config.dim_out = 68
    config.model.config.dropout = 0.2
    config.model.config.normalize = False
    config.model.config.log_transform = False
    config.model.config.dim_hidden = 256

    return config
