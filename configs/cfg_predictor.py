import ml_collections

from chioso.modules import LinearPredictor

def get_config():
    config = ml_collections.ConfigDict()
    config.name = "celltype_linear_model"

    config.dataset = ml_collections.ConfigDict()

    # Training data location. This is the output of chioso.pp-ref
    config.dataset.path = "ref_data/tome.ds"

    # Output file name
    config.dataset.outname = "ref_embedding.ds"

    # We randomly mask-off certain portion of the input data during training, which help increases
    # the model robustness.
    config.dataset.dropout = 0.5

    config.train = ml_collections.ConfigDict()

    # Seed value for the random number generator
    config.train.seed = 1234

    # Training batch size. Adjust according to the GPU memory size
    config.train.batchsize = 128

    # Training steps. The default value is good for a medium size (~ 1 million cells) dataset
    config.train.train_steps = 100000

    # How frequent should we compute validation metrics
    config.train.validation_interval = 10000

    # Learning rate
    config.train.lr = 1e-4

    # Model weight L2 regularization factor
    config.train.weight_decay = 1e-3

    # Fraction of the training data reserved for validation purpose
    config.train.val_split = 0.2

    # Whether to train for the balanced loss or simple cross-entropy loss
    config.train.balanced_loss = True

    config.model = ml_collections.ConfigDict()

    # Model type. Don't change
    config.model.type = LinearPredictor

    config.model.config = ml_collections.ConfigDict()

    # Number of genes in the dataset
    config.model.config.n_genes = 27504

    # Number of cell types
    config.model.config.dim_out = 68

    # Dropout rate during training
    config.model.config.dropout = 0.2

    # Whether to normalized gene expression profile
    config.model.config.normalize = False

    # Whether to perform log1p transformation
    config.model.config.log_transform = False

    # Dimsion of the latent features
    config.model.config.dim_hidden = 256

    return config
