from ml_collections.config_dict import ConfigDict


def get_fnet_config() -> ConfigDict:
    config = ConfigDict()
    config.vocab_size = 8192
    config.max_len = 40
    config.embed_dim = 256
    config.latent_dim = 512
    config.num_heads = 8
    return config


def _get_optimizer_config() -> ConfigDict:
    config = ConfigDict()
    config.learnable = False
    config.learning_rate = 1e-3
    config.type = "adam"
    return config


def _get_prng_config() -> ConfigDict:
    config = ConfigDict()
    config.key = 69
    return config


def _get_training_config() -> ConfigDict:
    config = ConfigDict()
    config.epochs = 1
    config.batch_size = 32
    return config


def get_config() -> ConfigDict:
    config = ConfigDict()
    config.fnet = get_fnet_config()
    config.optimizer = _get_optimizer_config()
    config.prng = _get_prng_config()
    config.training = _get_training_config()
    return config
