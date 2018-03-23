from logging import getLogger

logger = getLogger(__name__)


def load_best_model_weight(model):
    """
    :param cchess_alphazero.agent.model.CChessModel model:
    :return:
    """
    return model.load(model.config.resource.model_best_config_path, model.config.resource.model_best_weight_path)


def save_as_best_model(model):
    """

    :param cchess_alphazero.agent.model.CChessModel model:
    :return:
    """
    return model.save(model.config.resource.model_best_config_path, model.config.resource.model_best_weight_path)


def need_to_reload_best_model_weight(model):
    """

    :param cchess_alphazero.agent.model.CChessModel model:
    :return:
    """
    if model.config.model.distributed:
        return load_best_model_weight(model)
    else:
        logger.debug("start reload the best model if changed")
        digest = model.fetch_digest(model.config.resource.model_best_weight_path)
        if digest != model.digest:
            return True

        logger.debug("the best model is not changed")
        return False

def load_model_weight(model, config_path, weight_path, name=None):
    if name is not None:
        logger.info(f"{name}: load model from {config_path}")
    return model.load(config_path, weight_path)

def save_as_next_generation_model(model):
    return model.save(model.config.resource.next_generation_config_path, model.config.resource.next_generation_weight_path)


def load_sl_best_model_weight(model):
    """
    :param cchess_alphazero.agent.model.CChessModel model:
    :return:
    """
    return model.load(model.config.resource.sl_best_config_path, model.config.resource.sl_best_weight_path)


def save_as_sl_best_model(model):
    """

    :param cchess_alphazero.agent.model.CChessModel model:
    :return:
    """
    return model.save(model.config.resource.sl_best_config_path, model.config.resource.sl_best_weight_path)
