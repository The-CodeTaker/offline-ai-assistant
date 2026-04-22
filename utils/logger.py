from loguru import logger

def get_logger(name: str):
    """Returns a loguru logger bound to the specific module name."""
    return logger.bind(module=name)