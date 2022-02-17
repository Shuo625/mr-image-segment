import logging


def init_logger(name, log_file, level=logging.INFO):
    logger = logging.getLogger(name)
    
    logger.parent = None
    logger.setLevel(level)

    # Output to the console and the file
    ch = logging.StreamHandler()
    fh = logging.FileHandler(log_file)

    MSG_FORMATTER = '[%(levelname)s][%(asctime)s]"%(pathname)s:%(lineno)d": %(message)s'
    DATE_FORMATTER = '%Y-%m-%d %H:%M'
    formatter = logging.Formatter(fmt=MSG_FORMATTER, datefmt=DATE_FORMATTER)

    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(ch)
    logger.addHandler(fh)


def get_logger(name):
    return logging.getLogger(name)

