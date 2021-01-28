import logging
import json

log_dir = 'log/'

def make_logger(log_file, name='Pose'):
    
    """with open("logging.json", "rt") as file:
        config = json.load(file)
    
    logging.config.dictConfig(config)
    logger = logging.getLogger()"""

    logger = logging.getLogger(name)
    #log level의 가장 낮은 단계 DEBUG,  -> INFO -> WARNING ....
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    console = logging.StreamHandler()
    file_handler = logging.FileHandler(filename=log_dir+log_file+".log")
    
    console.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger
