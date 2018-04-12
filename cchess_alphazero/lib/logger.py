from logging import StreamHandler, basicConfig, DEBUG, getLogger, Formatter, FileHandler


def setup_logger(log_filename):
    format_str = '%(asctime)s@%(name)s %(levelname)s # %(message)s'
    basicConfig(filename=log_filename, level=DEBUG, format=format_str)
    stream_handler = StreamHandler()
    stream_handler.setFormatter(Formatter(format_str))
    getLogger().addHandler(stream_handler)

def setup_file_logger(log_filename):
    format_str = '%(asctime)s@%(name)s %(levelname)s # %(message)s'
    basicConfig(filename=log_filename, level=DEBUG, format=format_str)

if __name__ == '__main__':
    setup_logger("test.log")
    logger = getLogger("test")
    logger.info("OK")
