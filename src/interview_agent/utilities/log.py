import json
import logging
from logging.config import dictConfig
from typing import Optional

logging_initialized = False


class JsonFormatter(logging.Formatter):
    def format(self, record):
        record_message = super().format(record)
        log_record = {
            "process_name": record.processName,
            "name": record.name,
            "message": record_message,
            "level": record.levelname,
            "time": record.created,
        }
        return json.dumps(log_record)


class ColorFormatter(logging.Formatter):
    green = "\x1b[32;20m"
    default = "\x1b[39;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s %(processName)15s %(levelname)-8s %(name)s: %(message)s"

    FORMATS = {  # noqa: RUF012
        logging.DEBUG: logging.Formatter(green + format + reset),
        logging.INFO: logging.Formatter(default + format + reset),
        logging.WARNING: logging.Formatter(yellow + format + reset),
        logging.ERROR: logging.Formatter(red + format + reset),
        logging.CRITICAL: logging.Formatter(bold_red + format + reset),
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        return log_fmt.format(record)


FORMATTERS = {
    "default": {"()": ColorFormatter},
    "json": {"()": JsonFormatter},
    "file": {
        "format": "%(asctime)s %(processName)15s %(levelname)-8s %(name)s: %(message)s"
    },
}


def get_console_handler_config_level() -> int:
    return logging.DEBUG


def get_file_handler_config_level() -> int:
    return logging.DEBUG


def get_mindsdb_log_level() -> int:
    console_handler_config_level = get_console_handler_config_level()
    file_handler_config_level = get_file_handler_config_level()

    return min(console_handler_config_level, file_handler_config_level)


def get_handlers_config(process_name: Optional[str]) -> dict:
    handlers_config = {}
    console_handler_config_level = logging.DEBUG
    if True:
        handlers_config["console"] = {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": console_handler_config_level,
        }

    file_handler_config_level = logging.DEBUG
    if True:
        file_name = "test.log"
        if process_name is not None:
            if "." in file_name:
                parts = file_name.rpartition(".")
                file_name = f"{parts[0]}_{process_name}.{parts[2]}"
            else:
                file_name = f"{file_name}_{process_name}"
        handlers_config["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "file",
            "level": file_handler_config_level,
            "filename": "/home/azureuser/voice-agent/logs/" + file_name,
            "maxBytes": 524288,  # 0.5 Mb
            "backupCount": 1,
        }
    return handlers_config


def configure_logging(process_name: Optional[str] = None) -> None:
    handlers_config = get_handlers_config(process_name)
    mindsdb_log_level = get_mindsdb_log_level()

    logging_config = dict(
        version=1,
        formatters=FORMATTERS,
        handlers=handlers_config,
        loggers={
            "": {  # root logger
                "handlers": list(handlers_config.keys()),
                "level": mindsdb_log_level,
            },
            "__main__": {
                "level": mindsdb_log_level,
            },
            "mindsdb": {
                "level": mindsdb_log_level,
            },
            "alembic": {
                "level": mindsdb_log_level,
            },
        },
    )

    dictConfig(logging_config)


def initialize_logging(process_name: Optional[str] = None) -> None:
    """Initialyze logging"""
    global logging_initialized
    if not logging_initialized:
        configure_logging(process_name)
        logging_initialized = True


# I would prefer to leave code to use logging.getLogger(), but there are a lot of complicated situations
# in MindsDB with processes being spawned that require logging to be configured again in a lot of cases.
# Using a custom logger-getter like this lets us do that logic here, once.
def getLogger(name: Optional[str] = None) -> logging.Logger:  # noqa: N802
    """
    Get a new logger, configuring logging first if it hasn't been done yet.
    """
    initialize_logging()
    return logging.getLogger(name)
