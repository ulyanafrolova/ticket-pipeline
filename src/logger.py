import json
import logging
import sys
from datetime import datetime, timezone

_LOGRECORD_ATTRS = frozenset({
    "name", "msg", "args", "created", "filename", "funcName", "levelname",
    "levelno", "lineno", "module", "msecs", "message", "pathname", "process",
    "processName", "relativeCreated", "thread", "threadName", "stack_info",
    "exc_info", "exc_text", "taskName",
})

class _JsonFormatter(logging.Formatter):
    def format(self, record):
        record.message = record.getMessage()
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "component": record.name,
            "message": record.message,
        }
        for key, val in record.__dict__.items():
            if key not in _LOGRECORD_ATTRS:
                log_obj[key] = val
        return json.dumps(log_obj)


class _RunIdFilter(logging.Filter):
    """Injects the pipeline run_id into every record emitted by the logger."""

    def __init__(self, run_id: str):
        super().__init__()
        self.run_id = run_id

    def filter(self, record):
        if not hasattr(record, "run_id"):
            record.run_id = self.run_id
        return True


def get_logger(component_name: str, run_id: str = None) -> logging.Logger:
    """
    Return a logger that emits JSON to stdout.
    Each log line is a JSON object on a single line.
    If run_id is provided, it is included in every log line as "run_id".
    """
    logger = logging.getLogger(component_name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
        logger.setLevel(logging.DEBUG)
    if run_id is not None:
        for f in list(logger.filters):
            if isinstance(f, _RunIdFilter):
                logger.removeFilter(f)
        logger.addFilter(_RunIdFilter(run_id))
    return logger
