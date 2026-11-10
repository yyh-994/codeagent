"""生产级日志配置 — 控制台 + 文件轮转 + text/json 双模式"""
import json
import os, logging, sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from back.common.config import LOG_LEVEL, LOG_FORMAT, LOG_MAX_MB, LOG_BACKUPS
from back.common.middleware import request_id_var

LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "logs"))
LOG_FILE = os.environ.get("LOG_FILE", os.path.join(LOG_DIR, "app.log"))


class RequestIdFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = request_id_var.get("-")
        return True


class SafeFormatter(logging.Formatter):
    """容错 formatter — 缺失字段不抛异常"""
    def format(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return super().format(record)


class JsonFormatter(logging.Formatter):
    """JSON 结构化格式 — 方便 ELK/日志平台检索"""
    def format(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "req_id": record.request_id,
            "logger": record.name,
            "msg": record.getMessage(),
        }, ensure_ascii=False)


def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    root.addFilter(RequestIdFilter())

    use_json = LOG_FORMAT == "json"

    if use_json:
        fmt = JsonFormatter()
    else:
        fmt = SafeFormatter(
            "%(asctime)s | %(levelname)-5s | [%(request_id)s] | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=LOG_MAX_MB * 1024 * 1024,
        backupCount=LOG_BACKUPS, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    for noisy in ("uvicorn", "uvicorn.error", "uvicorn.access", "httpx", "chromadb", "sentence_transformers", "watchfiles"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
        # uvicorn 自己的 formatter 不包含 request_id，避免 KeyError
        for h in logging.getLogger(noisy).handlers[:]:
            logging.getLogger(noisy).removeHandler(h)
        logging.getLogger(noisy).addHandler(console)

    root.info("Logging initialized | format=%s | file=%s | level=%s",
              "json" if use_json else "text", LOG_FILE, LOG_LEVEL)
