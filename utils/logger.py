#\utils\logger.py
import logging
from logging.handlers import RotatingFileHandler
from config.settings import settings
from pathlib import Path

class BotLogFilter(logging.Filter):
    """Фільтр для очищення зайвих повідомлень з логів"""
    
    def filter(self, record):
        # Приховуємо зайві WebSocket повідомлення
        if "Received message type: None, topic: orderbook" in record.getMessage():
            return False
        
        # Приховуємо деякі інші зайві повідомлення
        if any(msg in record.getMessage() for msg in [
            "RAW:",
            "Failed to parse JSON",
            "Empty trades list for",
            "Message without topic"
        ]):
            return False
            
        return True

def setup_logger():
    logger = logging.getLogger("bot")
    if logger.handlers:
        return logger

    mode = settings.logging.mode.lower()
    if mode == "debug":
        console_level_name = settings.logging.console_level_debug
        file_level_name = settings.logging.file_level_debug
    else:
        console_level_name = settings.logging.console_level_work
        file_level_name = settings.logging.file_level_work

    console_level = getattr(logging, console_level_name.upper(), logging.INFO)
    file_level = getattr(logging, file_level_name.upper(), logging.DEBUG)

    logger.setLevel(min(console_level, file_level))

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Ensure directory
    settings.logging.log_dir.mkdir(parents=True, exist_ok=True)

    fh = RotatingFileHandler(settings.logging.common_log, maxBytes=10_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(file_level)
    fh.addFilter(BotLogFilter())  # 🔑 Додаємо фільтр
    logger.addHandler(fh)

    eh = RotatingFileHandler(settings.logging.errors_log, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    eh.setFormatter(fmt)
    eh.setLevel(logging.ERROR)
    eh.addFilter(BotLogFilter())  # 🔑 Додаємо фільтр
    logger.addHandler(eh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(console_level)
    ch.addFilter(BotLogFilter())  # 🔑 Додаємо фільтр
    logger.addHandler(ch)

    logger.info(f"[LOGGER] mode={mode} console={console_level_name} file={file_level_name}")
    return logger

logger = setup_logger()