# app/utils/logger.py

import logging

def get_logger(name: str) -> logging.Logger:
    """
    이름별로 로거를 생성해서 반환합니다.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

# 기본 글로벌 로거
logger = get_logger("newsom")
