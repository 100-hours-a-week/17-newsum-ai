# src/core/utils.py
import hashlib
import json
from pathlib import Path
import logging
from src import settings

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

def generate_id_from_url(url: str) -> str:
    """URL 기반 고유 ID 생성"""
    return hashlib.sha1(url.encode()).hexdigest()[:10]

def save_json(data: dict, filepath: Path):
    """JSON 파일 저장"""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"JSON saved: {filepath}")
    except Exception as e:
        logger.error(f"Failed to save JSON {filepath}: {e}")

def save_image(image_obj, filepath: Path):
    """Pillow 이미지 객체 파일 저장"""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        image_obj.save(filepath)
        logger.info(f"Image saved: {filepath}")
    except Exception as e:
        logger.error(f"Failed to save image {filepath}: {e}")

# ----------------------------------------