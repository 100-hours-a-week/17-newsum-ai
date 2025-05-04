# app/utils/logger.py (최종 버전)

import logging
import logging.config
import yaml
import os
import json
from typing import Any

# 기본 로깅 설정 함수
def setup_logging(config_path='app/config/logging_config.yaml', default_level=logging.INFO):
    """
    설정 파일(YAML)을 읽어 로깅 시스템을 설정합니다.
    파일이 없으면 기본 설정을 적용합니다.
    """
    path = config_path
    log_config_loaded = False
    if os.path.exists(path):
        try:
            # --- 디렉토리 생성 로직 추가 ---
            # YAML 설정 파일 내의 핸들러 경로를 읽어 디렉토리 생성 시도
            with open(path, 'rt', encoding='utf-8') as f:
                config = yaml.safe_load(f.read())
                if config and 'handlers' in config:
                    for handler_name, handler_config in config.get('handlers', {}).items():
                        if 'filename' in handler_config:
                            log_dir = os.path.dirname(handler_config['filename'])
                            if log_dir: # 빈 문자열이 아닌 경우
                                os.makedirs(log_dir, exist_ok=True)
                                print(f"Ensured log directory exists: {log_dir}")
                # ---------------------------
                logging.config.dictConfig(config) # 설정 적용
                log_config_loaded = True
                print(f"Logging configured successfully from {path}")
        except Exception as e:
            print(f"Error loading logging configuration from {path}: {e}")
            print("Falling back to basic logging configuration.")
            # 기본 설정 시에도 로그 디렉토리 생성 시도 (선택적)
            # default_log_dir = "app/log"
            # os.makedirs(default_log_dir, exist_ok=True)
            logging.basicConfig(level=default_level, format="[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s")
    else:
        print(f"Logging configuration file not found at {path}. Using basic logging configuration.")
        logging.basicConfig(level=default_level, format="[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s")

# 로거 인스턴스 반환 함수
def get_logger(name: str) -> logging.Logger:
    """
    이름별로 로거 인스턴스를 반환합니다. 설정은 setup_logging을 통해 적용됩니다.
    """
    return logging.getLogger(name)
# --- 상태/업데이트 요약 로깅 헬퍼 함수 ---

# 내부 함수: 객체를 안전하게 JSON 직렬화 가능한 형태로 변환
def _safe_serialize_value(value, max_len=100):
    if isinstance(value, (str, int, float, bool, type(None))):
        # 문자열 길이 제한
        return str(value)[:max_len] + ('...' if len(str(value)) > max_len else '') if isinstance(value, str) else value
    elif isinstance(value, list):
        return f"[List len={len(value)}]"
    elif isinstance(value, dict):
        return f"{{Dict len={len(value)}}}"
    try:
        return str(value) # 기타 타입은 문자열로
    except Exception:
        return f"<{type(value).__name__} object (unserializable)>"

# State 객체 또는 업데이트 딕셔너리를 로깅용 문자열로 요약
def summarize_for_logging(data: Any, is_state_object: bool = True, max_len=100) -> str:
    """
    State 객체 또는 노드 업데이트 딕셔너리를 로깅용 JSON 문자열로 요약합니다.

    Args:
        data: State 객체 또는 업데이트 딕셔너리.
        is_state_object: data가 State 객체인지 여부 (True면 __dict__ 접근 시도).
        max_len: 문자열 값의 최대 로깅 길이.

    Returns:
        JSON 형식의 요약 문자열.
    """
    summary = {}
    # State 객체면 __dict__ 사용, 아니면 data 자체(딕셔너리) 사용
    target_dict = data.__dict__ if is_state_object and hasattr(data, '__dict__') else data

    if not isinstance(target_dict, dict):
        return f"Cannot summarize non-dict data of type: {type(data)}"

    # 모든 필드/키에 대해 요약 생성 시도
    for field_name, value in target_dict.items():
        if value is None: # None 값은 제외 (선택 사항)
            continue

        # 특정 필드는 더 상세하게 또는 다르게 처리
        if field_name in ["trace_id", "comic_id", "timestamp", "initial_query", "decision", "error_message"]:
            summary[field_name] = _safe_serialize_value(value, max_len)
        elif field_name == "config":
            summary[field_name] = f"{{Keys: {list(value.keys())}}}" if isinstance(value, dict) else type(value).__name__
        elif field_name == "topic_analysis":
             main_topic = value.get('main_topic', '') if isinstance(value, dict) else ''
             summary[field_name] = f"{{topic: '{str(main_topic)[:30]}...'}}" if main_topic else type(value).__name__
        elif field_name in ["final_summary", "progress_report", "scenario_prompt"]:
            summary[field_name] = f"'{_safe_serialize_value(value, max_len)}'" # 따옴표 추가
        elif field_name == "chosen_idea": # 아이디어 노드용
             title = value.get('idea_title', '') if isinstance(value, dict) else ''
             summary[field_name] = f"{{title: '{str(title)[:30]}...'}}" if title else type(value).__name__
        # 리스트/딕셔너리는 길이만
        elif isinstance(value, list):
            summary[field_name] = f"[List len={len(value)}]"
        elif isinstance(value, dict):
            # 상세 처리가 필요한 다른 dict 필드 추가 가능
            summary[field_name] = f"{{Dict len={len(value)}}}"
        # 처리 시간 필드
        elif isinstance(value, (float, int)) and "processing_stats" in field_name:
             summary[field_name] = f"{value:.2f}s"
        # 기타: 타입 이름 로깅 (선택적)
        # else:
        #     summary[field_name] = type(value).__name__

    try:
        # 보기 좋게 JSON 문자열로 변환 (들여쓰기, 한글 유지)
        return json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        # JSON 변환 실패 시 오류 메시지 반환
        return f"Error summarizing data for logging: {e}"

# --- 사용 예시 (참고용) ---
# # 앱 초기화 시 (예: main.py)
# from app.utils.logger import setup_logging, get_logger
#
# # YAML 설정 파일 로드 및 로깅 시스템 초기화
# # 이 함수 내에서 app/log 디렉토리 존재 여부 확인 및 생성을 시도합니다.
# setup_logging('logging_config.yaml')
#
# # 이후 필요한 곳에서 로거 가져와 사용
# module_logger = get_logger("my_module")
# module_logger.info("My module started.")
# module_logger.debug("This is a debug message.")