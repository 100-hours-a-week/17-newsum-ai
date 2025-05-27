# ai/app/utils/logger.py
import logging
import logging.config
import yaml
from pathlib import Path
import sys
import json
from typing import Any, Dict, List, Optional

# --- 로깅 설정 파일 경로 및 로그 디렉토리 ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # ai/
DEFAULT_LOGGING_CONFIG_PATH = PROJECT_ROOT / 'logging_config.yaml'
# ⬇️ 기본 로그 디렉토리 경로를 프로젝트 루트 기준으로 변경 ⬇️
DEFAULT_LOG_DIR = PROJECT_ROOT / 'logs'


# --- 로깅 필터 ---
class ContextFilter(logging.Filter):
    """로그 레코드에 trace_id, node_name, retry_count를 추가하는 필터 (기본값 포함)"""

    def filter(self, record):
        record.trace_id = getattr(record, 'trace_id', 'N/A')
        record.node_name = getattr(record, 'node_name', 'N/A')
        record.retry_count = getattr(record, 'retry_count', 0)
        return True


# --- 로깅 설정 함수 ---
def setup_logging(config_path: Path = DEFAULT_LOGGING_CONFIG_PATH, default_level=logging.INFO):
    """
    설정 파일(YAML)을 읽어 로깅 시스템을 설정합니다.
    파일이 없으면 기본 설정을 적용하고, 컨텍스트 필터를 핸들러에 추가합니다.
    로그 디렉토리 생성도 시도합니다 (YAML 설정에 따라).
    """
    log_config = None
    log_dir_to_ensure = DEFAULT_LOG_DIR  # 기본값으로 초기화

    if config_path.exists():
        try:
            with open(config_path, 'rt', encoding='utf-8') as f:
                log_config = yaml.safe_load(f.read())

            if log_config and isinstance(log_config, dict):
                # --- YAML 설정에서 file_handler의 filename 경로 처리 ---
                file_handler_config = log_config.get('handlers', {}).get('file_handler')
                if file_handler_config and 'filename' in file_handler_config:
                    original_filename = Path(file_handler_config['filename'])
                    if not original_filename.is_absolute():
                        # 상대 경로이면 PROJECT_ROOT 기준으로 절대 경로화
                        absolute_log_file_path = (PROJECT_ROOT / original_filename).resolve()
                        # dictConfig에 적용하기 위해 문자열로 변환하여 설정 업데이트
                        file_handler_config['filename'] = str(absolute_log_file_path)
                        log_dir_to_ensure = absolute_log_file_path.parent
                        print(
                            f"Log Handler: Relative filename '{original_filename}' in YAML resolved to '{absolute_log_file_path}'")
                    else:
                        # 절대 경로이면 해당 경로의 부모 디렉토리 사용
                        log_dir_to_ensure = original_filename.parent
                        print(f"Log Handler: Absolute filename '{original_filename}' found in YAML.")
                else:
                    print(
                        f"Log Handler: No 'file_handler' with 'filename' found in YAML. Defaulting log directory to: {DEFAULT_LOG_DIR}")
                    # log_dir_to_ensure는 이미 DEFAULT_LOG_DIR로 설정되어 있음
            else:
                print(
                    f"Warning: Logging config file is empty or not a dictionary: {config_path}. Log directory check skipped.")
                # log_dir_to_ensure는 이미 DEFAULT_LOG_DIR로 설정되어 있음

        except Exception as e:
            print(
                f"Warning: Error reading logging config for directory check ({config_path}): {e}. Defaulting log directory to: {DEFAULT_LOG_DIR}")
            # log_config는 None으로 유지되어 basicConfig로 폴백됨
            log_config = None  # 명시적으로 None 처리하여 basicConfig 폴백 유도

    # --- 로그 디렉토리 생성 ---
    try:
        log_dir_to_ensure.mkdir(parents=True, exist_ok=True)
        print(f"Log directory ensured/created: {log_dir_to_ensure}")
    except OSError as e:
        print(f"Warning: Failed to create log directory {log_dir_to_ensure}: {e}. File logging might fail.")

    # --- 로깅 설정 적용 ---
    if log_config and isinstance(log_config, dict):  # 유효한 설정이 로드되었는지 확인
        try:
            logging.config.dictConfig(log_config)
            print(f"Logging configuration successfully loaded from: {config_path}")
        except Exception as e:
            print(f"CRITICAL: Error applying logging config from '{config_path}': {e}", file=sys.stderr)
            print("CRITICAL: Falling back to basic logging.", file=sys.stderr)
            logging.basicConfig(level=default_level,
                                format="[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s (BasicConfig)",
                                stream=sys.stdout)
    else:
        if config_path.exists():  # 파일은 존재하나 내용이 부적절했던 경우
            print(f"Warning: Logging config file '{config_path}' was found but could not be used. Using basic logging.")
        else:  # 파일 자체가 없는 경우
            print(f"Warning: Logging config file not found: {config_path}. Using basic logging.")
        logging.basicConfig(level=default_level,
                            format="[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s (BasicConfig)",
                            stream=sys.stdout)

    # --- 컨텍스트 필터를 모든 핸들러에 추가 ---
    # (basicConfig 사용 시 루트 로거에 핸들러가 자동 추가됨)
    # (dictConfig 사용 시 설정된 핸들러들이 존재함)
    context_filter = ContextFilter()

    # 루트 로거 및 모든 정의된 로거의 핸들러에 필터 추가
    loggers_to_process = [logging.root] + [logging.getLogger(name) for name in logging.root.manager.loggerDict]

    applied_handlers_count = 0
    for logger_instance in loggers_to_process:
        if hasattr(logger_instance, 'handlers'):
            for handler in logger_instance.handlers:
                if not any(isinstance(f, ContextFilter) for f in handler.filters):
                    try:
                        handler.addFilter(context_filter)
                        applied_handlers_count += 1
                    except Exception as e:  # 핸들러에 필터 추가 실패 시 경고
                        print(
                            f"Warning: Could not add ContextFilter to handler {handler} for logger {logger_instance.name}: {e}")

    if applied_handlers_count > 0:
        print(f"ContextFilter applied to {applied_handlers_count} handler(s).")
    else:
        # BasicConfig 사용 시에도 루트 핸들러가 있으므로 이 경우는 거의 발생 안 함
        print("Warning: No handlers found to apply ContextFilter. This might indicate a deeper logging setup issue.")


# --- 로거 인스턴스 반환 함수 ---
_loggers: Dict[str, logging.Logger] = {}


def get_logger(name: str) -> logging.Logger:
    """이름별로 로거 인스턴스를 반환합니다. (캐싱 사용)"""
    global _loggers
    if name not in _loggers:
        _loggers[name] = logging.getLogger(name)
        # get_logger 호출 시점에는 이미 setup_logging이 완료되었다고 가정.
        # 만약 setup_logging 전에 get_logger가 호출될 가능성이 있다면,
        # 여기서 로거 레벨 등을 기본값으로 설정하는 로직이 필요할 수 있으나,
        # 보통은 애플리케이션 시작점(main, lifespan 등)에서 setup_logging을 먼저 호출.
    return _loggers[name]


# --- 로깅용 데이터 요약 헬퍼 함수 (이전과 동일) ---
def _safe_serialize_value(value: Any, max_len: int) -> Any:
    if isinstance(value, dict):
        keys_preview = list(value.keys())[:3]
        keys_str = ", ".join(map(str, keys_preview))
        if len(value) > 3: keys_str += "..."
        return f"{{Dict len={len(value)}, keys=[{keys_str}]}}"
    elif isinstance(value, list):
        items_preview = ""
        if value:
            try:
                first_item_str = str(value[0])
                if len(first_item_str) > 30:
                    first_item_str = first_item_str[:30] + "..."
            except Exception:
                first_item_str = "(...)"
            items_preview = f"items=[{first_item_str}" + ("...]" if len(value) > 1 else "]")
        return f"[List len={len(value)}, {items_preview}]"
    elif isinstance(value, str):
        if len(value) > max_len:
            return value[:max_len] + "..."
        return value
    elif isinstance(value, bytes):
        try:
            decoded = value.decode('utf-8', errors='ignore')
            prefix = "<bytes"
            if len(value) > max_len:
                return f"{prefix} len={len(value)}> {decoded[:max_len]}..."
            else:
                return f"{prefix} len={len(value)}> {decoded}"
        except Exception:
            return f"<bytes len={len(value)}>"
    elif isinstance(value, (int, float, bool, type(None))):
        return value
    else:
        try:
            s_val = str(value)
            type_name = type(value).__name__
            if len(s_val) > max_len:
                return f"<{type_name} '{s_val[:max_len]}...'>"
            return f"<{type_name} '{s_val}'>"
        except Exception as e:
            return f"<{type(value).__name__} object (str error: {e})>"


def summarize_for_logging(
        data: Any,
        max_len: int = 100,
        fields_to_show: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None
) -> str:
    summary = {}
    target_dict: Optional[Dict] = None

    try:
        if hasattr(data, 'model_dump'):
            target_dict = data.model_dump(exclude_none=False)
        elif isinstance(data, dict):
            target_dict = data.copy()
        else:
            return str(_safe_serialize_value(data, max_len))

        if target_dict is None: return "{}"
        keys_to_exclude = set(exclude_keys) if exclude_keys else set()

        if fields_to_show is not None:
            keys_to_process = [k for k in fields_to_show if k in target_dict and k not in keys_to_exclude]
        else:
            keys_to_process = [k for k in target_dict.keys() if k not in keys_to_exclude]

        for field_name in keys_to_process:
            value = target_dict[field_name]
            summary[field_name] = _safe_serialize_value(value, max_len)

        return json.dumps(summary, indent=None, ensure_ascii=False, default=str)
    except Exception as e:
        try:
            fallback_summary = {"logging_summary_error": str(e)}
            if isinstance(data, dict):
                fallback_summary["data_keys"] = list(data.keys())[:5]  # 최대 5개 키
            elif hasattr(data, '__dict__'):
                fallback_summary["data_keys"] = list(data.__dict__.keys())[:5]
            return json.dumps(fallback_summary, default=str)
        except Exception:
            return '{ "logging_summary_error": "Failed to serialize summary error during fallback." }'