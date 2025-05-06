# ai/app/utils/logger.py
import logging
import logging.config
import yaml
from pathlib import Path
import sys
import json  # JSON 직렬화 위해 추가
from typing import Any, Dict, List, Optional  # 타입 힌트 추가

# --- 로깅 설정 파일 경로 및 로그 디렉토리 ---
# 이 파일 기준 상위 2단계인 ai/ 디렉토리를 프로젝트 루트로 가정
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOGGING_CONFIG_PATH = PROJECT_ROOT / 'logging_config.yaml'
# 기본 로그 디렉토리 경로도 설정 파일 로드 전에 필요할 수 있으므로 정의
DEFAULT_LOG_DIR = PROJECT_ROOT / 'app' / 'log'


# --- 로깅 필터 ---
class ContextFilter(logging.Filter):
    """로그 레코드에 trace_id, node_name, retry_count를 추가하는 필터 (기본값 포함)"""

    def filter(self, record):
        # LogRecord 객체에 해당 속성이 없으면 defaults 값 사용 (포매터와 연동)
        record.trace_id = getattr(record, 'trace_id', 'N/A')
        record.node_name = getattr(record, 'node_name', 'N/A')
        # retry_count는 정수형이어야 하므로 기본값 0 사용
        record.retry_count = getattr(record, 'retry_count', 0)
        return True


# --- 로깅 설정 함수 ---
def setup_logging(config_path: Path = DEFAULT_LOGGING_CONFIG_PATH, default_level=logging.INFO):
    """
    설정 파일(YAML)을 읽어 로깅 시스템을 설정합니다.
    파일이 없으면 기본 설정을 적용하고, 컨텍스트 필터를 핸들러에 추가합니다.
    로그 디렉토리 생성도 시도합니다.
    """
    log_config_loaded = False
    config = None
    log_dir_to_ensure = DEFAULT_LOG_DIR  # 기본 로그 디렉토리

    # --- 설정 파일에서 로그 디렉토리 경로 읽기 시도 (선택적) ---
    if config_path.exists():
        try:
            with open(config_path, 'rt', encoding='utf-8') as f_temp:
                temp_config = yaml.safe_load(f_temp.read())
                if isinstance(temp_config, dict) and 'handlers' in temp_config:
                    # 파일 핸들러를 우선적으로 찾아 경로 설정 시도
                    handler_configs = temp_config.get('handlers', {})
                    found_file_handler = False
                    for handler_config in handler_configs.values():
                        if isinstance(handler_config, dict) and 'filename' in handler_config:
                            log_dir_path = Path(handler_config['filename']).resolve().parent
                            # 프로젝트 루트 내에 있는지 확인 (선택적 보안 강화)
                            try:
                                if log_dir_path.is_relative_to(PROJECT_ROOT):
                                    log_dir_to_ensure = log_dir_path
                                    found_file_handler = True
                                    break  # 첫 번째 유효한 경로만 사용
                                else:
                                    print(
                                        f"Warning: Log directory '{log_dir_path}' from config is outside project root {PROJECT_ROOT}. Using default: {DEFAULT_LOG_DIR}")
                            except ValueError:  # Python < 3.9 에서는 is_relative_to 없음, 또는 경로 구조가 다를 때
                                print(
                                    f"Warning: Could not verify if log directory '{log_dir_path}' is relative to project root {PROJECT_ROOT}. Using it cautiously.")
                                log_dir_to_ensure = log_dir_path
                                found_file_handler = True
                                break
                    if not found_file_handler:
                        print(
                            f"No file handler with 'filename' found in logging config. Using default log directory: {DEFAULT_LOG_DIR}")

        except Exception as e:
            print(
                f"Warning: Error reading logging config for directory check ({config_path}): {e}. Using default: {DEFAULT_LOG_DIR}")

    # --- 로그 디렉토리 생성 ---
    try:
        log_dir_to_ensure.mkdir(parents=True, exist_ok=True)
        print(f"Log directory ensured: {log_dir_to_ensure}")
    except OSError as e:
        print(f"Warning: Failed to create log directory {log_dir_to_ensure}: {e}")
        # 디렉토리 생성 실패 시 파일 핸들러가 오류를 발생시킬 수 있음

    # --- 로깅 설정 적용 ---
    if config_path.exists():
        try:
            with open(config_path, 'rt', encoding='utf-8') as f:
                config = yaml.safe_load(f.read())
                if config and isinstance(config, dict):  # dict 타입인지 명시적 확인
                    logging.config.dictConfig(config)
                    log_config_loaded = True
                    print(f"Logging configuration loaded from: {config_path}")
                else:
                    print(f"Logging config file is empty or not a dictionary: {config_path}. Using basic config.")
                    logging.basicConfig(level=default_level,
                                        format="[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
                                        stream=sys.stdout)
        except Exception as e:
            print(f"Error loading logging config file ({config_path}): {e}", file=sys.stderr)  # 오류는 stderr로
            print("Falling back to basic logging.", file=sys.stderr)
            logging.basicConfig(level=default_level, format="[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
                                stream=sys.stdout)
    else:
        print(f"Logging config file not found: {config_path}. Using basic logging.")
        logging.basicConfig(level=default_level, format="[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
                            stream=sys.stdout)

    # --- 컨텍스트 필터를 모든 핸들러에 추가 ---
    context_filter = ContextFilter()
    loggers_to_check = [logging.getLogger(name) for name in logging.root.manager.loggerDict if
                        not logging.getLogger(name).propagate]
    loggers_to_check.append(logging.root)  # 루트 로거와 propagate=False인 로거들만 직접 체크

    applied_handlers = set()
    for logger_instance in loggers_to_check:
        for handler in logger_instance.handlers:
            if handler not in applied_handlers:
                if not any(isinstance(f, ContextFilter) for f in handler.filters):
                    try:
                        handler.addFilter(context_filter)
                    except Exception as e:
                        print(
                            f"Warning: Could not add ContextFilter to handler {handler} for logger {logger_instance.name}: {e}")
                applied_handlers.add(handler)

    # 루트 핸들러에도 적용 (propagate=True인 로거들이 사용할 수 있도록)
    if logging.root.handlers:
        for handler in logging.root.handlers:
            if handler not in applied_handlers:
                if not any(isinstance(f, ContextFilter) for f in handler.filters):
                    try:
                        handler.addFilter(context_filter)
                    except Exception as e:
                        print(f"Warning: Could not add ContextFilter to root handler {handler}: {e}")
                applied_handlers.add(handler)

    if applied_handlers:
        print(f"ContextFilter potentially applied to {len(applied_handlers)} unique handlers.")
    else:
        print("No handlers found to apply ContextFilter (or configuration/basic logging did not set handlers).")


# --- 로거 인스턴스 반환 함수 ---
_loggers = {}  # 생성된 로거 캐시 (모듈 레벨)


def get_logger(name: str) -> logging.Logger:
    """이름별로 로거 인스턴스를 반환합니다. (캐싱 사용)"""
    global _loggers
    if name not in _loggers:
        logger_instance = logging.getLogger(name)
        # 기본 로거 생성 시 레벨 설정 (설정 파일 없을 때 대비)
        if not logger_instance.hasHandlers() and not logging.root.hasHandlers():
            # basicConfig가 호출되지 않았을 수 있는 극단적 경우 방지
            # 또는 로거별 기본 레벨 설정 등 추가 가능
            pass
        _loggers[name] = logger_instance

    return _loggers[name]


# --- 로깅용 데이터 요약 헬퍼 함수 ---
# 이전 단계에서 개선된 버전 사용
def _safe_serialize_value(value: Any, max_len: int) -> Any:
    """
    로깅을 위해 값을 안전하게 직렬화합니다. 복잡한 타입은 요약하고, 문자열은 자릅니다.
    JSON으로 직접 인코딩 가능한 기본 타입(int, float, bool, None)은 그대로 반환합니다.
    """
    if isinstance(value, dict):
        keys_preview = list(value.keys())[:3]
        keys_str = ", ".join(map(str, keys_preview))
        if len(value) > 3: keys_str += "..."
        return f"{{Dict len={len(value)}, keys=[{keys_str}]}}"
    elif isinstance(value, list):
        items_preview = ""
        if value:
            try:
                first_item_str = str(value[0])  # 재귀 호출 대신 str 사용
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
        return value  # 기본 타입은 그대로 반환
    else:
        try:
            s_val = str(value)
            type_name = type(value).__name__
            if len(s_val) > max_len:
                return f"<{type_name} '{s_val[:max_len]}...'>"
            return f"<{type_name} '{s_val}'>"
        except Exception as e:
            return f"<{type(value).__name__} object (str error: {e})>"


# --- 메인 로깅 요약 함수 (수정됨) ---
def summarize_for_logging(
        data: Any,
        max_len: int = 100,
        fields_to_show: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None  # 특정 키 제외 기능 추가
) -> str:
    """
    State 객체(Pydantic 모델 포함) 또는 딕셔너리를 로깅용 JSON 문자열로 요약합니다.
    (업그레이드됨: _safe_serialize_value 헬퍼 사용, exclude_keys 추가)
    """
    summary = {}
    target_dict: Optional[Dict] = None

    try:
        if hasattr(data, 'model_dump'):  # Pydantic V2+
            # Pydantic 모델의 경우, exclude_none=False로 하여 모든 필드를 가져오고
            # _safe_serialize_value에서 None을 처리하는 것이 더 일관적일 수 있음
            target_dict = data.model_dump(exclude_none=False)  # None 포함하여 가져오기
        elif isinstance(data, dict):
            target_dict = data.copy()
        else:
            # 딕셔너리나 Pydantic 모델 아니면 단순 요약 시도
            # 이 경우 str()을 직접 호출하는 것이 더 안전할 수 있음
            return str(_safe_serialize_value(data, max_len))

        if target_dict is None:
            return "{}"

        keys_to_exclude = set(exclude_keys) if exclude_keys else set()

        if fields_to_show is not None:
            keys_to_process = [k for k in fields_to_show if k in target_dict and k not in keys_to_exclude]
        else:
            keys_to_process = [k for k in target_dict.keys() if k not in keys_to_exclude]

        for field_name in keys_to_process:
            value = target_dict[field_name]
            summary[field_name] = _safe_serialize_value(value, max_len)  # None 값도 처리됨

        # JSON 변환 (한 줄 요약, 한글 유지)
        # default=str 추가하여 직렬화 불가능한 객체 발생 시 문자열 변환 시도
        return json.dumps(summary, indent=None, ensure_ascii=False, default=str)

    except Exception as e:
        # 요약 중 오류 발생 시 안전한 반환 (오류 정보 포함)
        try:
            # 오류 발생 시에도 최소한의 정보라도 로깅 시도
            fallback_summary = {"logging_summary_error": str(e)}
            if isinstance(data, dict):
                fallback_summary["data_keys"] = list(data.keys())
            elif hasattr(data, '__dict__'):
                fallback_summary["data_keys"] = list(data.__dict__.keys())
            return json.dumps(fallback_summary, default=str)
        except Exception:  # 이중 오류 방지
            return '{ "logging_summary_error": "Failed to serialize summary error." }'