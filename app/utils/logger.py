# app/utils/logger.py

import logging
import logging.config
import yaml
import os

def setup_logging(config_path='logging_config.yaml', default_level=logging.INFO):
    """
    설정 파일(YAML)을 읽어 로깅 시스템을 설정합니다.
    파일이 없으면 기본 설정을 적용합니다.
    """
    path = config_path
    if os.path.exists(path):
        try:
            with open(path, 'rt', encoding='utf-8') as f:
                config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)
            print(f"Logging configured successfully from {path}")
        except Exception as e:
            print(f"Error loading logging configuration from {path}: {e}")
            print("Using basic logging configuration.")
            logging.basicConfig(level=default_level, format="[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s")
    else:
        print(f"Logging configuration file not found at {path}. Using basic logging configuration.")
        logging.basicConfig(level=default_level, format="[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s")

# 애플리케이션 시작 시 로깅 설정 로드 (예: 메인 스크립트나 앱 초기화 부분)
# setup_logging() # 이 줄은 실제 앱 초기화 시점에 호출되어야 합니다.

# get_logger 함수는 이제 단순히 로거 인스턴스를 반환하는 역할만 합니다.
# 설정은 setup_logging에서 한 번만 수행됩니다.
def get_logger(name: str) -> logging.Logger:
    """
    이름별로 로거 인스턴스를 반환합니다. 설정은 setup_logging을 통해 적용됩니다.
    """
    return logging.getLogger(name)

# 기본 글로벌 로거 (설정 파일에 'newsom' 로거 정의가 있다면 해당 설정을 따름)
# logger = get_logger("newsom")
# 참고: 전역 로거 대신 필요한 곳에서 get_logger를 호출하는 것이 더 권장될 수 있습니다.
# 또는 앱 컨텍스트나 의존성 주입을 통해 로거를 전달하는 방식을 고려할 수 있습니다.

# --- 사용 예시 ---
# 앱 초기화 시 (예: main.py)
# from app.utils.logger import setup_logging, get_logger
# setup_logging('path/to/your/logging_config.yaml') # 설정 로드
#
# logger = get_logger("newsom") # newsom 로거 가져오기
# logger.info("Newsom logger initialized.")
#
# other_module_logger = get_logger("other_module") # 다른 모듈 로거 가져오기 (root 설정 따름)
# other_module_logger.debug("This debug message might not show depending on root level.")
# other_module_logger.warning("This warning message should show.")