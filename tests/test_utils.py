# tests/test_utils.py

import pytest
import logging
from unittest.mock import patch, MagicMock
from pathlib import Path

from app.utils.logger import ContextFilter


class TestContextFilter:
    """ContextFilter 클래스 테스트"""

    @pytest.fixture
    def context_filter(self):
        """ContextFilter 인스턴스"""
        return ContextFilter()

    @pytest.fixture
    def mock_log_record(self):
        """Mock LogRecord 객체"""
        record = MagicMock()
        # 기본적으로 속성들이 없는 상태로 설정
        delattr(record, 'trace_id') if hasattr(record, 'trace_id') else None
        delattr(record, 'node_name') if hasattr(record, 'node_name') else None
        delattr(record, 'retry_count') if hasattr(record, 'retry_count') else None
        return record

    def test_filter_adds_default_values(self, context_filter, mock_log_record):
        """필터가 기본값을 추가하는지 테스트"""
        # 속성이 없는 상태에서 필터 적용
        result = context_filter.filter(mock_log_record)
        
        # 필터는 항상 True를 반환해야 함
        assert result is True
        
        # 기본값들이 설정되었는지 확인
        assert mock_log_record.trace_id == 'N/A'
        assert mock_log_record.node_name == 'N/A'
        assert mock_log_record.retry_count == 0

    def test_filter_preserves_existing_values(self, context_filter, mock_log_record):
        """필터가 기존 값을 보존하는지 테스트"""
        # 기존 값들 설정
        mock_log_record.trace_id = 'existing-trace-id'
        mock_log_record.node_name = 'existing-node'
        mock_log_record.retry_count = 5
        
        result = context_filter.filter(mock_log_record)
        
        assert result is True
        # 기존 값들이 보존되어야 함
        assert mock_log_record.trace_id == 'existing-trace-id'
        assert mock_log_record.node_name == 'existing-node'
        assert mock_log_record.retry_count == 5

    def test_filter_partial_existing_values(self, context_filter, mock_log_record):
        """일부 속성만 존재하는 경우 테스트"""
        # trace_id만 설정
        mock_log_record.trace_id = 'partial-trace-id'
        
        result = context_filter.filter(mock_log_record)
        
        assert result is True
        # 기존 값은 보존, 없는 값은 기본값
        assert mock_log_record.trace_id == 'partial-trace-id'
        assert mock_log_record.node_name == 'N/A'
        assert mock_log_record.retry_count == 0

    def test_filter_retry_count_default_type(self, context_filter, mock_log_record):
        """retry_count 기본값이 정수형인지 테스트"""
        result = context_filter.filter(mock_log_record)
        
        assert result is True
        assert isinstance(mock_log_record.retry_count, int)
        assert mock_log_record.retry_count == 0

    def test_filter_with_none_values(self, context_filter, mock_log_record):
        """None 값이 설정된 속성들에 대한 테스트"""
        mock_log_record.trace_id = None
        mock_log_record.node_name = None
        mock_log_record.retry_count = None
        
        result = context_filter.filter(mock_log_record)
        
        assert result is True
        # getattr의 기본값이 적용되어야 함
        assert mock_log_record.trace_id == 'N/A'
        assert mock_log_record.node_name == 'N/A'
        assert mock_log_record.retry_count == 0

    def test_filter_with_empty_string_values(self, context_filter, mock_log_record):
        """빈 문자열 값에 대한 테스트"""
        mock_log_record.trace_id = ''
        mock_log_record.node_name = ''
        mock_log_record.retry_count = 0
        
        result = context_filter.filter(mock_log_record)
        
        assert result is True
        # 빈 문자열과 0은 유효한 값으로 보존되어야 함
        assert mock_log_record.trace_id == ''
        assert mock_log_record.node_name == ''
        assert mock_log_record.retry_count == 0


class TestLoggerConstants:
    """Logger 모듈의 상수 및 설정 테스트"""

    def test_project_root_calculation(self):
        """PROJECT_ROOT 경로 계산 테스트"""
        from app.utils.logger import PROJECT_ROOT
        
        # PROJECT_ROOT가 Path 객체인지 확인
        assert isinstance(PROJECT_ROOT, Path)
        
        # 절대 경로인지 확인
        assert PROJECT_ROOT.is_absolute()
        
        # 경로가 존재하는지 확인 (프로젝트 구조상 존재해야 함)
        # 실제 파일시스템에 의존하므로 존재 여부는 환경에 따라 다를 수 있음
        # assert PROJECT_ROOT.exists()  # 주석 처리

    def test_default_logging_config_path(self):
        """DEFAULT_LOGGING_CONFIG_PATH 설정 테스트"""
        from app.utils.logger import DEFAULT_LOGGING_CONFIG_PATH, PROJECT_ROOT
        
        # Path 객체인지 확인
        assert isinstance(DEFAULT_LOGGING_CONFIG_PATH, Path)
        
        # PROJECT_ROOT를 기준으로 한 상대 경로인지 확인
        assert DEFAULT_LOGGING_CONFIG_PATH.parent == PROJECT_ROOT
        
        # 파일명이 올바른지 확인
        assert DEFAULT_LOGGING_CONFIG_PATH.name == 'logging_config.yaml'

    def test_default_log_dir_path(self):
        """DEFAULT_LOG_DIR 설정 테스트"""
        from app.utils.logger import DEFAULT_LOG_DIR, PROJECT_ROOT
        
        # Path 객체인지 확인
        assert isinstance(DEFAULT_LOG_DIR, Path)
        
        # PROJECT_ROOT 하위의 app/log 경로인지 확인
        expected_path = PROJECT_ROOT / 'app' / 'log'
        assert DEFAULT_LOG_DIR == expected_path


class TestLoggerModuleImports:
    """Logger 모듈 임포트 테스트"""

    def test_logger_module_imports(self):
        """필요한 모듈들이 올바르게 임포트되는지 테스트"""
        try:
            import logging
            import logging.config
            import yaml
            from pathlib import Path
            import sys
            import json
            from typing import Any, Dict, List, Optional
            
            # 모든 임포트가 성공하면 테스트 통과
            assert True
        except ImportError as e:
            pytest.fail(f"필수 모듈 임포트 실패: {e}")

    def test_context_filter_import(self):
        """ContextFilter 클래스 임포트 테스트"""
        from app.utils.logger import ContextFilter
        
        # 클래스인지 확인
        assert isinstance(ContextFilter, type)
        
        # logging.Filter를 상속받는지 확인
        assert issubclass(ContextFilter, logging.Filter)

    def test_context_filter_has_filter_method(self):
        """ContextFilter가 filter 메서드를 가지고 있는지 테스트"""
        from app.utils.logger import ContextFilter
        
        filter_instance = ContextFilter()
        
        # filter 메서드가 존재하는지 확인
        assert hasattr(filter_instance, 'filter')
        assert callable(filter_instance.filter)


class TestLoggerFilterIntegration:
    """ContextFilter와 실제 로깅 시스템 통합 테스트"""

    def test_filter_with_real_log_record(self):
        """실제 LogRecord 객체와 함께 필터 테스트"""
        from app.utils.logger import ContextFilter
        
        # 실제 LogRecord 생성
        logger = logging.getLogger('test_logger')
        record = logging.LogRecord(
            name='test_logger',
            level=logging.INFO,
            pathname='/test/path.py',
            lineno=100,
            msg='Test message',
            args=(),
            exc_info=None
        )
        
        # 필터 적용
        context_filter = ContextFilter()
        result = context_filter.filter(record)
        
        assert result is True
        assert hasattr(record, 'trace_id')
        assert hasattr(record, 'node_name')
        assert hasattr(record, 'retry_count')
        assert record.trace_id == 'N/A'
        assert record.node_name == 'N/A'
        assert record.retry_count == 0

    def test_filter_integration_with_logger(self):
        """Logger와 Filter 통합 테스트"""
        from app.utils.logger import ContextFilter
        
        # 테스트용 로거 생성
        test_logger = logging.getLogger('integration_test_logger')
        test_logger.setLevel(logging.DEBUG)
        
        # 메모리 핸들러 추가
        from io import StringIO
        import sys
        
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.DEBUG)
        
        # 필터 추가
        context_filter = ContextFilter()
        handler.addFilter(context_filter)
        
        # 포매터 설정 (컨텍스트 필드 포함)
        formatter = logging.Formatter(
            '%(levelname)s - %(trace_id)s - %(node_name)s - %(retry_count)d - %(message)s'
        )
        handler.setFormatter(formatter)
        
        test_logger.addHandler(handler)
        
        # 로그 메시지 출력
        test_logger.info('Test integration message')
        
        # 출력 결과 확인
        log_output = log_stream.getvalue()
        assert 'INFO' in log_output
        assert 'N/A' in log_output  # trace_id, node_name 기본값
        assert '0' in log_output     # retry_count 기본값
        assert 'Test integration message' in log_output
        
        # 핸들러 정리
        test_logger.removeHandler(handler)
        handler.close()


# 테스트 실행을 위한 pytest 설정
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
