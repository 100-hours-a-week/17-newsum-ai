# tests/test_background_tasks.py

import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
import json

from app.api.v1.background_tasks import trigger_workflow_task, _g


class TestHelperFunction:
    """_g 헬퍼 함수 테스트"""

    def test_g_single_level_key_exists(self):
        """단일 레벨 키가 존재하는 경우"""
        data = {"key1": "value1"}
        result = _g(data, "key1")
        assert result == "value1"

    def test_g_single_level_key_not_exists(self):
        """단일 레벨 키가 존재하지 않는 경우"""
        data = {"key1": "value1"}
        result = _g(data, "key2")
        assert result is None

    def test_g_single_level_key_with_default(self):
        """단일 레벨 키가 존재하지 않고 기본값이 있는 경우"""
        data = {"key1": "value1"}
        result = _g(data, "key2", default="default_value")
        assert result == "default_value"

    def test_g_nested_keys_exists(self):
        """중첩된 키가 모두 존재하는 경우"""
        data = {
            "level1": {
                "level2": {
                    "level3": "target_value"
                }
            }
        }
        result = _g(data, "level1", "level2", "level3")
        assert result == "target_value"

    def test_g_nested_keys_partial_exists(self):
        """중첩된 키 중 일부만 존재하는 경우"""
        data = {
            "level1": {
                "level2": "value"
            }
        }
        result = _g(data, "level1", "level2", "level3")
        assert result is None

    def test_g_nested_keys_with_default(self):
        """중첩된 키가 존재하지 않고 기본값이 있는 경우"""
        data = {"level1": {}}
        result = _g(data, "level1", "level2", "level3", default="default")
        assert result == "default"

    def test_g_none_value_handling(self):
        """None 값이 중간에 있는 경우"""
        data = {"level1": None}
        result = _g(data, "level1", "level2")
        assert result is None

    def test_g_empty_dict(self):
        """빈 딕셔너리인 경우"""
        data = {}
        result = _g(data, "any_key", default="default")
        assert result == "default"

    def test_g_value_is_none_but_exists(self):
        """키는 존재하지만 값이 None인 경우"""
        data = {"key1": None}
        result = _g(data, "key1", default="default")
        assert result == "default"  # None은 default로 대체됨

    def test_g_value_is_empty_string(self):
        """값이 빈 문자열인 경우"""
        data = {"key1": ""}
        result = _g(data, "key1", default="default")
        assert result == ""  # 빈 문자열은 유효한 값


class TestTriggerWorkflowTask:
    """trigger_workflow_task 함수 테스트"""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock 의존성들 설정"""
        background_tasks = MagicMock()
        compiled_app = AsyncMock()
        db_client = AsyncMock()
        return background_tasks, compiled_app, db_client

    @pytest.fixture
    def sample_config(self):
        """테스트용 설정"""
        return {
            "writer_id": "test_writer",
            "user_site_preferences": {
                "code_related": ["github.com"],
                "news": ["news.naver.com"]
            }
        }

    @pytest.mark.asyncio
    async def test_trigger_workflow_task_basic(self, mock_dependencies, sample_config):
        """기본적인 워크플로우 태스크 트리거 테스트"""
        background_tasks, compiled_app, db_client = mock_dependencies
        
        result = await trigger_workflow_task(
            query="테스트 쿼리",
            config=sample_config,
            background_tasks=background_tasks,
            compiled_app=compiled_app,
            db_client=db_client
        )
        
        # UUID 형식 검증
        assert len(result) == 36  # UUID4 문자열 길이
        assert result.count('-') == 4  # UUID4 하이픈 개수
        
        # DB 초기 상태 설정 호출 검증
        db_client.set.assert_called_once()
        call_args = db_client.set.call_args
        comic_id = call_args[0][0]
        initial_state = call_args[0][1]
        
        assert comic_id == result
        assert initial_state["comic_id"] == result
        assert initial_state["status"] == "PENDING"
        assert initial_state["query"] == "테스트 쿼리"
        assert "timestamp_accepted" in initial_state
        
        # 백그라운드 태스크 추가 검증
        background_tasks.add_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_workflow_task_minimal_config(self, mock_dependencies):
        """최소 설정으로 워크플로우 태스크 트리거 테스트"""
        background_tasks, compiled_app, db_client = mock_dependencies
        
        minimal_config = {}
        
        result = await trigger_workflow_task(
            query="최소 쿼리",
            config=minimal_config,
            background_tasks=background_tasks,
            compiled_app=compiled_app,
            db_client=db_client
        )
        
        # 결과 검증
        assert isinstance(result, str)
        assert len(result) == 36
        
        # DB 호출 검증
        db_client.set.assert_called_once()
        call_args = db_client.set.call_args
        initial_state = call_args[0][1]
        assert initial_state["query"] == "최소 쿼리"

    @pytest.mark.asyncio
    async def test_trigger_workflow_task_db_error(self, mock_dependencies, sample_config):
        """DB 오류 발생 시 테스트"""
        background_tasks, compiled_app, db_client = mock_dependencies
        
        # DB set 오류 설정
        db_client.set.side_effect = Exception("DB 연결 실패")
        
        # 예외 발생 검증
        with pytest.raises(Exception) as exc_info:
            await trigger_workflow_task(
                query="테스트 쿼리",
                config=sample_config,
                background_tasks=background_tasks,
                compiled_app=compiled_app,
                db_client=db_client
            )
        
        assert "DB 연결 실패" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_workflow_runner_success_scenario(self, mock_dependencies, sample_config):
        """workflow_runner 성공 시나리오 테스트 (모킹된 실행)"""
        background_tasks, compiled_app, db_client = mock_dependencies
        
        # 성공적인 워크플로우 출력 모킹
        mock_final_output = {
            "meta": {
                "trace_id": "test-trace-id",
                "comic_id": "test-comic-id",
                "current_stage": "END",
                "error_message": None
            },
            "report": {
                "report_content": "테스트 보고서 내용",
                "saved_report_path": "/path/to/report.txt"
            },
            "idea": {
                "comic_ideas": [
                    {"title": "아이디어 1"},
                    {"title": "아이디어 2"}
                ]
            },
            "scenario": {
                "comic_scenarios": [
                    {"scene_count": 4}
                ]
            },
            "image": {
                "generated_comic_images": [
                    {"scene_id": 1, "path": "/image1.png"},
                    {"scene_id": 2, "path": "/image2.png"}
                ]
            },
            "search": {
                "raw_search_results": [
                    {"title": "검색 결과 1", "url": "http://example1.com"},
                    {"title": "검색 결과 2", "url": "http://example2.com"}
                ]
            }
        }
        
        compiled_app.ainvoke.return_value = mock_final_output
        
        # workflow 실행
        result = await trigger_workflow_task(
            query="성공 테스트 쿼리",
            config=sample_config,
            background_tasks=background_tasks,
            compiled_app=compiled_app,
            db_client=db_client
        )
        
        # 백그라운드 태스크가 추가되었는지 확인
        background_tasks.add_task.assert_called_once()
        
        # 태스크 함수 직접 실행하여 테스트
        task_func = background_tasks.add_task.call_args[0][0]
        task_args = background_tasks.add_task.call_args[0][1:]
        
        # workflow_runner 직접 실행
        await task_func(*task_args)
        
        # compiled_app.ainvoke 호출 검증
        compiled_app.ainvoke.assert_called_once()
        invoke_args = compiled_app.ainvoke.call_args
        input_data = invoke_args[0][0]
        
        assert input_data["query"]["original_query"] == "성공 테스트 쿼리"
        assert input_data["config"]["config"] == sample_config
        assert "comic_id" in input_data["meta"]
        assert "trace_id" in input_data["meta"]
        
        # DB 업데이트 호출 검증 (최소 2번: PENDING + 최종 상태)
        assert db_client.set.call_count >= 2

    @pytest.mark.asyncio
    async def test_workflow_runner_failure_scenario(self, mock_dependencies, sample_config):
        """workflow_runner 실패 시나리오 테스트"""
        background_tasks, compiled_app, db_client = mock_dependencies
        
        # 실패 워크플로우 출력 모킹
        mock_failed_output = {
            "meta": {
                "trace_id": "test-trace-id",
                "comic_id": "test-comic-id",
                "current_stage": "ERROR",
                "error_message": "워크플로우 실행 중 오류 발생"
            }
        }
        
        compiled_app.ainvoke.return_value = mock_failed_output
        
        # workflow 실행
        await trigger_workflow_task(
            query="실패 테스트 쿼리",
            config=sample_config,
            background_tasks=background_tasks,
            compiled_app=compiled_app,
            db_client=db_client
        )
        
        # 백그라운드 태스크 함수 직접 실행
        task_func = background_tasks.add_task.call_args[0][0]
        task_args = background_tasks.add_task.call_args[0][1:]
        
        await task_func(*task_args)
        
        # 최종 DB 업데이트에서 FAILED 상태 확인
        final_call = db_client.set.call_args_list[-1]
        final_state = final_call[0][1]
        
        assert final_state["status"] == "FAILED"
        assert "오류 발생" in final_state["message"]
        assert final_state["error_details"] is not None


# 테스트 실행을 위한 pytest 설정
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
