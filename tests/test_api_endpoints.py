# tests/test_api_endpoints.py

import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException, BackgroundTasks
from fastapi.testclient import TestClient
import json

from app.api.v1.endpoints import request_comic_generation, get_comic_status
from app.api.v1.schemas import AsyncComicRequest, RequestDataPayload, SitePreferencesPayload


class TestRequestComicGeneration:
    """POST /comics 엔드포인트 테스트"""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock 의존성들 설정"""
        compiled_app = AsyncMock()
        db_client = AsyncMock()
        background_tasks = BackgroundTasks()
        return compiled_app, db_client, background_tasks

    @pytest.fixture
    def sample_request_data(self):
        """테스트용 요청 데이터"""
        return AsyncComicRequest(
            writer_id="test_writer",
            data=RequestDataPayload(
                query="테스트 쿼리",
                site=SitePreferencesPayload(
                    code_related=["github.com", "stackoverflow.com"],
                    news=["news.naver.com"]
                )
            )
        )

    @pytest.mark.asyncio
    async def test_request_comic_generation_success(self, mock_dependencies, sample_request_data):
        """정상적인 만화 생성 요청 테스트"""
        compiled_app, db_client, background_tasks = mock_dependencies
        
        with patch('app.api.v1.endpoints.trigger_workflow_task') as mock_trigger:
            # Mock 설정
            expected_comic_id = str(uuid.uuid4())
            mock_trigger.return_value = expected_comic_id
            
            # API 호출
            result = await request_comic_generation(
                compiled_app=compiled_app,
                db_client=db_client,
                request_data=sample_request_data,
                background_tasks=background_tasks
            )
            
            # 검증
            assert result.comic_id == expected_comic_id
            assert result.status == "PENDING"
            assert "백그라운드에서 시작되었습니다" in result.message
            
            # trigger_workflow_task 호출 검증
            mock_trigger.assert_called_once()
            call_args = mock_trigger.call_args
            assert call_args[1]['query'] == "테스트 쿼리"
            assert call_args[1]['config']['writer_id'] == "test_writer"
            assert 'user_site_preferences' in call_args[1]['config']

    @pytest.mark.asyncio
    async def test_request_comic_generation_no_site_preferences(self, mock_dependencies):
        """사이트 설정 없는 요청 테스트"""
        compiled_app, db_client, background_tasks = mock_dependencies
        
        request_data = AsyncComicRequest(
            writer_id=None,
            data=RequestDataPayload(query="테스트 쿼리")
        )
        
        with patch('app.api.v1.endpoints.trigger_workflow_task') as mock_trigger:
            expected_comic_id = str(uuid.uuid4())
            mock_trigger.return_value = expected_comic_id
            
            result = await request_comic_generation(
                compiled_app=compiled_app,
                db_client=db_client,
                request_data=request_data,
                background_tasks=background_tasks
            )
            
            # 검증
            assert result.comic_id == expected_comic_id
            call_args = mock_trigger.call_args
            assert call_args[1]['config']['writer_id'] is None
            assert 'user_site_preferences' not in call_args[1]['config']

    @pytest.mark.asyncio
    async def test_request_comic_generation_http_exception(self, mock_dependencies, sample_request_data):
        """HTTP 예외 발생 테스트"""
        compiled_app, db_client, background_tasks = mock_dependencies
        
        with patch('app.api.v1.endpoints.trigger_workflow_task') as mock_trigger:
            # HTTP 예외 발생 설정
            mock_trigger.side_effect = HTTPException(status_code=400, detail="테스트 오류")
            
            # 예외 발생 검증
            with pytest.raises(HTTPException) as exc_info:
                await request_comic_generation(
                    compiled_app=compiled_app,
                    db_client=db_client,
                    request_data=sample_request_data,
                    background_tasks=background_tasks
                )
            
            assert exc_info.value.status_code == 400
            assert exc_info.value.detail == "테스트 오류"

    @pytest.mark.asyncio
    async def test_request_comic_generation_generic_exception(self, mock_dependencies, sample_request_data):
        """일반 예외 발생 테스트"""
        compiled_app, db_client, background_tasks = mock_dependencies
        
        with patch('app.api.v1.endpoints.trigger_workflow_task') as mock_trigger:
            # 일반 예외 발생 설정
            mock_trigger.side_effect = Exception("알 수 없는 오류")
            
            # 500 에러로 변환되는지 검증
            with pytest.raises(HTTPException) as exc_info:
                await request_comic_generation(
                    compiled_app=compiled_app,
                    db_client=db_client,
                    request_data=sample_request_data,
                    background_tasks=background_tasks
                )
            
            assert exc_info.value.status_code == 500
            assert "내부 서버 오류" in exc_info.value.detail


class TestGetComicStatus:
    """GET /comics/status/{comic_id} 엔드포인트 테스트"""

    @pytest.fixture
    def mock_db_client(self):
        """Mock DB 클라이언트"""
        return AsyncMock()

    @pytest.fixture
    def sample_comic_id(self):
        """테스트용 comic_id"""
        return str(uuid.uuid4())

    @pytest.fixture
    def sample_status_data(self, sample_comic_id):
        """테스트용 상태 데이터"""
        return {
            "comic_id": sample_comic_id,
            "status": "DONE",
            "message": "성공",
            "query": "테스트 쿼리",
            "writer_id": "test_writer",
            "user_site_preferences_provided": True,
            "timestamp_accepted": "2025-05-23T10:00:00Z",
            "timestamp_start": "2025-05-23T10:00:05Z",
            "timestamp_end": "2025-05-23T10:02:00Z",
            "duration_seconds": 115.5,
            "result": {"images_cnt": 4, "ideas_cnt": 3},
            "error_details": None
        }

    @pytest.mark.asyncio
    async def test_get_comic_status_success_dict(self, mock_db_client, sample_comic_id, sample_status_data):
        """딕셔너리 형태 상태 데이터 조회 성공 테스트"""
        # Mock 설정 - dict 반환
        mock_db_client.get.return_value = sample_status_data
        
        result = await get_comic_status(
            db_client=mock_db_client,
            comic_id=sample_comic_id
        )
        
        # 검증
        assert result.comic_id == sample_comic_id
        assert result.status == "DONE"
        assert result.message == "성공"
        assert result.duration_seconds == 115.5
        mock_db_client.get.assert_called_once_with(sample_comic_id)

    @pytest.mark.asyncio
    async def test_get_comic_status_success_json_string(self, mock_db_client, sample_comic_id, sample_status_data):
        """JSON 문자열 형태 상태 데이터 조회 성공 테스트"""
        # Mock 설정 - JSON 문자열 반환
        mock_db_client.get.return_value = json.dumps(sample_status_data)
        
        result = await get_comic_status(
            db_client=mock_db_client,
            comic_id=sample_comic_id
        )
        
        # 검증
        assert result.comic_id == sample_comic_id
        assert result.status == "DONE"

    @pytest.mark.asyncio
    async def test_get_comic_status_not_found(self, mock_db_client, sample_comic_id):
        """상태 데이터 없음 (404) 테스트"""
        # Mock 설정 - None 반환
        mock_db_client.get.return_value = None
        
        with pytest.raises(HTTPException) as exc_info:
            await get_comic_status(
                db_client=mock_db_client,
                comic_id=sample_comic_id
            )
        
        assert exc_info.value.status_code == 404
        assert "찾을 수 없습니다" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_get_comic_status_invalid_json(self, mock_db_client, sample_comic_id):
        """잘못된 JSON 형식 테스트"""
        # Mock 설정 - 잘못된 JSON 반환
        mock_db_client.get.return_value = "invalid json {"
        
        with pytest.raises(HTTPException) as exc_info:
            await get_comic_status(
                db_client=mock_db_client,
                comic_id=sample_comic_id
            )
        
        assert exc_info.value.status_code == 500
        assert "형식 오류" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_get_comic_status_unsupported_data_type(self, mock_db_client, sample_comic_id):
        """지원하지 않는 데이터 타입 테스트"""
        # Mock 설정 - 지원하지 않는 타입 반환
        mock_db_client.get.return_value = 123  # int 타입
        
        with pytest.raises(HTTPException) as exc_info:
            await get_comic_status(
                db_client=mock_db_client,
                comic_id=sample_comic_id
            )
        
        assert exc_info.value.status_code == 500
        assert "형식 오류" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_get_comic_status_pydantic_validation_error(self, mock_db_client, sample_comic_id):
        """Pydantic 유효성 검사 오류 테스트"""
        # Mock 설정 - 필수 필드 누락된 데이터
        invalid_data = {"invalid_field": "value"}
        mock_db_client.get.return_value = invalid_data
        
        with pytest.raises(HTTPException) as exc_info:
            await get_comic_status(
                db_client=mock_db_client,
                comic_id=sample_comic_id
            )
        
        assert exc_info.value.status_code == 500
        assert "처리 중 오류" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_get_comic_status_db_exception(self, mock_db_client, sample_comic_id):
        """DB 연결 오류 테스트"""
        # Mock 설정 - DB 예외 발생
        mock_db_client.get.side_effect = Exception("DB 연결 실패")
        
        with pytest.raises(HTTPException) as exc_info:
            await get_comic_status(
                db_client=mock_db_client,
                comic_id=sample_comic_id
            )
        
        assert exc_info.value.status_code == 500
        assert "내부 서버 오류" in exc_info.value.detail


# 테스트 실행을 위한 pytest 설정
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
