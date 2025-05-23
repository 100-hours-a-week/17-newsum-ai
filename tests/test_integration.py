# tests/test_integration.py

import pytest
import json
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI
import uuid

from app.api.v1.endpoints import router as v1_router
from app.api.v1.schemas import AsyncComicRequest, RequestDataPayload, SitePreferencesPayload


@pytest.mark.integration
class TestAPIIntegration:
    """API 엔드포인트 통합 테스트"""

    @pytest.fixture
    def test_app(self):
        """테스트용 FastAPI 앱"""
        app = FastAPI()
        app.include_router(v1_router, prefix="/api/v1")
        return app

    @pytest.fixture
    def client(self, test_app):
        """테스트 클라이언트"""
        return TestClient(test_app)

    @pytest.fixture
    def mock_dependencies_patch(self, mock_compiled_app, mock_db_client):
        """의존성 패치"""
        with patch('app.api.v1.endpoints.CompiledWorkflowDep', return_value=mock_compiled_app), \
             patch('app.api.v1.endpoints.DatabaseClientDep', return_value=mock_db_client):
            yield mock_compiled_app, mock_db_client

    def test_post_comics_endpoint_success(self, client, mock_dependencies_patch):
        """POST /api/v1/comics 성공 테스트"""
        mock_compiled_app, mock_db_client = mock_dependencies_patch
        
        # 요청 데이터
        request_data = {
            "writer_id": "test_writer",
            "data": {
                "query": "AI 뉴스 검색",
                "site": {
                    "code_related": ["github.com"],
                    "news": ["news.naver.com"]
                }
            }
        }
        
        with patch('app.api.v1.endpoints.trigger_workflow_task') as mock_trigger:
            expected_comic_id = str(uuid.uuid4())
            mock_trigger.return_value = expected_comic_id
            
            # API 호출
            response = client.post("/api/v1/comics", json=request_data)
            
            # 응답 검증
            assert response.status_code == 202
            response_data = response.json()
            assert response_data["comic_id"] == expected_comic_id
            assert response_data["status"] == "PENDING"
            assert "백그라운드에서 시작되었습니다" in response_data["message"]

    def test_post_comics_endpoint_validation_error(self, client, mock_dependencies_patch):
        """POST /api/v1/comics 유효성 검사 오류 테스트"""
        # 잘못된 요청 데이터 (query 누락)
        request_data = {
            "writer_id": "test_writer",
            "data": {
                "site": {
                    "news": ["news.naver.com"]
                }
            }
        }
        
        # API 호출
        response = client.post("/api/v1/comics", json=request_data)
        
        # 422 Unprocessable Entity 응답 확인
        assert response.status_code == 422
        response_data = response.json()
        assert "detail" in response_data

    def test_get_comics_status_success(self, client, mock_dependencies_patch, sample_status_data):
        """GET /api/v1/comics/status/{comic_id} 성공 테스트"""
        mock_compiled_app, mock_db_client = mock_dependencies_patch
        
        comic_id = sample_status_data["comic_id"]
        mock_db_client.get.return_value = sample_status_data
        
        # API 호출
        response = client.get(f"/api/v1/comics/status/{comic_id}")
        
        # 응답 검증
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["comic_id"] == comic_id
        assert response_data["status"] == "DONE"
        assert response_data["duration_seconds"] == 145.0

    def test_get_comics_status_not_found(self, client, mock_dependencies_patch):
        """GET /api/v1/comics/status/{comic_id} 404 테스트"""
        mock_compiled_app, mock_db_client = mock_dependencies_patch
        
        # 존재하지 않는 comic_id
        comic_id = str(uuid.uuid4())
        mock_db_client.get.return_value = None
        
        # API 호출
        response = client.get(f"/api/v1/comics/status/{comic_id}")
        
        # 404 응답 확인
        assert response.status_code == 404
        response_data = response.json()
        assert "찾을 수 없습니다" in response_data["detail"]


@pytest.mark.integration
class TestWorkflowIntegration:
    """워크플로우 통합 테스트"""

    @pytest.mark.asyncio
    async def test_complete_workflow_success(self, mock_compiled_app, mock_db_client, sample_config):
        """완전한 워크플로우 성공 시나리오 테스트"""
        from app.api.v1.background_tasks import trigger_workflow_task
        from fastapi import BackgroundTasks
        
        background_tasks = MagicMock()
        
        # workflow 트리거
        comic_id = await trigger_workflow_task(
            query="통합 테스트 쿼리",
            config=sample_config,
            background_tasks=background_tasks,
            compiled_app=mock_compiled_app,
            db_client=mock_db_client
        )
        
        # 결과 검증
        assert isinstance(comic_id, str)
        assert len(comic_id) == 36  # UUID 길이
        
        # DB 호출 검증
        mock_db_client.set.assert_called()
        
        # 백그라운드 태스크 등록 검증
        background_tasks.add_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_workflow_with_error_handling(self, mock_compiled_app, mock_db_client):
        """워크플로우 오류 처리 통합 테스트"""
        from app.api.v1.background_tasks import trigger_workflow_task
        from fastapi import BackgroundTasks
        
        # DB 오류 시뮬레이션
        mock_db_client.set.side_effect = Exception("DB 연결 실패")
        
        background_tasks = MagicMock()
        
        # 예외 발생 확인
        with pytest.raises(Exception) as exc_info:
            await trigger_workflow_task(
                query="오류 테스트 쿼리",
                config={},
                background_tasks=background_tasks,
                compiled_app=mock_compiled_app,
                db_client=mock_db_client
            )
        
        assert "DB 연결 실패" in str(exc_info.value)


@pytest.mark.integration
class TestSchemaIntegration:
    """스키마 통합 테스트"""

    def test_request_response_schema_compatibility(self):
        """요청-응답 스키마 호환성 테스트"""
        # 복잡한 요청 데이터 생성
        request_data = AsyncComicRequest(
            writer_id="integration_writer",
            data=RequestDataPayload(
                query="통합 테스트를 위한 복잡한 쿼리",
                site=SitePreferencesPayload(
                    code_related=["github.com", "gitlab.com"],
                    research_paper=["arxiv.org", "scholar.google.com"],
                    deep_dive_tech=["medium.com", "dev.to"],
                    community=["reddit.com", "hacker-news.com"],
                    news=["techcrunch.com", "venturebeat.com"]
                )
            )
        )
        
        # 직렬화/역직렬화 테스트
        json_data = request_data.model_dump()
        reconstructed = AsyncComicRequest(**json_data)
        
        # 데이터 일치 확인
        assert reconstructed.writer_id == request_data.writer_id
        assert reconstructed.data.query == request_data.data.query
        assert reconstructed.data.site.code_related == request_data.data.site.code_related

    def test_status_response_schema_flexibility(self, sample_status_data):
        """상태 응답 스키마 유연성 테스트"""
        from app.api.v1.schemas import ComicStatusResponse
        
        # 기본 데이터로 스키마 생성
        status_response = ComicStatusResponse(**sample_status_data)
        
        # 일부 선택적 필드 제거
        minimal_data = {
            "comic_id": "minimal-test-id",
            "status": "PENDING",
            "message": "처리 중"
        }
        
        minimal_response = ComicStatusResponse(**minimal_data)
        
        # 선택적 필드들이 None으로 설정되었는지 확인
        assert minimal_response.result is None
        assert minimal_response.error_details is None
        assert minimal_response.duration_seconds is None


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndIntegration:
    """종단간 통합 테스트"""

    @pytest.fixture
    def integrated_app(self):
        """완전히 통합된 앱 (실제 의존성들은 mock)"""
        app = FastAPI()
        app.include_router(v1_router, prefix="/api/v1")
        return app

    def test_complete_comic_generation_flow(self, integrated_app, mock_compiled_app, mock_db_client):
        """완전한 만화 생성 플로우 테스트"""
        client = TestClient(integrated_app)
        
        with patch('app.api.v1.endpoints.CompiledWorkflowDep', return_value=mock_compiled_app), \
             patch('app.api.v1.endpoints.DatabaseClientDep', return_value=mock_db_client), \
             patch('app.api.v1.endpoints.trigger_workflow_task') as mock_trigger:
            
            test_comic_id = str(uuid.uuid4())
            mock_trigger.return_value = test_comic_id
            
            # 1단계: 만화 생성 요청
            request_data = {
                "writer_id": "e2e_writer",
                "data": {
                    "query": "종단간 테스트 쿼리",
                    "site": {
                        "code_related": ["github.com"],
                        "news": ["news.ycombinator.com"]
                    }
                }
            }
            
            response = client.post("/api/v1/comics", json=request_data)
            assert response.status_code == 202
            
            response_data = response.json()
            comic_id = response_data["comic_id"]
            assert comic_id == test_comic_id
            
            # 2단계: 상태 조회 (PENDING)
            mock_db_client.get.return_value = {
                "comic_id": comic_id,
                "status": "PENDING",
                "message": "처리 중",
                "query": "종단간 테스트 쿼리",
                "writer_id": "e2e_writer",
                "user_site_preferences_provided": True,
                "timestamp_accepted": "2025-05-23T12:00:00Z",
                "timestamp_start": None,
                "timestamp_end": None,
                "duration_seconds": None,
                "result": None,
                "error_details": None
            }
            
            status_response = client.get(f"/api/v1/comics/status/{comic_id}")
            assert status_response.status_code == 200
            
            status_data = status_response.json()
            assert status_data["status"] == "PENDING"
            assert status_data["comic_id"] == comic_id
            
            # 3단계: 완료된 상태 조회 시뮬레이션
            mock_db_client.get.return_value = {
                "comic_id": comic_id,
                "status": "DONE",
                "message": "성공적으로 완료됨",
                "query": "종단간 테스트 쿼리",
                "writer_id": "e2e_writer",
                "user_site_preferences_provided": True,
                "timestamp_accepted": "2025-05-23T12:00:00Z",
                "timestamp_start": "2025-05-23T12:00:05Z",
                "timestamp_end": "2025-05-23T12:02:30Z",
                "duration_seconds": 145.0,
                "result": {
                    "trace_id": comic_id,
                    "comic_id": comic_id,
                    "final_stage": "END",
                    "images_cnt": 4,
                    "ideas_cnt": 3,
                    "scenarios_cnt": 1
                },
                "error_details": None
            }
            
            final_status_response = client.get(f"/api/v1/comics/status/{comic_id}")
            assert final_status_response.status_code == 200
            
            final_status_data = final_status_response.json()
            assert final_status_data["status"] == "DONE"
            assert final_status_data["duration_seconds"] == 145.0
            assert final_status_data["result"]["images_cnt"] == 4

    def test_error_flow_integration(self, integrated_app, mock_compiled_app, mock_db_client):
        """오류 플로우 통합 테스트"""
        client = TestClient(integrated_app)
        
        with patch('app.api.v1.endpoints.CompiledWorkflowDep', return_value=mock_compiled_app), \
             patch('app.api.v1.endpoints.DatabaseClientDep', return_value=mock_db_client):
            
            # 잘못된 요청으로 400 오류 테스트
            invalid_request = {
                "writer_id": "error_writer",
                "data": {
                    # query 필드 누락
                    "site": {
                        "news": ["invalid.com"]
                    }
                }
            }
            
            response = client.post("/api/v1/comics", json=invalid_request)
            assert response.status_code == 422  # Validation Error
            
            # 존재하지 않는 상태 조회로 404 오류 테스트
            mock_db_client.get.return_value = None
            
            nonexistent_id = str(uuid.uuid4())
            status_response = client.get(f"/api/v1/comics/status/{nonexistent_id}")
            assert status_response.status_code == 404


# 테스트 실행을 위한 pytest 설정
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
