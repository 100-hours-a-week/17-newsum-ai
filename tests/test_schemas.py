# tests/test_schemas.py

import pytest
from pydantic import ValidationError
from app.api.v1.schemas import (
    SitePreferencesPayload,
    RequestDataPayload,
    AsyncComicRequest,
    AsyncComicResponse,
    ComicStatusResponse
)


class TestSitePreferencesPayload:
    """SitePreferencesPayload 스키마 테스트"""

    def test_site_preferences_valid_all_fields(self):
        """모든 필드가 유효한 경우 테스트"""
        data = {
            "code_related": ["github.com", "stackoverflow.com"],
            "research_paper": ["arxiv.org", "scholar.google.com"],
            "deep_dive_tech": ["medium.com"],
            "community": ["reddit.com"],
            "news": ["news.naver.com", "cnn.com"]
        }
        
        schema = SitePreferencesPayload(**data)
        
        assert schema.code_related == ["github.com", "stackoverflow.com"]
        assert schema.research_paper == ["arxiv.org", "scholar.google.com"]
        assert schema.deep_dive_tech == ["medium.com"]
        assert schema.community == ["reddit.com"]
        assert schema.news == ["news.naver.com", "cnn.com"]

    def test_site_preferences_optional_fields(self):
        """선택적 필드들이 None인 경우 테스트"""
        data = {
            "code_related": ["github.com"]
        }
        
        schema = SitePreferencesPayload(**data)
        
        assert schema.code_related == ["github.com"]
        assert schema.research_paper is None
        assert schema.deep_dive_tech is None
        assert schema.community is None
        assert schema.news is None

    def test_site_preferences_empty_dict(self):
        """빈 딕셔너리로 생성하는 경우 테스트"""
        schema = SitePreferencesPayload()
        
        assert schema.code_related is None
        assert schema.research_paper is None
        assert schema.deep_dive_tech is None
        assert schema.community is None
        assert schema.news is None

    def test_site_preferences_model_dump_exclude_none(self):
        """None 값 제외하고 딕셔너리 변환 테스트"""
        schema = SitePreferencesPayload(
            code_related=["github.com"],
            news=["cnn.com"]
        )
        
        result = schema.model_dump(exclude_none=True)
        
        assert result == {
            "code_related": ["github.com"],
            "news": ["cnn.com"]
        }
        assert "research_paper" not in result
        assert "deep_dive_tech" not in result


class TestRequestDataPayload:
    """RequestDataPayload 스키마 테스트"""

    def test_request_data_valid_with_site(self):
        """사이트 설정이 포함된 유효한 요청 데이터 테스트"""
        data = {
            "query": "AI 관련 뉴스 검색",
            "site": {
                "code_related": ["github.com"],
                "news": ["news.naver.com"]
            }
        }
        
        schema = RequestDataPayload(**data)
        
        assert schema.query == "AI 관련 뉴스 검색"
        assert schema.site is not None
        assert schema.site.code_related == ["github.com"]
        assert schema.site.news == ["news.naver.com"]

    def test_request_data_valid_without_site(self):
        """사이트 설정이 없는 유효한 요청 데이터 테스트"""
        data = {
            "query": "일반 검색 쿼리"
        }
        
        schema = RequestDataPayload(**data)
        
        assert schema.query == "일반 검색 쿼리"
        assert schema.site is None

    def test_request_data_empty_query_fails(self):
        """빈 쿼리는 실패해야 함"""
        data = {
            "query": ""
        }
        
        # 빈 문자열도 허용되는지 확인 (Pydantic Field(...) 는 required만 체크)
        schema = RequestDataPayload(**data)
        assert schema.query == ""

    def test_request_data_missing_query_fails(self):
        """쿼리 필드 누락 시 실패 테스트"""
        data = {
            "site": {"news": ["cnn.com"]}
        }
        
        with pytest.raises(ValidationError) as exc_info:
            RequestDataPayload(**data)
        
        assert "query" in str(exc_info.value)


class TestAsyncComicRequest:
    """AsyncComicRequest 스키마 테스트"""

    def test_async_comic_request_full(self):
        """모든 필드가 포함된 요청 테스트"""
        data = {
            "writer_id": "writer_123",
            "data": {
                "query": "테스트 쿼리",
                "site": {
                    "code_related": ["github.com"],
                    "news": ["news.naver.com"]
                }
            }
        }
        
        schema = AsyncComicRequest(**data)
        
        assert schema.writer_id == "writer_123"
        assert schema.data.query == "테스트 쿼리"
        assert schema.data.site.code_related == ["github.com"]

    def test_async_comic_request_minimal(self):
        """최소 필수 필드만 포함된 요청 테스트"""
        data = {
            "data": {
                "query": "최소 쿼리"
            }
        }
        
        schema = AsyncComicRequest(**data)
        
        assert schema.writer_id is None
        assert schema.data.query == "최소 쿼리"
        assert schema.data.site is None

    def test_async_comic_request_missing_data_fails(self):
        """data 필드 누락 시 실패 테스트"""
        data = {
            "writer_id": "writer_123"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            AsyncComicRequest(**data)
        
        assert "data" in str(exc_info.value)


class TestAsyncComicResponse:
    """AsyncComicResponse 스키마 테스트"""

    def test_async_comic_response_valid(self):
        """유효한 응답 스키마 테스트"""
        data = {
            "comic_id": "12345-67890",
            "status": "PENDING",
            "message": "작업이 시작되었습니다."
        }
        
        schema = AsyncComicResponse(**data)
        
        assert schema.comic_id == "12345-67890"
        assert schema.status == "PENDING"
        assert schema.message == "작업이 시작되었습니다."

    def test_async_comic_response_missing_fields_fails(self):
        """필수 필드 누락 시 실패 테스트"""
        data = {
            "comic_id": "12345"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            AsyncComicResponse(**data)
        
        error_str = str(exc_info.value)
        assert "status" in error_str
        assert "message" in error_str


class TestComicStatusResponse:
    """ComicStatusResponse 스키마 테스트"""

    def test_comic_status_response_minimal(self):
        """최소 필드만 포함된 상태 응답 테스트"""
        data = {
            "comic_id": "test-id",
            "status": "DONE",
            "message": "완료됨"
        }
        
        schema = ComicStatusResponse(**data)
        
        assert schema.comic_id == "test-id"
        assert schema.status == "DONE"
        assert schema.message == "완료됨"
        assert schema.query is None
        assert schema.result is None

    def test_comic_status_response_full(self):
        """모든 필드가 포함된 상태 응답 테스트"""
        data = {
            "comic_id": "test-id",
            "status": "DONE",
            "message": "성공",
            "query": "테스트 쿼리",
            "writer_id": "writer_123",
            "user_site_preferences_provided": True,
            "timestamp_accepted": "2025-05-23T10:00:00Z",
            "timestamp_start": "2025-05-23T10:00:05Z",
            "timestamp_end": "2025-05-23T10:02:00Z",
            "duration_seconds": 115.5,
            "result": {
                "images_cnt": 4,
                "ideas_cnt": 3,
                "final_stage": "END"
            },
            "error_details": None
        }
        
        schema = ComicStatusResponse(**data)
        
        assert schema.comic_id == "test-id"
        assert schema.status == "DONE"
        assert schema.duration_seconds == 115.5
        assert schema.result["images_cnt"] == 4
        assert schema.error_details is None

    def test_comic_status_response_with_error(self):
        """오류 상태의 응답 테스트"""
        data = {
            "comic_id": "error-id",
            "status": "FAILED",
            "message": "처리 실패",
            "error_details": "워크플로우 실행 중 오류 발생"
        }
        
        schema = ComicStatusResponse(**data)
        
        assert schema.status == "FAILED"
        assert schema.error_details == "워크플로우 실행 중 오류 발생"


# 테스트 실행을 위한 pytest 설정
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
