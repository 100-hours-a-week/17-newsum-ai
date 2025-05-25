# from pydantic import BaseModel, Field, ValidationError
# from typing import List


# class SearchDocument(BaseModel):
#     title: str = Field(..., description="검색 문서 제목")
#     url: str = Field(..., description="문서 링크")
#     snippet: str = Field(..., description="문서 요약 또는 본문 일부")
#     site: str = Field(..., description="출처 도메인")


# class SearchResultSchema(BaseModel):
#     frame_index: int = Field(..., description="프레임 인덱스")
#     title: str = Field(..., description="프레임 제목")
#     purpose: str = Field(..., description="프레임 목적")
#     result_docs: List[SearchDocument] = Field(..., description="해당 프레임의 검색 결과 리스트")


# def validate_search_result(data: dict) -> SearchResultSchema:
#     """
#     dict → SearchResultSchema 로 변환 및 유효성 검사
#     """
#     return SearchResultSchema(**data)

# __all__ = [
#     "SearchResultSchema",
#     "SearchDocument",
#     "validate_search_result",
#     "ValidationError",
# ]
