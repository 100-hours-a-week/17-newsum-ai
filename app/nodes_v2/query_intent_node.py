from __future__ import annotations

import json
import itertools
import tldextract
from typing import List

from app.services.llm_service import LLMService
from app.tools.think_parser_tool import extract_json
from app.workflows.state_v2 import WorkflowState
from app.api.v2.schemas.nodes.query_intent import validate_query_analysis
from pydantic import ValidationError

# ---------------------------
# 1. 신뢰 도메인 정의
# ---------------------------
CATEGORY_DOMAINS = {
    "Politics": [
        "bbc.com", "cnn.com", "politico.com", "reuters.com", "nytimes.com",
        "theguardian.com", "chosun.com", "joongang.co.kr", "hani.co.kr",
        "khan.co.kr", "ohmynews.com"
    ],
    "IT": [
        "techcrunch.com", "wired.com", "theverge.com", "zdnet.com",
        "arstechnica.com", "zdnet.co.kr", "etnews.com", "bloter.net",
        "itworld.co.kr"
    ],
    "Economy": [
        "bloomberg.com", "ft.com", "reuters.com", "forbes.com",
        "marketwatch.com", "cnbc.com", "hankyung.com", "mk.co.kr",
        "sedaily.com", "etoday.co.kr"
    ],
    "Meme": [  # punchline 전용 커뮤니티
        "reddit.com", "knowyourmeme.com", "imgur.com", "memedroid.com",
        "9gag.com"
    ],
}
ALL_DOMAINS: List[str] = list(
    set(itertools.chain.from_iterable(CATEGORY_DOMAINS.values()))
)

CATEGORY_HINT = "Politics | IT | Economy"
PURPOSE_HINT = "explanation | conflict | punchline"


# ---------------------------
# 2. QueryIntentNode
# ---------------------------
class QueryIntentNode:
    def __init__(self, llm: LLMService):
        self.llm = llm

    # -----------------------
    # 2-1. LLM 프롬프트 생성
    # -----------------------
    def build_prompt(self, trusted_sites: list) -> str:
        schema_example = {
            "category": f"<{CATEGORY_HINT} 중 하나 선택>",
            "refined_intent": "<~를 풍자하는 4컷 만화 기획>",
            "frames": [
                {
                    "title": "<관점 제목>",
                    "purpose": f"<{PURPOSE_HINT} 중 하나>",
                    "search_terms": ["<검색어1>", "<검색어2>", "..."],
                    "preferred_sources": ["<trusted_sites 중 선택>"],
                }
            ],
        }
        json_example = json.dumps(schema_example, ensure_ascii=False, indent=2)
        trusted_sample = ", ".join(trusted_sites[:10]) + (
            " 외 다수" if len(trusted_sites) > 10 else ""
        )

        return f"""당신은 뉴스 기반 풍자 만화를 기획하기 위한 '의도 분석' 에이전트입니다.
다음 지침을 철저히 따른 **JSON** 만 반환하세요 (추가 텍스트, <think> 금지).

# 스키마 예시
{json_example}

# 작성 규칙
- category: {CATEGORY_HINT} 중 하나
- refined_intent: '무엇을 풍자하는 4컷 만화인지' 한 문장
- frames (정확히 3개):
    * purpose는 explanation / conflict / punchline 각 1회씩
    * title, search_terms(2~4개), preferred_sources 포함
    * preferred_sources는 아래 도메인 중 선택
- title, purpose, search_terms 모두 한글로 작성

# trusted sites (샘플): {trusted_sample}
"""

    # -----------------------
    # 2-2. 도메인 유틸
    # -----------------------
    @staticmethod
    def domain_from_url(url: str) -> str:
        return tldextract.extract(url).registered_domain

    def get_allowed_domains(self) -> list:
        """모든 카테고리 + Meme 도메인 허용 (혼합 허용)"""
        return ALL_DOMAINS

    # -----------------------
    # 2-3. 메인 실행 함수
    # -----------------------
    async def run(self, state: WorkflowState) -> WorkflowState:
        user_query = state.query.original_query or ""

        # ① 프롬프트 준비 & LLM 호출
        prompt = self.build_prompt(trusted_sites=ALL_DOMAINS)
        llm_resp = await self.llm.generate_text(
            system_prompt_content=prompt,
            prompt=user_query,
            temperature=0.7,
            max_tokens=1500,
        )
        raw_txt = llm_resp.get("generated_text", "")
        parsed_json = extract_json(raw_txt)

        # ② 구조 검증
        try:
            validated = validate_query_analysis(parsed_json)
        except ValidationError as e:
            raise RuntimeError(f"💥 JSON 스키마 검증 실패: {e}")

        # ③ 프레임·purpose 갯수 강제 점검
        if len(validated.frames) != 3:
            raise RuntimeError("💥 프레임 수가 3개가 아닙니다.")
        purpose_set = {f.purpose for f in validated.frames}
        if set(purpose_set) != {"explanation", "conflict", "punchline"}:
            raise RuntimeError("💥 purpose 분포가 요구사항(explanation·conflict·punchline 각 1개)에 맞지 않습니다.")

        # ④ 도메인 허용 여부 검사 (혼합 허용)
        allowed = self.get_allowed_domains()
        for f in validated.frames:
            for src in f.preferred_sources:
                if self.domain_from_url(src) not in allowed:
                    print(f"[경고] '{src}' → 허용되지 않은 도메인")

        # ⑤ 상태 업데이트
        state.query.original_query = user_query # logging 용도
        state.query.category = validated.category
        state.query.refined_intent = validated.refined_intent
        state.query.frames = validated.frames  # 그대로 Dict/List 구조 유지

        return state
