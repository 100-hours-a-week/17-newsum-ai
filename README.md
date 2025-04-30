# Newsom AI 서비스

LangGraph와 FastAPI를 사용하여 뉴스 기사로부터 4컷 만화를 생성하는 멀티스텝 AI 워크플로우

## 프로젝트 개요
- **프로젝트명**: Newsom AI 서비스
- **목적**: 뉴스 기사로부터 4컷 만화를 생성하는 멀티스텝 AI 워크플로우 구현
- **기술 스택**: FastAPI, LangGraph, Redis, httpx, BeautifulSoup, Pydantic

## 시작하기

### 필수 요구사항
- Python 3.9 이상
- Redis 서버
- 외부 API 키:
  - LLM API (OpenAI 또는 유사 서비스)
  - 검색 API (Google, Naver, Kakao, Tavily 중 하나 이상)
  - YouTube API (컨텍스트 분석용)

### 설치 방법

1. 레포지토리 클론
```bash
git clone https://github.com/your-username/17-team-4cut.git
cd 17-team-4cut/ai
```

2. 가상환경 생성 및 활성화
```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화 (Windows)
venv\Scripts\activate

# 가상환경 활성화 (macOS/Linux)
source venv/bin/activate
```

3. 의존성 패키지 설치
```bash
pip install -r requirements.txt
```

4. 환경 변수 설정
```bash
# .env.sample을 .env로 복사
cp .env.sample .env

# .env 파일을 편집하여 필요한 API 키와 설정값 추가
```

### 실행 방법

1. Redis 서버 실행 (별도 터미널에서)
```bash
# Redis가 설치되어 있지 않다면 설치
# Windows에서는 WSL 또는 Docker로 Redis 실행 권장
redis-server
```

2. FastAPI 서버 실행
```bash
# 개발 모드로 실행
uvicorn main:app --reload

# 프로덕션 모드로 실행
uvicorn main:app --host 0.0.0.0 --port 8000
```

3. API 문서 확인
- http://localhost:8000/docs 방문하여 Swagger UI로 API 테스트
- http://localhost:8000/redoc 방문하여 ReDoc 문서 확인

## 환경 구성 (.env 파일)

중요 환경 변수 설정 예시:
```
# API 키
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
NAVER_CLIENT_ID=your_naver_client_id
NAVER_CLIENT_SECRET=your_naver_client_secret
YOUTUBE_API_KEY=your_youtube_api_key

# 서비스 설정
REDIS_URL=redis://localhost:6379/0
DEFAULT_MODEL=gpt-4-0125-preview
DEFAULT_LANG=ko

# 결과 저장 설정
SAVE_AGENT_RESULTS=True
SAVE_AGENT_INPUTS=False
SAVE_DEBUG_INFO=True
RESULTS_DIR=./results

# 기타 설정
LANGSMITH_API_KEY=your_langsmith_api_key  # (선택사항) 워크플로우 모니터링용
SEARCH_ENGINE=google  # 기본 검색 엔진 (google, naver, kakao, tavily)
```

## 폴더 구조

```
project_root/
│
├── app/                          # 메인 애플리케이션 코드
│   ├── agents/                   # 각 작업을 수행하는 LangGraph 에이전트
│   ├── tools/                    # 에이전트가 사용하는 독립 기능 모듈
│   ├── workflows/                # LangGraph 기반 워크플로우 그래프, 상태, 전이 정의
│   ├── services/                 # 외부 API(LLM, 이미지, 스토리지, LangSmith) 통신 모듈
│   ├── api/                      # FastAPI HTTP 엔드포인트 및 스키마
│   ├── config/                   # 환경설정 및 상수 관리
│   └── utils/                    # 로거, 재시도, 타이머 등의 유틸리티 모듈
│
├── results/                      # 에이전트 실행 결과 JSON 저장 디렉토리
│
├── tests/                        # 테스트 코드
│   ├── agents/                   # 에이전트 단위 테스트
│   ├── run_tests.py              # 테스트 실행 스크립트
│   ├── test_end_to_end.py        # 엔드투엔드 테스트
│   ├── test_workflow.py          # 워크플로우 통합 테스트
│   └── test_*.py                 # 개별 에이전트 테스트 파일들
│
├── tools/                        # 유틸리티 스크립트
│   └── view_results.py           # 에이전트 결과 뷰어 CLI 도구
│
├── pre-test/                     # 개발 초기 단계에서의 자료 정리 및 기술 검증용 코드 저장소
│
├── main.py                       # FastAPI 서버 엔트리포인트
├── requirements.txt              # Python 패키지 의존성 목록
├── .env.sample                   # 환경설정 샘플 템플릿
└── .gitignore                    # Git 무시 파일 설정
```

## API 사용 방법

### 만화 생성 요청

```bash
curl -X POST "http://localhost:8000/api/v1/comics" \
  -H "Content-Type: application/json" \
  -d '{"query": "우크라이나 전쟁 최근 소식"}'
```

응답:
```json
{
  "comic_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "status": "PENDING",
  "message": "Workflow task accepted."
}
```

### 진행 상태 스트리밍

```bash
curl -X POST "http://localhost:8000/api/v1/comics/3fa85f64-5717-4562-b3fc-2c963f66afa6/stream" \
  -H "Accept: text/event-stream"
```

SSE 응답 예시:
```
event: update
data: {"status": "STARTED", "message": "Collecting news articles..."}

event: update
data: {"status": "IN_PROGRESS", "message": "Scraping article content..."}

event: update
data: {"status": "DONE", "message": "Comic generation completed", "result": {"final_comic_url": "https://storage-url.com/comics/3fa85f64-5717-4562-b3fc-2c963f66afa6.jpg"}}
```

## 에이전트 결과 저장 및 분석

### 결과 저장 기능

프로젝트는 각 에이전트의 처리 결과를 JSON 파일로 자동 저장하는 기능을 포함하고 있습니다. 이 기능은 다음과 같은 장점이 있습니다:

- 워크플로우 단계별 실행 결과 보존
- 디버깅 및 문제 해결 용이
- 결과 품질 분석 및 모니터링
- 성능 최적화를 위한 데이터 수집

### 설정 방법

결과 저장 기능은 기본적으로 활성화되어 있으며, 다음 환경 변수로 설정을 변경할 수 있습니다:

```
# .env 파일 설정
SAVE_AGENT_RESULTS=True  # 에이전트 결과 저장 활성화/비활성화
SAVE_AGENT_INPUTS=False  # 각 에이전트 입력 상태 저장 여부
SAVE_DEBUG_INFO=True     # 오류 및 디버그 정보 저장 여부
RESULTS_DIR=./results    # 결과 저장 경로 (기본값: 프로젝트 루트의 results 폴더)
```

### 결과 뷰어 사용법

제공된 결과 뷰어 CLI 도구를 사용하여 저장된 결과를 쉽게 확인할 수 있습니다:

```bash
# 결과 목록 보기
python tools/view_results.py list

# 상세 정보 포함 목록 보기
python tools/view_results.py list -v

# 특정 만화 ID의 에이전트 결과 상세 보기
python tools/view_results.py view <comic_id>

# 특정 에이전트 결과만 보기
python tools/view_results.py view <comic_id> -a <agent_name>

# 특정 에이전트의 전체 결과 내용 보기
python tools/view_results.py full <comic_id> <agent_name>

# 모든 결과 삭제
python tools/view_results.py clear

# 특정 만화 ID의 결과만 삭제
python tools/view_results.py clear <comic_id>
```

### 저장된 결과 디렉토리 구조

```
results/
├── <comic_id>/                 # 만화 생성 작업별 디렉토리
│   ├── collector_*.json        # Collector Agent 결과
│   ├── scraper_*.json          # Scraper Agent 결과
│   ├── humor_*.json            # Humorator Agent 결과
│   ├── inputs/                 # 에이전트 입력 상태 (선택 저장)
│   │   └── *_*.json
│   ├── errors/                 # 오류 정보 (발생 시)
│   │   └── *_*.json
│   └── debug/                  # 디버그 정보 (선택 저장)
│       └── *_*.json
└── <another_comic_id>/
    └── ...
```

## 개발 워크플로우

### 새 에이전트 추가하기

1. `app/agents/` 디렉토리에 새 에이전트 클래스 파일 생성
```python
# app/agents/new_agent.py
from app.utils.logger import get_logger
from app.services.llm_server_client import LLMServerClient

logger = get_logger(__name__)

class NewAgent:
    def __init__(self):
        self.llm_client = LLMServerClient()
    
    async def run(self, state):
        try:
            # 에이전트 로직 구현
            logger.info("NewAgent processing started")
            
            # 상태에서 필요한 데이터 추출
            input_data = state.get("some_input_key")
            
            # LLM 호출 및 처리
            prompt = "Your prompt here with {input}".format(input=input_data)
            response = await self.llm_client.call_llm_api(prompt)
            
            # 결과를 상태에 추가
            state["new_output_key"] = response
            
            return state
        except Exception as e:
            logger.error(f"Error in NewAgent: {str(e)}")
            state["error_message"] = f"NewAgent failed: {str(e)}"
            return state
```

2. `app/workflows/main_workflow.py` 파일에 에이전트 노드 추가
```python
from app.agents.new_agent import NewAgent
from app.utils.agent_wrapper import save_agent_result

# 기존 코드...

def build_main_workflow() -> StateGraph:
    # 기존 노드...
    
    # 새 노드 추가 (결과 저장 래퍼 적용)
    graph.add_node("new_agent", save_agent_result(NewAgent().run))
    
    # 흐름에 연결
    graph.add_edge("previous_node", "new_agent")
    graph.add_edge("new_agent", "next_node")
    
    # 기존 코드...
```

3. 테스트 파일 작성
```python
# tests/agents/test_new_agent.py
import pytest
from unittest.mock import AsyncMock, patch
from app.agents.new_agent import NewAgent

@pytest.mark.asyncio
async def test_new_agent_success():
    # 모의 객체 설정
    mock_llm_client = AsyncMock()
    mock_llm_client.call_llm_api.return_value = "Expected response"
    
    # 테스트 상태 준비
    state = {"some_input_key": "Test input"}
    
    # 에이전트 생성 및 모의 객체 주입
    agent = NewAgent()
    agent.llm_client = mock_llm_client
    
    # 실행
    result = await agent.run(state)
    
    # 검증
    assert "new_output_key" in result
    assert result["new_output_key"] == "Expected response"
    assert "error_message" not in result
```

### 테스트 실행

모든 테스트:
```bash
python tests/run_tests.py
```

특정 테스트:
```bash
python tests/run_tests.py tests/agents/test_new_agent.py
```

단일 에이전트 테스트:
```bash
python tests/test_new_agent.py
```

## 핵심 컴포넌트

### 1. 워크플로우 상태 (ComicState)
```python
class ComicState(BaseModel):
    """워크플로우 실행 중 전달되는 상태 정보 정의"""
    comic_id: Optional[str] = Field(default=None)  # 만화 생성 작업 고유 ID
    initial_query: Optional[str] = Field(default=None)
    search_results: Optional[List[Dict[str, str]]] = Field(default=None)
    news_urls: Optional[List[str]] = Field(default_factory=list)
    selected_url: Optional[str] = Field(default=None)
    articles: Optional[List[str]] = Field(default_factory=list)
    summaries: Optional[List[str]] = Field(default_factory=list)
    final_summary: Optional[str] = Field(default=None)
    additional_context: Optional[Dict[str, Any]] = Field(default=None)
    public_sentiment: Optional[Dict[str, Dict[str, float]]] = Field(default=None)
    humor_texts: Optional[List[str]] = Field(default_factory=list)
    scenarios: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    image_urls: Optional[List[str]] = Field(default_factory=list)
    final_comic_url: Optional[str] = None
    translated_texts: Optional[List[str]] = Field(default_factory=list)
    error_message: Optional[str] = None
    processing_stats: Optional[Dict[str, Any]] = Field(default_factory=dict)
```

### 2. 메인 워크플로우
```python
def build_main_workflow() -> StateGraph:
    """뉴스 수집부터 만화 완성까지 LangGraph 기반 플로우를 구성합니다."""
    graph = StateGraph(ComicState)

    # 상태 초기화 - comic_id 추가
    graph.add_node("initialize", add_comic_id)

    # 노드 등록 - 에이전트 래퍼 적용 (결과 저장)
    graph.add_node("collect", collect_news)
    graph.add_node("scrape", save_agent_result(ScraperAgent().run))
    graph.add_node("summarize_individual", save_agent_result(IndividualSummarizerAgent().run))
    graph.add_node("summarize_synthesis", save_agent_result(SynthesisSummarizerAgent().run))
    graph.add_node("analyze_content", save_agent_result(ContentSummarizerAgent().run))
    graph.add_node("humor", save_agent_result(HumoratorAgent().run))
    graph.add_node("scenario", save_agent_result(ScenarioWriterAgent().run))
    graph.add_node("image", save_agent_result(ImagerAgent().run))
    # graph.add_node("postprocess", save_agent_result(PostProcessorAgent().run))
    # graph.add_node("translate", save_agent_result(TranslatorAgent().run))

    # 흐름 연결
    graph.set_entry_point("initialize")
    graph.add_edge("initialize", "collect")
    graph.add_edge("collect", "scrape")
    graph.add_edge("scrape", "summarize_individual")
    graph.add_edge("summarize_individual", "summarize_synthesis")
    graph.add_edge("summarize_synthesis", "analyze_content")
    graph.add_edge("analyze_content", "humor")
    graph.add_edge("humor", "scenario")
    graph.add_edge("scenario", "image")
    graph.add_edge("image", END)

    return graph.compile()
```

## 구현된 에이전트 목록

1. **Collector Agent** (`collector_agent.py`)
   - 기능: 검색 엔진 API를 활용하여 뉴스 기사 URL 수집
   - 특징: 다중 검색 엔진 지원, 한국 키워드 인식, 폴백 메커니즘

2. **Scraper Agent** (`scraper_agent.py`)
   - 기능: URL 리스트를 받아 각 URL의 본문 텍스트를 비동기적으로 스크래핑
   - 특징: 병렬 처리, 다양한 웹사이트 구조 처리

3. **Individual Summarizer Agent** (`individual_summarizer_agent.py`)
   - 기능: 개별 기사 텍스트를 요약
   - 특징: LLM을 활용한 비동기 요약 처리

4. **Synthesis Summarizer Agent** (`synthesis_summarizer_agent.py`)
   - 기능: 개별 요약을 하나의 종합 요약으로 통합
   - 특징: 다중 소스의 정보 통합 능력

5. **Content Summarizer Agent** (`content_summarizer_agent.py`)
   - 기능: YouTube 비디오 및 댓글을 통해 추가 컨텍스트 수집
   - 특징: 다차원 소스에서 추가 관점 수집

6. **Humorator Agent** (`humorator_agent.py`)
   - 기능: 뉴스에서 유머러스한 관점 추출
   - 특징: 공감적 유머 생성, 컨텍스트 이해 기반 접근

7. **ScenarioWriter Agent** (`scenariowriter_agent.py`)
   - 기능: 유머 포인트를 기반으로 4컷 만화 시나리오 작성
   - 특징: 패널별 장면 묘사 및 대사 생성

8. **Imager Agent** (`imager_agent.py`)
   - 기능: 시나리오를 기반으로 이미지 생성 API 호출
   - 특징: 각 패널별 장면 이미지 생성, 프롬프트 최적화

## 문제 해결 가이드

### 일반적인 문제

1. **Redis 연결 오류**
   - 문제: `Cannot connect to Redis at localhost:6379`
   - 해결: Redis 서버가 실행 중인지 확인하고, 올바른 포트와 인증 정보를 설정했는지 확인합니다.

2. **API 키 오류**
   - 문제: `API key not valid` 또는 관련 오류
   - 해결: `.env` 파일에 올바른 API 키가 설정되어 있는지 확인합니다.

3. **응답 파싱 오류**
   - 문제: `Humorator agent failed: Cannot parse response`
   - 해결: LLM 모델 응답 형식을 확인하고 에이전트의 파싱 로직을 업데이트합니다.

4. **결과 파일이 생성되지 않는 문제**
   - 문제: 에이전트 결과 JSON 파일이 생성되지 않음
   - 해결: `.env` 파일에서 `SAVE_AGENT_RESULTS` 설정이 `True`로 되어 있는지 확인하고, `RESULTS_DIR` 경로에 쓰기 권한이 있는지 확인합니다.

### LLM 응답 품질 문제

1. **뉴스 요약 부정확**
   - 해결: `app/agents/individual_summarizer_agent.py`에서 프롬프트를 개선하거나 `max_tokens` 설정을 늘립니다.

2. **유머 생성 품질 낮음**
   - 해결: `app/agents/humorator_agent.py`에서 프롬프트를 개선하고 예시를 추가합니다.

## 다음 단계 작업

현재 미구현 상태인 다음 모듈에 대한 개발 가이드:

1. **후처리 에이전트**
   - 파일 위치: `app/agents/post_processor_agent.py`
   - 입력: `image_urls`, `scenarios` (텍스트 및 이미지)
   - 출력: `final_comic_url` (최종 레이아웃된 만화 URL)
   - 요구사항: 이미지 처리 라이브러리(PIL) 및 폰트 설정 필요

2. **번역 에이전트**
   - 파일 위치: `app/agents/translator_agent.py`
   - 입력: `scenarios` (만화 대사)
   - 출력: `translated_texts` (번역된 대사 리스트)
   - 요구사항: 다국어 지원, 컨텍스트 인식 번역

## 작업 흐름도

```
[사용자 쿼리] → [뉴스 수집] → [스크래핑] → [개별 요약] → [종합 요약] 
                  → [컨텍스트 분석] → [유머 추출] → [시나리오 작성] 
                  → [이미지 생성] → [후처리*] → [번역*] → [최종 만화]

* = 미구현 단계
```

## 기여 가이드

1. 브랜치 관리
   - `main`: 안정화된 릴리스 브랜치
   - `develop`: 개발 브랜치
   - 기능 개발: `feature/기능명`
   - 버그 수정: `bugfix/이슈번호`

2. 코드 스타일
   - `black` 포맷터와 `flake8` 린터 사용
   - 함수와 클래스에 타입 힌트 추가
   - 중요 함수에는 docstring 작성

3. 테스트
   - 새 기능에는 항상 테스트 작성
   - 단위 테스트는 `pytest`로 구현

## 참고 문서

- FastAPI 공식 문서: https://fastapi.tiangolo.com/
- LangGraph 공식 문서: https://python.langchain.com/docs/langgraph/
- Pydantic 공식 문서: https://docs.pydantic.dev/
- Redis 공식 문서: https://redis.io/documentation
