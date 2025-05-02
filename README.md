# Newsum AI 서비스

LangGraph와 FastAPI를 사용하여 뉴스 기사로부터 4컷 만화를 생성하는 멀티스텝 AI 워크플로우

## 프로젝트 개요
- **프로젝트명**: Newsum AI 서비스
- **목적**: 뉴스 기사로부터 4컷 만화를 생성하는 멀티스텝 AI 워크플로우 구현
- **기술 스택**: FastAPI, LangGraph, Redis, aiohttp, BeautifulSoup, Pydantic

## 시작하기

### 필수 요구사항
- Python 3.9 이상
- Redis 서버
- 외부 API 키:
  - LLM API (OpenAI 또는 유사 서비스)
  - 검색 API (Google, Naver, Kakao, Tavily 중 하나 이상)
  - 소셜 미디어 API (Twitter/X, Reddit)

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
TWITTER_API_KEY=your_twitter_api_key
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret

# 서비스 설정
REDIS_URL=redis://localhost:6379/0
DEFAULT_LLM_MODEL=gpt-4-0125-preview
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
ai/                             # AI 서비스 루트 디렉토리
├── app/                        # 애플리케이션 코드
│   ├── api/                    # API 인터페이스 (FastAPI)
│   │   ├── v1/                 # API 버전 1
│   │   │   ├── background_tasks.py # 백그라운드 작업 처리
│   │   │   ├── endpoints.py    # API 엔드포인트 정의
│   │   │   └── schemas.py      # API 데이터 모델(스키마)
│   │
│   ├── config/                 # 설정 및 환경 변수 관리
│   │   └── settings.py         # 중앙 설정 객체
│   │
│   ├── nodes/                  # LangGraph 워크플로우 노드
│   │   ├── n01_initialize_node.py    # 워크플로우 초기화
│   │   ├── n02_topic_analyzer_node.py # 주제 분석
│   │   ├── n03_news_collector_node.py # 뉴스 URL 수집
│   │   ├── n04_opinion_collector_node.py # 의견 URL 수집
│   │   ├── n05_news_scraper_node.py  # 뉴스 스크래핑
│   │   ├── n06_opinion_scraper_node.py # 의견 스크래핑
│   │   ├── n07_filter_node.py        # 의견 필터링 및 클러스터링
│   │   ├── n08_news_summarizer_node.py # 뉴스 요약
│   │   ├── n09_opinion_summarizer_node.py # 의견 요약
│   │   ├── n10_synthesis_summarizer_node.py # 종합 요약
│   │   ├── n11_evaluate_summary_node.py # 요약 평가
│   │   ├── n12_trend_analyzer_node.py # 트렌드 분석
│   │   ├── n13_progress_report_node.py # 진행 보고서
│   │   ├── n14_idea_generator_node.py # 아이디어 생성
│   │   ├── n15_scenario_writer_node.py # 시나리오 작성
│   │   ├── n16_scenario_report_node.py # 시나리오 보고서
│   │   ├── n17_imager_node.py        # 이미지 생성
│   │   ├── n18_translator_node.py    # 텍스트 번역
│   │   └── n19_postprocessor_node.py # 최종 후처리
│   │
│   ├── services/               # 외부 API/서비스 클라이언트
│   │   ├── database_con_client_v2.py # 데이터베이스(Redis) 클라이언트
│   │   ├── image_server_client_v2.py # 이미지 생성 API 클라이언트
│   │   ├── langsmith_client_v2.py # LangSmith 모니터링 클라이언트
│   │   ├── llm_server_client_v2.py # LLM API 클라이언트
│   │   ├── spam_detector.py   # 스팸 필터링 서비스
│   │   └── storage_client_v2.py # 스토리지 클라이언트
│   │
│   ├── tools/                  # 기능별 도구 모듈
│   │   ├── analysis/          # 분석 도구
│   │   │   ├── language_detector.py  # 언어 감지 도구
│   │   │   └── text_clusterer.py    # 텍스트 클러스터링 도구
│   │   ├── image/             # 이미지 처리 도구
│   │   ├── llm/               # LLM 유틸리티
│   │   ├── postprocessing/    # 후처리 도구
│   │   ├── scraping/          # 웹 스크래핑 도구
│   │   │   ├── article_scraper.py   # 뉴스 기사 스크래퍼
│   │   │   └── selenium_scraper.py  # Selenium 기반 스크래퍼
│   │   ├── search/            # 검색 도구
│   │   │   ├── google.py      # Google 검색
│   │   │   ├── naver.py       # Naver 검색
│   │   │   └── rss.py         # RSS 피드 검색
│   │   └── social/            # 소셜 미디어 도구
│   │       ├── reddit.py      # Reddit 도구
│   │       └── twitter.py     # Twitter(X) 도구
│   │
│   ├── utils/                  # 유틸리티 함수
│   │   ├── agent_wrapper.py   # 에이전트 래퍼
│   │   ├── logger.py          # 로깅 설정
│   │   ├── result_viewer.py   # 결과 뷰어
│   │   └── retry.py           # 재시도 로직
│   │
│   ├── workflows/              # LangGraph 워크플로우 정의
│   │   ├── main_workflow.py   # 메인 워크플로우 구성
│   │   ├── state.py           # 워크플로우 상태 모델
│   │   └── transitions.py     # 상태 전이 로직
│   │
│   └── dependencies.py         # FastAPI 의존성 주입
│
├── pre-test/                   # 사전 테스트 및 개발 자료
│
├── main.py                     # FastAPI 애플리케이션 엔트리포인트
├── requirements.txt            # 의존성 패키지 목록
├── .env.sample                 # 환경 변수 샘플
└── README.md                   # 프로젝트 설명
```

## 워크플로우 아키텍처

본 프로젝트는 LangGraph를 기반으로 한 유향 비순환 그래프(DAG)로 구성된 워크플로우를 사용합니다. 워크플로우는 다음과 같은 단계로 구성됩니다:

1. **초기화 (n01_initialize_node)**
   - 워크플로우 상태 초기화
   - 고유 ID 생성 및 설정 구성

2. **주제 분석 (n02_topic_analyzer_node)**
   - LLM을 활용한 쿼리 분석
   - 핵심 주제, 엔티티, 검색 키워드 추출
   - Redis 캐싱 기능 지원

3. **컨텐츠 수집 (병렬 처리)**
   - **뉴스 URL 수집 (n03_news_collector_node)**
     - Google, Naver, RSS 등 다양한 소스에서 뉴스 URL 수집
   - **의견 URL 수집 (n04_opinion_collector_node)**
     - Twitter, Reddit, Google 등에서 의견 URL 수집

4. **컨텐츠 스크래핑 (병렬 처리)**
   - **뉴스 스크래핑 (n05_news_scraper_node)**
     - 뉴스 기사 본문, 제목, 저자 등 추출
   - **의견 스크래핑 (n06_opinion_scraper_node)**
     - 소셜 미디어 의견, 댓글 등 추출
     - Selenium 기반 동적 페이지 스크래핑 지원

5. **의견 필터링 및 클러스터링 (n07_filter_node)**
   - 언어 감지 및 필터링
   - 스팸 필터링
   - 의견 클러스터링

6. **콘텐츠 요약 (미구현)**
   - **뉴스 요약 (n08_news_summarizer_node)**
   - **의견 요약 (n09_opinion_summarizer_node)**
   - **종합 요약 (n10_synthesis_summarizer_node)**

7. **요약 평가 및 트렌드 분석 (미구현)**
   - **요약 평가 (n11_evaluate_summary_node)**
   - **트렌드 분석 (n12_trend_analyzer_node)**

8. **만화 생성 (미구현)**
   - **아이디어 생성 (n14_idea_generator_node)**
   - **시나리오 작성 (n15_scenario_writer_node)**
   - **이미지 생성 (n17_imager_node)**
   - **번역 (n18_translator_node)**
   - **최종 후처리 (n19_postprocessor_node)**

**참고**: 현재 워크플로우는 7번째 단계(의견 필터링 및 클러스터링)까지만 구현되어 있습니다.

## API 엔드포인트

### 1. 만화 생성 요청
```
POST /v1/comics/generate
```
- **설명**: 사용자 쿼리를 받아 만화 생성 워크플로우를 백그라운드에서 실행합니다.
- **요청 파라미터**: `query` (string) - 만화로 변환할 주제 쿼리
- **응답**: 작업 ID와 상태 메시지
```json
{
  "message": "만화 생성 작업이 시작되었습니다.",
  "comic_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"
}
```

### 2. 만화 생성 상태 조회
```
GET /v1/comics/status/{comic_id}
```
- **설명**: 특정 comic_id에 대한 만화 생성 작업의 현재 상태를 조회합니다.
- **URL 파라미터**: `comic_id` (string) - 조회할 만화 생성 작업의 ID
- **응답**: 작업 상태, 메시지, 결과(완료된 경우)
```json
{
  "status": "IN_PROGRESS",
  "message": "의견 스크래핑 중...",
  "timestamp_start": "2024-05-01T14:30:00.123456Z"
}
```

## 주요 컴포넌트 설명

### 1. 워크플로우 상태 (ComicState)

워크플로우 실행 중에 노드 간에 데이터를 전달하는 상태 모델입니다. Pydantic `BaseModel`을 사용하여 다음과 같은 필드들을 정의합니다:

- **식별자**: `comic_id`, `trace_id`, `timestamp`, `initial_query`
- **설정**: `config` - 워크플로우 실행 설정
- **분석 결과**: `topic_analysis`, `search_keywords`
- **수집 URLs**: `fact_urls`, `opinion_urls`
- **스크래핑 결과**: `articles`, `opinions_raw`
- **필터링 결과**: `opinions_clean`
- **요약 결과**: `news_summaries`, `opinion_summaries`, `final_summary`
- **평가 및 분석**: `evaluation_metrics`, `decision`, `trend_scores`
- **창작 컨텐츠**: `comic_ideas`, `chosen_idea`, `scenarios`, `image_urls`
- **최종 결과**: `final_comic`
- **기타**: `used_links`, `processing_stats`, `error_message`

### 2. 노드 구현 패턴

모든 노드는 다음과 같은 일관된 패턴으로 구현됩니다:

1. **초기화 (`__init__`)**
   - 필요한 서비스/도구 인스턴스를 의존성 주입 받음

2. **실행 메서드 (`run` 또는 `execute`)**
   - `ComicState` 객체를 입력 받음
   - 노드 고유 로직 실행
   - 업데이트된 상태 필드를 딕셔너리로 반환

3. **도우미 메서드**
   - 프롬프트 생성, 응답 파싱, API 호출 등의 유틸리티 함수

### 3. 리소스 관리 패턴

`main.py`의 `lifespan` 컨텍스트 매니저를 통해 다음과 같은 리소스 관리가 이루어집니다:

1. **초기화 단계**:
   - HTTP 세션 (aiohttp.ClientSession) 생성
   - 데이터베이스 클라이언트 생성
   - 서비스 및 도구 인스턴스화
   - 노드 인스턴스화 (의존성 주입)
   - 워크플로우 그래프 빌드 및 컴파일

2. **종료 단계**:
   - 리소스 역순 정리
   - 비동기 및 동기 close 메서드 호출
   - Selenium 등 특수 리소스 정리

### 4. 백그라운드 작업 처리

FastAPI의 `BackgroundTasks`를 활용하여 장시간 실행 작업을 비동기적으로 처리합니다:

1. **워크플로우 트리거 (`trigger_workflow_task`)**:
   - 작업 ID 생성 및 초기 DB 상태 설정
   - 백그라운드 작업으로 워크플로우 실행 예약

2. **워크플로우 실행 함수 (`workflow_runner`)**:
   - 상태 업데이트 (STARTED)
   - LangGraph 워크플로우 실행
   - 최종 상태 추출 및 DB 업데이트
   - 예외 처리 및 오류 상태 기록

## 개발 가이드

### 새 노드 추가하기

1. `app/nodes/` 디렉토리에 새 노드 클래스 파일 생성:
```python
# app/nodes/nXX_new_node.py
from app.utils.logger import get_logger
from app.workflows.state import ComicState
from typing import Dict, Any

logger = get_logger("NewNode")

class NewNode:
    def __init__(self, dependency1, dependency2):
        self.dependency1 = dependency1
        self.dependency2 = dependency2
    
    async def execute(self, state: ComicState) -> Dict[str, Any]:
        """
        노드 실행 메서드
        
        Args:
            state: 현재 워크플로우 상태
            
        Returns:
            업데이트할 ComicState 필드 딕셔너리
        """
        try:
            # 노드 로직 구현
            result = await self._process_data(state.some_field)
            
            # 결과 및 처리 시간 반환
            return {
                "output_field": result,
                "error_message": None
            }
        except Exception as e:
            logger.error(f"Error in NewNode: {str(e)}", exc_info=True)
            return {"error_message": f"NewNode failed: {str(e)}"}
            
    async def _process_data(self, data):
        # 내부 처리 로직
        return processed_data
```

2. `main.py`의 `lifespan` 함수에 노드 인스턴스 생성 추가:
```python
# ... 기존 코드 ...
from app.nodes import nXX_new_node
# ... 기존 코드 ...

# 노드 인스턴스화
nodeXX = nXX_new_node.NewNode(dependency1=some_service, dependency2=some_tool)
```

3. `app/workflows/main_workflow.py`에 노드 추가:
```python
# import 추가
from app.nodes.nXX_new_node import NewNode

# build_main_workflow 함수 파라미터 추가
def build_main_workflow(
    # ... 기존 노드 파라미터 ...
    new_node: NewNode,
    # ...
) -> StateGraph:
    # ... 기존 코드 ...
    
    # 노드 추가
    graph.add_node("new_step", new_node.execute)
    
    # 엣지 추가 (예: node07 다음에 실행)
    graph.add_edge("filter_opinions", "new_step")
    
    # 다음 노드가 없으면 END로 연결
    graph.add_edge("new_step", END)
    
    # ... 기존 코드 ...
```

### 배포 고려사항

1. **메모리 관리**:
   - `lifespan` 컨텍스트 매니저에서 모든 리소스를 적절히 초기화하고 정리
   - 특히 Selenium 등의 무거운 리소스 사용 시 주의

2. **확장성**:
   - 워크플로우를 더 작은 서브그래프로 분할하여 관리 용이성 향상
   - 각 노드의 독립적 테스트 가능성 확보

3. **오류 처리**:
   - 각 노드에서 적절한 예외 처리 및 로깅
   - 중요 노드의 실패 시 대체 전략 구현 (예: 백업 API 사용)

4. **모니터링**:
   - LangSmith 및 내부 로깅을 통한 워크플로우 실행 추적
   - 성능 병목 식별 및 최적화

## 문제 해결 가이드

### 일반적인 문제

1. **Redis 연결 오류**
   - 문제: `Cannot connect to Redis at localhost:6379`
   - 해결: Redis 서버가 실행 중인지 확인하고, 올바른 포트와 인증 정보를 설정했는지 확인합니다.

2. **LLM API 응답 파싱 오류**
   - 문제: `Topic analyzer failed: Failed to parse LLM response`
   - 해결: `app/nodes/n02_topic_analyzer_node.py`의 응답 파싱 로직을 확인하고, LLM 서비스 응답 형식에 맞게 조정합니다.

3. **병렬 처리 이슈**
   - 문제: 뉴스와 의견 수집이 동시에 실행되지 않음
   - 해결: LangGraph 그래프 구성을 확인하고, 병렬 실행을 위한 엣지 연결이 올바른지 확인합니다.

4. **메모리 누수**
   - 문제: 장시간 실행 시 메모리 사용량 증가
   - 해결: `lifespan` 및 `finally` 블록에서 모든 리소스가 제대로 정리되고 있는지 확인합니다.

## 다음 단계

현재 구현된 7단계 이후의 노드들을 구현하여 완전한 만화 생성 파이프라인을 완성할 수 있습니다:

1. **요약 노드 구현 (n08~n10)**
   - 뉴스 및 의견 요약
   - 종합 요약 생성

2. **평가 및 분석 노드 구현 (n11~n12)**
   - 요약 품질 평가
   - 트렌드 분석

3. **만화 생성 노드 구현 (n14~n19)**
   - 아이디어 생성
   - 시나리오 작성
   - 이미지 생성 및 후처리

## 참고 문서

- FastAPI 공식 문서: https://fastapi.tiangolo.com/
- LangGraph 공식 문서: https://python.langchain.com/docs/langgraph/
- Pydantic 공식 문서: https://docs.pydantic.dev/
- Redis 공식 문서: https://redis.io/documentation