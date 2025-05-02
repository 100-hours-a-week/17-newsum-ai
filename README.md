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
  - 이미지 생성 API (추가됨)
  - 네이버 Papago 번역 API (추가됨)

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
TWITTER_BEARER_TOKEN=your_twitter_bearer_token  # 추가됨: TwitterCountsTool용
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
IMAGE_SERVER_URL=your_image_server_url          # 추가됨: 이미지 생성 API 엔드포인트
IMAGE_SERVER_API_TOKEN=your_image_api_token     # 추가됨: 이미지 생성 API 인증 토큰

# 서비스 설정
REDIS_URL=redis://localhost:6379/0
DEFAULT_LLM_MODEL=gpt-4-0125-preview
DEFAULT_LANG=ko
DEFAULT_SOURCE_LANG=ko                          # 추가됨: 번역 소스 언어
DEFAULT_TARGET_LANG=en                          # 추가됨: 번역 타겟 언어
ENABLE_TRANSLATION=True                         # 추가됨: 번역 활성화 여부
DEFAULT_TRANSLATOR_CONCURRENCY=3                # 추가됨: 번역 동시성 제한

# 이미지 생성 설정 (추가됨)
DEFAULT_IMAGE_MODEL=sdxl                        # 이미지 생성 모델
DEFAULT_IMAGE_HEIGHT=1024                       # 이미지 높이
DEFAULT_IMAGE_WIDTH=1024                        # 이미지 너비
DEFAULT_IMAGE_STYLE=comic book, manhwa style   # 기본 스타일
DEFAULT_IMAGE_STYLE_PRESET=comic               # 스타일 프리셋
DEFAULT_IMAGE_NEGATIVE_PROMPT=ugly, deformed    # 네거티브 프롬프트
DEFAULT_MAX_IMAGE_PROMPT_LEN=500               # 최대 프롬프트 길이
IMAGE_STORAGE_PATH=./results/images            # 이미지 저장 경로
IMAGE_API_RETRIES=3                            # 이미지 API 재시도 횟수

# ControlNet 설정 (추가됨)
ENABLE_CONTROLNET=False                        # ControlNet 활성화 여부
DEFAULT_CONTROLNET_TYPE=previous_panel         # ControlNet 타입
DEFAULT_CONTROLNET_WEIGHT=0.5                  # ControlNet 가중치

# 트렌드 분석 설정 (추가됨)
PYTRENDS_TIMEFRAME=now 7-d                     # Google Trends 시간 범위
PYTRENDS_GEO=KR                                # 지역 제한
PYTRENDS_HL=ko                                 # 언어 설정
PYTRENDS_BATCH_DELAY_SEC=3                     # 배치 간 딜레이
TWITTER_COUNTS_DELAY_SEC=1                     # Twitter API 호출 간 딜레이

# 결과 저장 설정
SAVE_AGENT_RESULTS=True
SAVE_AGENT_INPUTS=False
SAVE_DEBUG_INFO=True
RESULTS_DIR=./results

# 기타 설정
LANGSMITH_API_KEY=your_langsmith_api_key  # (선택사항) 워크플로우 모니터링용
SEARCH_ENGINE=google  # 기본 검색 엔진 (google, naver, kakao, tavily)
```

## 새로 추가된 기능

### 1. 트렌드 분석 도구
- **GoogleTrendsTool**: 키워드 관련 Google Trends 데이터를 가져와서 관심도 점수 분석
  - `pytrends` 라이브러리 활용
  - 배치 처리 및 내부 재시도 로직 구현
  - 설정 예시: `PYTRENDS_TIMEFRAME`, `PYTRENDS_GEO`, `PYTRENDS_HL`, `PYTRENDS_BATCH_DELAY_SEC`

- **TwitterCountsTool**: Twitter의 최근 트윗 카운트 API를 활용한 트렌드 분석
  - 키워드별 최근 7일간 트윗 수 계산
  - 비동기 처리 및 내부 재시도 로직 구현
  - 설정 예시: `TWITTER_BEARER_TOKEN`, `TWITTER_COUNTS_DELAY_SEC`

### 2. 이미지 생성 클라이언트
- **ImageGenerationClient**: 외부 이미지 생성 API와 상호작용
  - 다양한 이미지 생성 파라미터 지원 (모델, 크기, 스타일 등)
  - 에러 처리 및 응답 타입 유연한 처리 (JSON 또는 바이너리 이미지)
  - 로컬 이미지 저장 기능 
  - 설정 예시: `IMAGE_SERVER_URL`, `IMAGE_SERVER_API_TOKEN`, `IMAGE_STORAGE_PATH`

### 3. 번역 서비스
- **PapagoTranslationService**: 네이버 Papago NMT API를 활용한 번역 
  - 스크립트 대사 등의 다국어 지원 (기본: 한국어→영어)
  - 비동기 API 호출 및 내부 재시도 로직 구현
  - 설정 예시: `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET`, `DEFAULT_SOURCE_LANG`, `DEFAULT_TARGET_LANG`

### 4. 새 노드 구현
- **ImagerNode (n17)**: 4컷 시나리오 각 패널에 대한 이미지 생성
  - ImageGenerationClient 활용
  - 프롬프트 자동 구성 및 최적화
  - ControlNet 지원 옵션 (이전 패널 이미지 참조 등)
  - 병렬 처리 대신 순차 처리로 일관성 유지

- **TranslatorNode (n18)**: 시나리오 대사 및 텍스트 번역
  - PapagoTranslationService 활용
  - 멀티스레딩으로 동시 번역 처리 최적화
  - 비활성화 옵션 지원 (`ENABLE_TRANSLATION`)

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
│   │   ├── n12_trend_analyzer_node.py # 트렌드 분석 (구현됨)
│   │   ├── n13_progress_report_node.py # 진행 보고서
│   │   ├── n14_idea_generator_node.py # 아이디어 생성
│   │   ├── n15_scenario_writer_node.py # 시나리오 작성
│   │   ├── n16_scenario_report_node.py # 시나리오 보고서
│   │   ├── n17_imager_node.py        # 이미지 생성 (구현됨)
│   │   ├── n18_translator_node.py    # 텍스트 번역 (구현됨)
│   │   └── n19_postprocessor_node.py # 최종 후처리
│   │
│   ├── services/               # 외부 API/서비스 클라이언트
│   │   ├── database_con_client_v2.py # 데이터베이스(Redis) 클라이언트
│   │   ├── image_server_client_v2.py # 이미지 생성 API 클라이언트 (추가됨)
│   │   ├── langsmith_client_v2.py # LangSmith 모니터링 클라이언트
│   │   ├── llm_server_client_v2.py # LLM API 클라이언트
│   │   ├── papago_translation_service.py # Papago API 클라이언트 (추가됨)
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
│   │   ├── social/            # 소셜 미디어 도구
│   │   │   ├── reddit.py      # Reddit 도구
│   │   │   └── twitter.py     # Twitter(X) 도구
│   │   └── trends/            # 트렌드 분석 도구 (추가됨)
│   │       ├── google_trends.py # Google Trends 분석 (추가됨)
│   │       └── twitter_counts.py # Twitter 카운트 분석 (추가됨)
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

6. **콘텐츠 요약**
   - **뉴스 요약 (n08_news_summarizer_node)**
   - **의견 요약 (n09_opinion_summarizer_node)**
   - **종합 요약 (n10_synthesis_summarizer_node)**

7. **요약 평가 및 트렌드 분석**
   - **요약 평가 (n11_evaluate_summary_node)**
   - **트렌드 분석 (n12_trend_analyzer_node)** (구현됨)
     - Google Trends 분석
     - Twitter 카운트 분석 

8. **아이디어 생성 및 시나리오 작성**
   - **진행 보고서 (n13_progress_report_node)**
   - **아이디어 생성 (n14_idea_generator_node)**
   - **시나리오 작성 (n15_scenario_writer_node)**
   - **시나리오 보고서 (n16_scenario_report_node)**

9. **만화 생성 및 후처리** (구현됨)
   - **이미지 생성 (n17_imager_node)** (구현됨)
     - 각 패널별 이미지 생성
     - ControlNet 지원 (옵션)
   - **번역 (n18_translator_node)** (구현됨)
     - 대사 등의 다국어 지원
   - **최종 후처리 (n19_postprocessor_node)**
     - 이미지와 텍스트 결합
     - 최종 만화 생성

**참고**: 워크플로우는 이제 9번째 단계(만화 생성 및 후처리)까지 구현되었습니다.

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
- **창작 컨텐츠**: `comic_ideas`, `chosen_idea`, `scenarios`, `image_urls`, `translated_text`
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
  "message": "이미지 생성 중...",
  "timestamp_start": "2024-05-01T14:30:00.123456Z"
}
```