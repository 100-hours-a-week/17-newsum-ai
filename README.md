# NewSum AI 서비스

NewSum AI 서비스는 뉴스 기사나 사용자 의견을 기반으로 만화를 생성하는 FastAPI 기반 AI 플랫폼입니다. LangGraph 워크플로우를 통한 자동화된 콘텐츠 생성과 실시간 AI 채팅 기능을 제공합니다.

## 목차

- [프로젝트 개요](#프로젝트-개요)
- [주요 기능](#주요-기능)
- [아키텍처](#아키텍처)
- [워크플로우](#워크플로우)
- [설치 및 실행](#설치-및-실행)
- [API 사용법](#api-사용법)
- [Worker 시스템](#worker-시스템)
- [프로젝트 구조](#프로젝트-구조)
- [설정 가이드](#설정-가이드)
- [서비스 관리](#서비스-관리)
- [개발 가이드](#개발-가이드)
- [문제 해결](#문제-해결)

## 프로젝트 개요

NewSum AI는 텍스트 기반 정보를 시각적 만화 콘텐츠로 변환하는 종합 AI 플랫폼입니다. 복잡한 뉴스나 정보를 쉽고 재미있는 만화로 재구성하여 정보 접근성을 높이고, 실시간 AI 채팅을 통해 사용자와 상호작용합니다.

### 핵심 가치
- **정보의 시각화**: 복잡한 텍스트를 친숙한 만화로 변환
- **AI 기반 자동화**: 검색부터 이미지 생성까지 완전 자동화
- **실시간 상호작용**: 즉시 응답하는 AI 채팅 시스템
- **확장 가능한 아키텍처**: 마이크로서비스 기반 모듈식 설계

## 주요 기능

### 🎨 **자동 만화 생성**
- **지능형 검색**: Google Search API를 통한 관련 정보 수집
- **AI 각색**: LLM 기반 창의적 만화 시나리오 생성
- **자동 이미지 생성**: Stable Diffusion 기반 만화 이미지 생성
- **다국어 지원**: Naver Papago API를 통한 번역

### 💬 **실시간 AI 채팅**
- **컨텍스트 인식**: 대화 맥락을 이해하는 지능형 응답
- **토큰 최적화**: 자동 대화 요약으로 효율적 메모리 관리
- **세션 관리**: PostgreSQL 기반 대화 이력 저장
- **큐 기반 처리**: Redis Queue를 통한 비동기 메시지 처리

### 🚀 **고성능 비동기 처리**
- **FastAPI**: 고성능 비동기 웹 프레임워크
- **Worker 시스템**: 분산 처리를 위한 독립적 Worker 프로세스
- **큐 시스템**: Redis 기반 작업 큐와 결과 캐싱
- **상태 추적**: 실시간 작업 진행 상황 모니터링

### 📊 **모니터링 & 관리**
- **LangSmith 통합**: 워크플로우 실행 추적 및 성능 분석
- **구조화된 로깅**: 중앙집중식 로그 관리
- **자동 서비스 관리**: 스크립트 기반 서비스 시작/중지/재시작

## 아키텍처

### 기술 스택
- **Backend**: FastAPI + Python 3.8+
- **워크플로우 엔진**: LangGraph
- **데이터베이스**: PostgreSQL (메인), Redis (캐시/큐)
- **AI 서비스**: 
  - LLM: vLLM + Llama3 (OpenAI 호환 API)
  - 이미지: Stable Diffusion WebUI API
- **외부 API**: Google Search, Naver Papago, AWS S3
- **모니터링**: LangSmith

### 시스템 아키텍처
```
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Application (Port 8000)             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │    API v1       │  │    API v2       │  │  Background     │  │
│  │ ┌─────────────┐ │  │ ┌─────────────┐ │  │   Tasks         │  │
│  │ │/comics      │ │  │ │/chat        │ │  │ ┌─────────────┐ │  │
│  │ │/llm/tasks   │ │  │ │(Async Chat) │ │  │ │ Workflow    │ │  │
│  │ │/status      │ │  │ └─────────────┘ │  │ │ Execution   │ │  │
│  │ └─────────────┘ │  └─────────────────┘  │ └─────────────┘ │  │
│  └─────────────────┘                       └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                        Worker System                            │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                 Chat Worker                                 │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │ │
│  │  │Chat Handler │  │Summary      │  │Qwen Tokenizer       │ │ │
│  │  │(Room-based) │  │Handler      │  │Support              │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                   LangGraph Workflow Engine                     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │        10-Node Sequential Pipeline                         │ │
│  │   N01→N02→N03→N04→N05→[N05-HITL]→N06→N06A→N07→N08→N09→N10  │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
               ┌────────────────▼────────────────┐
               │           Data Layer            │
               │  ┌─────────────┐ ┌────────────┐ │
               │  │PostgreSQL   │ │   Redis    │ │
               │  │(Chat,State) │ │(Queue,Cache)│ │
               │  └─────────────┘ └────────────┘ │
               └─────────────────────────────────┘
                                │
               ┌────────────────▼────────────────┐
               │         External APIs           │
               │  ┌─────────────┐ ┌────────────┐ │
               │  │vLLM Server  │ │Google API  │ │
               │  │Stable Diff  │ │Naver API   │ │
               │  │AWS S3       │ │LangSmith   │ │
               │  └─────────────┘ └────────────┘ │
               └─────────────────────────────────┘
```

## 워크플로우

LangGraph 기반 10단계 순차 워크플로우:

### 📋 **입력 및 분석 (N01-N02)**
1. **N01 - 초기화**: 워크플로우 상태 초기화 및 입력 검증
2. **N02 - 쿼리 분석**: 사용자 쿼리의 의도와 컨텍스트 분석

### 🔍 **정보 수집 (N03-N04)**
3. **N03 - 검색 계획**: 효과적인 검색 전략 수립
4. **N04 - 검색 실행**: Google Search를 통한 관련 정보 수집

### 📝 **콘텐츠 생성 (N05-N06A)**
5. **N05 - 보고서 생성**: 수집된 정보를 종합한 상세 보고서 작성
6. **N05 HITL - 품질 검토**: Human-in-the-Loop 품질 관리 (선택적)
7. **N06 - 보고서 저장**: HTML 형태로 보고서 저장 및 번역
8. **N06A - 맥락 요약**: 만화 제작을 위한 핵심 요약 생성

### 🎨 **만화 제작 (N07-N09)**
9. **N07 - 아이디어 생성**: 창의적 만화 컨셉 개발
10. **N08 - 시나리오 작성**: 상세한 장면별 시나리오 생성
11. **N09 - 이미지 생성**: Stable Diffusion을 통한 장면 이미지 생성

### 📤 **완료 처리 (N10)**
12. **N10 - 최종화**: S3 업로드 및 외부 시스템 알림

## 설치 및 실행

### 사전 요구사항
- **Python 3.8+**
- **PostgreSQL 12+**
- **Redis 6+**
- **외부 서비스**:
  - vLLM 서버 (Llama3 모델)
  - Stable Diffusion WebUI
  - Google Search API 키
  - Naver Papago API 키
  - AWS S3 (선택적)

### 설치 과정

1. **저장소 클론**
```bash
git clone <repository-url>
cd 17-team-4cut/ai
```

2. **Python 환경 설정**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

3. **환경 설정**
```bash
cp .env.sample .env
# .env 파일을 편집하여 필요한 API 키와 설정 입력
```

4. **데이터베이스 설정**
```bash
# PostgreSQL 데이터베이스 생성
createdb newsum

# 채팅 스키마 적용
psql -d newsum -f scripts/chat_schema.sql

# Redis 서버 실행 확인
redis-cli ping  # PONG 응답 확인
```

5. **디렉토리 생성**
```bash
mkdir -p generated_images results/reports logs
```

### 서비스 실행

**방법 1: 자동 스크립트 사용**
```bash
# 모든 서비스 시작
./scripts/start_services.sh

# 상태 확인
./scripts/status_services.sh
```

**방법 2: 수동 실행**
```bash
# 터미널 1: FastAPI 서버
python main.py

# 터미널 2: Chat Worker
python run_chat_worker.py

# 터미널 3: Unified LLM Worker (선택적)
python run_unified_llm_worker.py
```

**서비스 접속:**
- API 서버: http://localhost:8000
- API 문서: http://localhost:8000/swagger

## API 사용법

### 📋 **API 문서**
- **Swagger UI**: http://localhost:8000/swagger
- **OpenAPI 스키마**: http://localhost:8000/schema.json

### 🎨 **만화 생성 API (v1)**

**만화 생성 요청:**
```bash
POST /api/v1/comics
```

**요청 예시:**
```json
{
  "writer_id": "creative_writer",
  "data": {
    "query": "2024년 AI 기술 발전과 사회 변화",
    "site": {
      "news": ["techcrunch.com", "wired.com"],
      "research_paper": ["arxiv.org", "nature.com"],
      "community": ["reddit.com", "hacker-news.com"]
    }
  }
}
```

**응답:**
```json
{
  "comic_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "PENDING",
  "message": "만화 생성 작업이 수락되어 백그라운드에서 시작되었습니다."
}
```

**상태 조회:**
```bash
GET /api/v1/comics/status/{comic_id}
```

### 💬 **채팅 API (v2)**

**채팅 메시지 전송:**
```bash
POST /api/v2/chat
```

**요청 예시:**
```json
{
  "request_id": "req_12345",
  "room_id": "room_abc",
  "user_id": "user_123",
  "message": "AI 기술의 미래에 대해 어떻게 생각해?"
}
```

**응답:**
```json
{
  "message": "메시지가 성공적으로 접수되어 처리 대기 중입니다.",
  "request_id": "req_12345"
}
```

### 🔧 **LLM 작업 API (v1)**

**LLM 작업 생성:**
```bash
POST /api/v1/llm/tasks
```

**요청 예시:**
```json
{
  "prompt": "다음 텍스트를 요약해주세요: [긴 텍스트]",
  "max_tokens": 512,
  "temperature": 0.7,
  "model_name": "llama3"
}
```

### 상태 코드
- `202 Accepted`: 작업이 큐에 추가됨
- `200 OK`: 작업 완료
- `404 Not Found`: 작업 ID를 찾을 수 없음
- `500 Internal Server Error`: 서버 오류

## Worker 시스템

### Chat Worker (`run_chat_worker.py`)
```bash
python run_chat_worker.py
```

**주요 기능:**
- **채팅 처리**: `chat_task_queue`에서 메시지 처리
- **요약 생성**: 토큰 임계값 초과 시 자동 요약
- **토큰 관리**: Qwen Tokenizer를 통한 정확한 토큰 계산
- **세션 관리**: PostgreSQL 기반 대화 이력 저장

**처리 큐:**
- `chat_task_queue`: 실시간 채팅 메시지
- `llm_summary_queue`: 대화 요약 요청

### Unified LLM Worker (`run_unified_llm_worker.py`)
```bash
python run_unified_llm_worker.py
```

**처리 큐:**
- `llm_request_queue`: 일반 LLM 작업 (워크플로우)
- `llm_chat_queue`: 채팅 작업
- `llm_summary_queue`: 요약 작업

### Worker 특징
- **비동기 처리**: asyncio 기반 병렬 작업 처리
- **오류 복구**: 자동 재시도 및 Dead Letter Queue
- **성능 모니터링**: 처리량 및 응답 시간 추적
- **Graceful Shutdown**: SIGINT/SIGTERM 신호 처리
## 프로젝트 구조

```
ai/
├── app/
│   ├── api/                         # REST API 레이어
│   │   ├── v1/                      # API 버전 1
│   │   │   ├── endpoints.py         # 만화 생성 API
│   │   │   ├── llm_endpoints.py     # LLM 작업 API  
│   │   │   ├── schemas.py           # 요청/응답 스키마
│   │   │   └── background_tasks.py  # 백그라운드 작업
│   │   └── v2/                      # API 버전 2
│   │       ├── endpoints.py         # 채팅 API
│   │       └── schemas.py           # 채팅 스키마
│   ├── config/                      # 설정 관리
│   │   ├── settings.py              # 중앙집중식 설정
│   │   └── logging_config.yaml      # 로깅 설정
│   ├── nodes/                       # 워크플로우 노드 (N01-N10)
│   │   ├── n01_initialize_node.py   # 워크플로우 초기화
│   │   ├── n02_analyze_query_node.py # 쿼리 분석
│   │   ├── n03_understand_and_plan_node.py # 검색 계획
│   │   ├── n04_execute_search_node.py # 검색 실행
│   │   ├── n05_report_generation_node.py # 보고서 생성
│   │   ├── n05_hitl_review_node.py  # 품질 검토
│   │   ├── n06_save_report_node.py  # 보고서 저장
│   │   ├── n06a_contextual_summary_node.py # 맥락 요약
│   │   ├── n07_comic_ideation_node.py # 아이디어 생성
│   │   ├── n08_scenario_generation_node.py # 시나리오 작성
│   │   ├── n09_image_generation_node.py # 이미지 생성
│   │   └── n10_finalize_and_notify_node.py # 최종 처리
│   ├── services/                    # 핵심 서비스
│   │   ├── llm_service.py           # LLM 상호작용
│   │   ├── image_service.py         # 이미지 생성
│   │   ├── database_client.py       # Redis 클라이언트
│   │   ├── postgresql_service.py    # PostgreSQL 서비스
│   │   ├── backend_client.py        # 백엔드 API 클라이언트
│   │   ├── translation_service.py   # Naver Papago 번역
│   │   ├── storage_service.py       # AWS S3 스토리지
│   │   ├── langsmith_service.py     # LangSmith 모니터링
│   │   └── spam_service.py          # 스팸 필터링
│   ├── tools/                       # 외부 도구
│   │   ├── search/                  # 검색 도구
│   │   │   ├── Google_Search_tool.py # Google 검색
│   │   │   ├── naver.py             # Naver 검색
│   │   │   └── rss.py               # RSS 피드
│   │   ├── scraping/                # 웹 스크래핑
│   │   ├── social/                  # 소셜 미디어
│   │   └── trends/                  # 트렌드 분석
│   ├── utils/                       # 유틸리티
│   │   ├── logger.py                # 로깅 설정
│   │   └── debug.py                 # 디버깅 도구
│   ├── workers/                     # Worker 시스템
│   │   ├── chat_worker.py           # 채팅 워커
│   │   └── handlers/                # 작업 핸들러
│   │       ├── chat_handler.py      # 채팅 처리
│   │       ├── summary_handler.py   # 요약 처리
│   │       ├── tokenizer.json       # Qwen 토크나이저
│   │       └── tokenizer_config.json
│   ├── workflows/                   # 워크플로우 정의
│   │   ├── main_workflow.py         # 메인 워크플로우
│   │   ├── state.py                 # 상태 모델 v1
│   │   └── state_v2.py              # 상태 모델 v2
│   ├── dependencies.py              # FastAPI 의존성 주입
│   └── lifespan.py                  # 애플리케이션 라이프사이클
├── generated_images/                # 생성된 이미지 저장소
├── results/                         # 워크플로우 결과
│   └── reports/                     # HTML 보고서
├── logs/                            # 로그 파일
├── scripts/                         # 서비스 관리 스크립트
│   ├── chat_schema.sql              # PostgreSQL 스키마
│   ├── start_services.sh            # 서비스 시작
│   ├── stop_services.sh             # 서비스 중지
│   ├── restart_services.sh          # 서비스 재시작
│   └── status_services.sh           # 서비스 상태 확인
├── tests/                           # 테스트 코드
│   ├── test_llm_api.py
│   ├── test_llm_queue.py
│   ├── check_llm_system.py
│   └── postgre/                     # PostgreSQL 테스트
├── .env                             # 환경 변수
├── .env.sample                      # 환경 변수 템플릿
├── requirements.txt                 # Python 의존성
├── logging_config.yaml              # 로깅 설정
├── main.py                          # 애플리케이션 진입점
├── run_chat_worker.py               # 채팅 워커 실행
└── run_unified_llm_worker.py        # 통합 LLM 워커 실행
```

## 설정 가이드

### 필수 환경 변수

**.env 파일 주요 설정:**

**애플리케이션 기본 설정:**
```env
APP_NAME="NewSum AI Service"
APP_VERSION="0.1.0"
APP_HOST="0.0.0.0"
APP_PORT=8000
LOG_LEVEL="INFO"
```

**데이터베이스 설정:**
```env
# PostgreSQL (메인 데이터베이스)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=newsum_user
POSTGRES_PASSWORD=your_password
POSTGRES_DB=newsum
POSTGRES_MIN_CONNECTIONS=5
POSTGRES_MAX_CONNECTIONS=20

# Redis (캐시 및 큐)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=your_redis_password
```

**LLM 서비스 설정:**
```env
# vLLM OpenAI 호환 서버
LLM_API_ENDPOINT=http://localhost:8088/v1/chat/completions
LLM_API_TIMEOUT=60
LLM_API_RETRIES=3
DEFAULT_LLM_MODEL="llama3"
```

**이미지 생성 설정:**
```env
# Stable Diffusion WebUI API
IMAGE_SERVER_URL=http://localhost:7860/sdapi/v1/txt2img
IMAGE_STORAGE_PATH=generated_images
```

**외부 API 설정:**
```env
# Google Search API
GOOGLE_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GOOGLE_CSE_ID=cx_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Naver Papago API
NAVER_CLIENT_ID=your_naver_client_id
NAVER_CLIENT_SECRET=your_naver_client_secret

# AWS S3 (선택적)
S3_BUCKET_NAME=your-s3-bucket-name
AWS_REGION=ap-northeast-2
```

**채팅 시스템 설정:**
```env
# 채팅 성능 튜닝
CHAT_TOKEN_THRESHOLD=500
CHAT_RECENT_HISTORY_LIMIT=10
CHAT_MAX_MESSAGE_LENGTH=2000

# Worker 설정
LLM_WORKER_TIMEOUT=60
LLM_WORKER_MAX_EMPTY_CYCLES=5
LLM_WORKER_STATS_INTERVAL=50
```

**모니터링 설정:**
```env
# LangSmith (선택적)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LANGCHAIN_PROJECT=NewSum-Project
```

### 외부 서비스 설정

**1. vLLM 서버 설정**
```bash
# Llama3 모델로 vLLM 서버 실행
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --port 8088 \
    --host 0.0.0.0
```

**2. Stable Diffusion WebUI 설정**
```bash
# AUTOMATIC1111 WebUI API 모드로 실행
python launch.py --api --listen --port 7860
```

**3. Redis 서버**
```bash
# Redis 서버 실행
redis-server

# 또는 Docker 사용
docker run -d -p 6379:6379 redis:alpine
```

## 서비스 관리

### 자동화 스크립트

**모든 서비스 시작:**
```bash
./scripts/start_services.sh
```

**서비스 상태 확인:**
```bash
./scripts/status_services.sh
```

**서비스 재시작:**
```bash
./scripts/restart_services.sh
```

**모든 서비스 중지:**
```bash
./scripts/stop_services.sh
```

### 개별 서비스 관리

**FastAPI 서버:**
```bash
# 프로덕션 모드
python main.py

# 개발 모드
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Worker 프로세스:**
```bash
# 채팅 워커
python run_chat_worker.py

# 통합 LLM 워커
python run_unified_llm_worker.py
```

### 프로세스 모니터링

**실행 중인 서비스 확인:**
```bash
ps aux | grep python
netstat -tulpn | grep :8000
```

**Redis 큐 모니터링:**
```bash
# 큐 길이 확인
redis-cli LLEN chat_task_queue
redis-cli LLEN llm_request_queue

# 큐 내용 확인
redis-cli LRANGE chat_task_queue 0 -1
```

**로그 모니터링:**
```bash
# 실시간 로그 확인
tail -f logs/app.log

# 오류 로그만 확인
grep -i error logs/app.log

# 특정 워커 로그 확인
grep "ChatWorker" logs/app.log
```
## 개발 가이드

### 새로운 워크플로우 노드 추가

**1. 노드 클래스 생성:**
```python
# app/nodes/n11_new_feature_node.py
from app.workflows.state_v2 import WorkflowState
from app.utils.logger import get_logger
from typing import Dict, Any

class N11NewFeatureNode:
    def __init__(self, required_service):
        self.service = required_service
        self.logger = get_logger(__name__)
    
    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        """새로운 기능을 처리하는 노드"""
        try:
            # 노드 로직 구현
            result = await self.service.process(state.query)
            
            # 상태 업데이트
            return {
                "new_feature_section": {
                    "result": result,
                    "timestamp": "2024-05-27T10:00:00Z"
                }
            }
        except Exception as e:
            self.logger.error(f"N11 노드 실행 중 오류: {e}")
            raise
```

**2. 워크플로우에 통합:**
```python
# app/workflows/main_workflow.py에 추가
from app.nodes.n11_new_feature_node import N11NewFeatureNode

# 노드 인스턴스 생성
n11_new_feature = N11NewFeatureNode(required_service=some_service)

# 워크플로우에 노드 추가
workflow.add_node("n11_new_feature", n11_new_feature.run)

# 엣지 연결 (예: N10 → N11 → END)
workflow.add_edge("n10_finalize_and_notify", "n11_new_feature")
workflow.add_edge("n11_new_feature", END)
```

**3. 상태 모델 업데이트:**
```python
# app/workflows/state_v2.py에 새 섹션 추가
class NewFeatureSection(BaseModel):
    result: Optional[str] = None
    timestamp: Optional[str] = None

class WorkflowState(TypedDict):
    # 기존 필드들...
    new_feature_section: NewFeatureSection
```

### 새로운 API 엔드포인트 추가

**1. 스키마 정의:**
```python
# app/api/v1/schemas.py 또는 새로운 스키마 파일
class NewFeatureRequest(BaseModel):
    input_data: str = Field(..., description="입력 데이터")
    options: Optional[Dict[str, Any]] = None

class NewFeatureResponse(BaseModel):
    result_id: str
    status: str
    message: str
```

**2. 엔드포인트 구현:**
```python
# app/api/v1/endpoints.py 또는 새로운 엔드포인트 파일
@router.post("/new-feature")
async def create_new_feature(
    request: NewFeatureRequest,
    background_tasks: BackgroundTasks,
    db_client: DatabaseClientDep
):
    """새로운 기능 API 엔드포인트"""
    # 비동기 작업 생성
    task_id = str(uuid.uuid4())
    
    # 백그라운드 작업 추가
    background_tasks.add_task(
        process_new_feature,
        task_id=task_id,
        input_data=request.input_data,
        options=request.options
    )
    
    return NewFeatureResponse(
        result_id=task_id,
        status="PENDING",
        message="작업이 시작되었습니다."
    )
```

### 새로운 Worker 추가

**1. Worker 클래스 생성:**
```python
# app/workers/new_worker.py
import asyncio
from app.services.database_client import DatabaseClient
from app.utils.logger import get_logger

class NewWorker:
    def __init__(self):
        self.db_client = DatabaseClient()
        self.logger = get_logger(__name__)
        self.running = False
    
    async def start(self):
        """Worker 시작"""
        self.running = True
        self.logger.info("NewWorker started")
        
        while self.running:
            try:
                # 큐에서 작업 가져오기
                task = await self.db_client.brpop("new_task_queue", timeout=5)
                if task:
                    await self.process_task(task[1])
            except Exception as e:
                self.logger.error(f"Worker error: {e}")
                await asyncio.sleep(1)
    
    async def process_task(self, task_data):
        """작업 처리"""
        # 작업 로직 구현
        pass
    
    async def stop(self):
        """Worker 중지"""
        self.running = False
        await self.db_client.close()
```

**2. Worker 실행 스크립트:**
```python
# run_new_worker.py
import asyncio
from app.workers.new_worker import NewWorker

async def main():
    worker = NewWorker()
    try:
        await worker.start()
    except KeyboardInterrupt:
        await worker.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### 테스트 작성

**1. 단위 테스트:**
```python
# tests/test_new_feature.py
import pytest
from app.nodes.n11_new_feature_node import N11NewFeatureNode

@pytest.mark.asyncio
async def test_new_feature_node():
    # 테스트 설정
    mock_service = MockService()
    node = N11NewFeatureNode(mock_service)
    
    # 테스트 상태
    state = WorkflowState(query="test query")
    
    # 실행
    result = await node.run(state)
    
    # 검증
    assert "new_feature_section" in result
    assert result["new_feature_section"]["result"] is not None
```

**2. 통합 테스트:**
```python
# tests/test_api_integration.py
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_new_feature_api():
    response = client.post("/api/v1/new-feature", json={
        "input_data": "test data"
    })
    assert response.status_code == 202
    assert "result_id" in response.json()
```

**테스트 실행:**
```bash
# 전체 테스트
pytest

# 특정 테스트 파일
pytest tests/test_new_feature.py

# 커버리지 확인
pytest --cov=app tests/
```

## 문제 해결

### 일반적인 문제들

**1. 서버 시작 실패**

문제: `Connection refused` 오류
```bash
# 데이터베이스 연결 확인
pg_isready -h localhost -p 5432
redis-cli ping

# 포트 사용 확인
netstat -tulpn | grep :8000
lsof -i :8000

# 로그 확인
tail -f logs/app.log
```

문제: 환경 변수 로드 실패
```bash
# .env 파일 존재 확인
ls -la .env

# 환경 변수 테스트
python -c "from app.config.settings import settings; print(settings.POSTGRES_HOST)"
```

**2. Worker 연결 실패**

문제: Redis 큐 연결 안됨
```bash
# Redis 서버 상태 확인
systemctl status redis
# 또는
redis-cli info server

# 큐 상태 확인
redis-cli LLEN chat_task_queue
redis-cli LLEN llm_request_queue

# Worker 로그 확인
grep "Worker" logs/app.log
```

**3. LLM 서비스 오류**

문제: vLLM 서버 응답 없음
```bash
# vLLM 서버 상태 확인
curl http://localhost:8088/v1/models

# 모델 로드 상태 확인
curl http://localhost:8088/health
```

문제: 토큰 한도 초과
```bash
# 토큰 사용량 모니터링
grep "tokens" logs/app.log

# 요약 기능 작동 확인
redis-cli LLEN llm_summary_queue
```

**4. 이미지 생성 실패**

문제: Stable Diffusion API 오류
```bash
# WebUI API 상태 확인
curl http://localhost:7860/sdapi/v1/progress

# API 설정 확인
curl http://localhost:7860/sdapi/v1/options
```

**5. 데이터베이스 문제**

PostgreSQL 연결 오류:
```bash
# 연결 테스트
python tests/check_llm_system.py

# DB 스키마 확인
psql -d newsum -c "\dt"

# 테이블 데이터 확인
psql -d newsum -c "SELECT COUNT(*) FROM chat_messages;"
```

Redis 메모리 부족:
```bash
# Redis 메모리 사용량 확인
redis-cli info memory

# 만료 정책 확인
redis-cli config get maxmemory-policy
```

### 성능 최적화

**1. 동시성 조정**
```env
# 워커 성능 튜닝
LLM_WORKER_TIMEOUT=120
LLM_WORKER_MAX_EMPTY_CYCLES=10

# 데이터베이스 연결 풀
POSTGRES_MIN_CONNECTIONS=10
POSTGRES_MAX_CONNECTIONS=50

# Redis 성능
REDIS_DB=1  # 전용 DB 사용
```

**2. 토큰 최적화**
```env
# 채팅 토큰 관리
CHAT_TOKEN_THRESHOLD=300  # 더 자주 요약
CHAT_RECENT_HISTORY_LIMIT=5  # 최근 대화만 유지

# LLM 응답 제한
DEFAULT_MAX_TOKENS=512
```

**3. 캐싱 전략**
```python
# Redis 캐싱 활용
await redis_client.setex(
    f"llm_response:{hash(prompt)}", 
    3600,  # 1시간 캐시
    response_json
)
```

### 로그 분석

**주요 로그 위치:**
- `logs/app.log`: 메인 애플리케이션 로그
- `scripts/logs/`: 서비스 스크립트 로그

**유용한 로그 명령어:**
```bash
# 실시간 로그 모니터링
tail -f logs/app.log

# 에러 로그만 필터링
grep -i "error\|exception" logs/app.log

# 특정 시간 범위 로그
grep "2024-05-27 10:" logs/app.log

# 워커별 로그 분리
grep "ChatWorker" logs/app.log > chat_worker.log
grep "WorkflowExecution" logs/app.log > workflow.log

# 성능 메트릭 추출
grep "duration_seconds\|processing_time" logs/app.log

# API 요청 추적
grep "POST\|GET" logs/app.log | grep -E "comics|chat|llm"
```

### 모니터링 및 알림

**LangSmith 대시보드:**
- 워크플로우 실행 추적
- 성능 메트릭 분석
- 오류율 모니터링

**시스템 모니터링:**
```bash
# 시스템 리소스 확인
htop
iostat -x 1
free -h

# 디스크 사용량
df -h
du -sh generated_images/
du -sh logs/

# 네트워크 연결
netstat -an | grep :8000
ss -tulpn | grep :6379
```

## 기여 및 개발

### 코딩 스타일

**Python 코딩 규칙:**
- PEP 8 준수
- Type hints 사용 필수
- async/await 일관성 유지
- 에러 핸들링 명시적 구현

**예시:**
```python
from typing import Optional, Dict, Any
import asyncio

async def process_data(
    input_data: str,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """데이터 처리 함수
    
    Args:
        input_data: 입력 데이터
        options: 선택적 설정
        
    Returns:
        처리 결과 딕셔너리
        
    Raises:
        ValueError: 잘못된 입력 데이터
        ProcessingError: 처리 중 오류
    """
    if not input_data:
        raise ValueError("입력 데이터가 비어있습니다")
    
    try:
        # 비동기 처리 로직
        result = await some_async_operation(input_data)
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"데이터 처리 중 오류: {e}")
        raise ProcessingError(f"처리 실패: {e}")
```

### 배포 가이드

**Docker 배포 (권장):**
```dockerfile
# Dockerfile 예시
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py"]
```

**docker-compose.yml:**
```yaml
version: '3.8'
services:
  newsum-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - POSTGRES_HOST=postgres
      - REDIS_HOST=redis
    depends_on:
      - postgres
      - redis
  
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: newsum
      POSTGRES_USER: newsum_user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

---

**프로젝트 정보:**
- **버전**: v1.0.0
- **최종 업데이트**: 2024-05-27
- **Python 버전**: 3.8+
- **FastAPI 버전**: 0.115+
- **주요 의존성**: LangGraph, PostgreSQL, Redis, vLLM

**문의 및 지원:**
- GitHub Issues: 버그 리포트 및 기능 요청
- 문서: 프로젝트 Wiki 참조
- 라이선스: [LICENSE 파일 참조]