# NewSum AI 서비스

NewSum AI 서비스는 뉴스 기사나 사용자 의견을 기반으로 만화를 생성하는 FastAPI 기반 애플리케이션입니다. LangGraph 워크플로우를 활용하여 쿼리 분석부터 이미지 생성까지의 전체 과정을 자동화합니다.

## 목차

- [프로젝트 개요](#프로젝트-개요)
- [주요 특징](#주요-특징)
- [아키텍처](#아키텍처)
- [워크플로우](#워크플로우)
- [설치 및 실행](#설치-및-실행)
- [API 사용법](#api-사용법)
- [프로젝트 구조](#프로젝트-구조)
- [개발 가이드](#개발-가이드)
- [설정 가이드](#설정-가이드)
- [문제 해결](#문제-해결)

## 프로젝트 개요

NewSum AI는 텍스트 기반 정보를 시각적 콘텐츠로 변환하는 AI 기반 서비스입니다. 복잡한 뉴스나 시사 내용을 친숙한 만화 형태로 재구성하여 정보의 접근성과 이해도를 높입니다.

### 핵심 기능
- **자동 콘텐츠 분석**: 사용자 쿼리를 분석하여 관련 정보를 수집
- **인텔리전트 검색**: Google Custom Search를 활용한 맞춤형 정보 수집
- **창의적 각색**: AI 기반 만화 아이디어 및 시나리오 생성
- **자동 이미지 생성**: 시나리오 기반 만화 이미지 생성
- **다국어 지원**: 자동 번역 및 다국어 콘텐츠 처리

## 주요 특징

### 🚀 **비동기 처리**
- FastAPI 기반 고성능 비동기 API
- 백그라운드 작업을 통한 논블로킹 워크플로우
- 실시간 상태 추적 및 진행률 확인

### 🧠 **지능형 워크플로우**
- LangGraph 기반 11단계 순차 처리
- 각 노드별 독립적 오류 처리
- Human-in-the-Loop(HITL) 품질 관리

### 🔧 **모듈형 아키텍처**
- 서비스별 독립적 관리
- 손쉬운 확장 및 커스터마이징
- 환경별 설정 분리

### 📊 **모니터링 & 로깅**
- LangSmith 통합 모니터링
- 구조화된 로깅 시스템
- 상세한 오류 추적

## 아키텍처

### 기술 스택
- **Backend**: FastAPI + Python 3.8+
- **워크플로우**: LangGraph
- **데이터베이스**: Redis (상태 관리)
- **AI 서비스**: LLM (Llama 3), 이미지 생성 API
- **검색**: Google Custom Search API
- **스토리지**: 로컬 파일 시스템 + AWS S3
- **모니터링**: LangSmith

### 시스템 구성도
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │  LangGraph       │    │   External      │
│                 │    │  Workflow        │    │   Services      │
│ ┌─────────────┐ │    │                  │    │                 │
│ │ REST API    │ │◄──►│ ┌──────────────┐ │◄──►│ • LLM API       │
│ │ Endpoints   │ │    │ │ 11 Nodes     │ │    │ • Image Gen API │
│ └─────────────┘ │    │ │ Sequential   │ │    │ • Google Search │
│                 │    │ │ Processing   │ │    │ • Translation   │
│ ┌─────────────┐ │    │ └──────────────┘ │    │ • AWS S3        │
│ │ Background  │ │    │                  │    │                 │
│ │ Tasks       │ │    │ ┌──────────────┐ │    └─────────────────┘
│ └─────────────┘ │    │ │ State        │ │
└─────────────────┘    │ │ Management   │ │    ┌─────────────────┐
                       │ └──────────────┘ │    │   Storage       │
┌─────────────────┐    └──────────────────┘    │                 │
│   Redis DB      │◄───────────────────────────►│ • Local Files   │
│                 │                             │ • Generated     │
│ • Workflow      │                             │   Images        │
│   States        │                             │ • Reports       │
│ • Results       │                             │ • S3 Uploads    │
└─────────────────┘                             └─────────────────┘
```

## 워크플로우

전체 워크플로우는 11개의 순차적 노드로 구성됩니다:

### 📥 **입력 처리**
1. **N01 - 초기화**: 워크플로우 상태 설정 및 쿼리 유효성 검사
2. **N02 - 쿼리 분석**: 사용자 쿼리 심층 분석 및 컨텍스트 설정

### 🔍 **정보 수집**
3. **N03 - 검색 계획**: 쿼리 기반 검색 전략 수립
4. **N04 - 검색 실행**: Google 검색 도구를 사용한 웹 검색 수행

### 📝 **콘텐츠 생성**
5. **N05 - 보고서 생성**: 검색 결과 기반 종합 보고서 작성
6. **N05 HITL - 품질 검토**: Human-in-the-Loop 품질 관리 (선택적)
7. **N06 - 보고서 저장**: 생성된 보고서 저장 및 번역
8. **N06A - 문맥 요약**: 보고서 기반 핵심 요약 생성

### 🎨 **만화 제작**
9. **N07 - 아이디어 생성**: 창의적인 만화 아이디어 개발
10. **N08 - 시나리오 작성**: 선택된 아이디어 기반 상세 시나리오 생성
11. **N09 - 이미지 생성**: 시나리오의 각 장면 이미지 생성

### 📤 **결과 처리**
12. **N10 - 완료 및 알림**: S3 업로드 및 외부 시스템 알림

## 설치 및 실행

### 사전 요구사항
- Python 3.8 이상
- Redis 서버
- 외부 API 접근 권한:
  - LLM API (Llama 3 등)
  - 이미지 생성 API
  - Google Custom Search API
  - AWS S3 (선택적)

### 설치 단계

1. **저장소 복제**
```bash
git clone https://github.com/your-username/17-team-4cut.git
cd 17-team-4cut/ai
```

2. **가상 환경 설정**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

3. **의존성 설치**
```bash
pip install -r requirements.txt
```

4. **환경 변수 설정**
```bash
cp .env.sample .env
```
`.env` 파일을 편집하여 필요한 API 키와 설정값을 입력하세요.

5. **Redis 서버 확인**
```bash
# Redis가 실행 중인지 확인
redis-cli ping
# 응답: PONG
```

6. **디렉토리 구조 생성**
```bash
mkdir -p generated_images results/reports
```

### 서버 실행

**프로덕션 모드:**
```bash
python main.py
```

**개발 모드 (자동 재시작):**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8090
```

서버는 `http://localhost:8090`에서 실행됩니다.

## API 사용법

### 📋 **API 문서 접근**
- **Swagger UI**: http://localhost:8090/swagger
- **스키마**: http://localhost:8090/schema.json

### 🎬 **만화 생성 요청**

**엔드포인트:** `POST /api/v1/comics`

**요청 예시:**
```json
{
  "writer_id": "creative_writer",
  "data": {
    "query": "기후 변화가 해안 도시에 미치는 영향에 대한 최신 연구",
    "site": {
      "news": ["bbc.com", "cnn.com", "yonhap.co.kr"],
      "research_paper": ["nature.com", "science.org"],
      "deep_dive_tech": ["medium.com", "towardsdatascience.com"]
    }
  }
}
```

**응답 예시:**
```json
{
  "comic_id": "uuid-string",
  "status": "PENDING",
  "message": "만화 생성 작업이 수락되어 백그라운드에서 시작되었습니다."
}
```

### 📊 **상태 조회**

**엔드포인트:** `GET /api/v1/comics/status/{comic_id}`

**응답 예시:**
```json
{
  "comic_id": "uuid-string",
  "status": "DONE",
  "message": "워크플로우 성공적으로 완료됨",
  "query": "기후 변화가 해안 도시에 미치는 영향",
  "duration_seconds": 164.5,
  "result": {
    "trace_id": "uuid-string",
    "final_stage": "n09_image_generation",
    "report_len": 8750,
    "saved_report_path": "results/reports/report_uuid.html",
    "ideas_cnt": 3,
    "scenarios_cnt": 1,
    "images_cnt": 4
  }
}
```

### 🔄 **상태 코드**
- `PENDING`: 요청 수락, 처리 대기 중
- `STARTED`: 워크플로우 실행 시작
- `DONE`: 성공적으로 완료
- `FAILED`: 처리 중 오류 발생

## 프로젝트 구조

```
ai/
├── app/
│   ├── api/v1/                  # REST API 정의
│   │   ├── endpoints.py         # API 라우트
│   │   ├── schemas.py           # 요청/응답 모델
│   │   └── background_tasks.py  # 백그라운드 작업
│   ├── config/                  # 설정 관리
│   │   ├── settings.py          # 애플리케이션 설정
│   │   └── logging_config.yaml  # 로깅 설정
│   ├── nodes/                   # 워크플로우 노드
│   │   ├── n01_initialize_node.py
│   │   ├── n02_analyze_query_node.py
│   │   ├── ...                  # N03~N10
│   │   └── n05_hitl_review_node.py
│   ├── services/                # 핵심 서비스
│   │   ├── llm_service.py       # LLM 상호작용
│   │   ├── image_service.py     # 이미지 생성
│   │   ├── database_client.py   # Redis 클라이언트
│   │   ├── translation_service.py
│   │   ├── storage_service.py   # S3 업로드
│   │   └── spam_service.py      # 콘텐츠 필터링
│   ├── tools/                   # 외부 도구
│   │   ├── search/              # 검색 도구
│   │   ├── scraping/            # 웹 스크래핑
│   │   ├── social/              # 소셜 미디어
│   │   └── trends/              # 트렌드 분석
│   ├── utils/                   # 유틸리티
│   │   ├── logger.py            # 로깅 설정
│   │   └── debug.py             # 디버깅 도구
│   └── workflows/               # 워크플로우 정의
│       ├── main_workflow.py     # 메인 워크플로우
│       ├── state.py             # 상태 모델 (v1)
│       └── state_v2.py          # 상태 모델 (v2)
├── generated_images/            # 생성된 이미지
├── results/                     # 보고서 및 결과
│   └── reports/                 # HTML 보고서
├── .env                         # 환경 변수
├── .env.sample                  # 환경 변수 템플릿
├── requirements.txt             # Python 의존성
├── logging_config.yaml          # 로깅 설정
└── main.py                      # 애플리케이션 진입점
```

## 개발 가이드

### 🔧 **새로운 노드 추가**

1. **노드 클래스 생성**
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
        # 노드 로직 구현
        return {"new_section": {"new_data": "value"}}
```


3. **상태 모델 업데이트**
```python
# app/workflows/state_v2.py에 새 섹션 추가
```

### 🔌 **새로운 서비스 추가**

1. **서비스 클래스 구현**
```python
# app/services/new_service.py
class NewService:
    def __init__(self):
        # 초기화 로직
        pass
    
    async def process(self, data):
        # 서비스 로직
        return result
    
    async def close(self):
        # 리소스 정리
        pass
```

2. **의존성 등록**
```python
# app/dependencies.py에 의존성 함수 추가
# app/lifespan.py에 초기화 코드 추가
```

### 🧪 **테스트 실행**

```bash
# 전체 테스트
pytest

# 특정 테스트
pytest app/tests/test_specific.py

# 커버리지 확인
pytest --cov=app
```

### 🐛 **디버깅**

**로그 레벨 조정:**
```python
# .env 파일에서
LOG_LEVEL=DEBUG
```

**워크플로우 상태 확인:**
```python
from app.utils.debug import print_final_state_debug
# 워크플로우 실행 후 상태 출력
```

## 설정 가이드

### 필수 환경 변수

**LLM 서비스:**
```env
LLM_API_ENDPOINT=http://your-llm-api.com
LLM_API_TIMEOUT=60
DEFAULT_LLM_MODEL=llama3-8b
```

**이미지 생성:**
```env
IMAGE_SERVER_URL=http://your-image-api.com
IMAGE_SERVER_API_TOKEN=your-token
IMAGE_STORAGE_PATH=generated_images
```

**검색 도구:**
```env
GOOGLE_API_KEY=your-google-api-key
GOOGLE_CSE_ID=your-custom-search-engine-id
```

**데이터베이스:**
```env
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your-password
```

**스토리지 (선택적):**
```env
S3_BUCKET_NAME=your-bucket
AWS_REGION=us-east-1
ACCESS_KEY=your-access-key
SECRET_KEY=your-secret-key
```

**모니터링:**
```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_PROJECT=NewSum-Project
```

### 로깅 설정

`logging_config.yaml`을 통해 로깅 레벨과 출력 형식을 조정할 수 있습니다:

```yaml
version: 1
handlers:
  console:
    class: logging.StreamHandler
    level: INFO
  file:
    class: logging.handlers.RotatingFileHandler
    filename: app.log
    maxBytes: 10485760  # 10MB
    backupCount: 5
```

## 문제 해결

### 일반적인 문제들

**1. 서버 시작 실패**
```bash
# Redis 연결 확인
redis-cli ping

# 환경 변수 확인
python -c "from app.config.settings import settings; print(settings.REDIS_HOST)"
```

**2. API 호출 오류**
- API 키가 올바른지 확인
- 엔드포인트 URL이 정확한지 확인
- 네트워크 연결 상태 확인

**3. 워크플로우 실행 실패**
```bash
# 로그 파일 확인
tail -f app.log

# 특정 노드 로그 필터링
grep "N05" app.log
```

**4. 메모리 부족**
- 이미지 생성 병렬 처리 수 조정: `IMAGE_MAX_PARALLEL_TASKS=1`
- 배치 크기 조정: `TRANSLATION_BATCH_SIZE=30`

### 성능 최적화

**Redis 최적화:**
```env
REDIS_DB=0
# 적절한 Redis 메모리 정책 설정
```

**타임아웃 조정:**
```env
LLM_API_TIMEOUT=120
EXTERNAL_API_TIMEOUT_SECONDS=60
```

**병렬 처리:**
```env
IMAGE_MAX_PARALLEL_TASKS=2
S3_MAX_PARALLEL_UPLOADS=3
```

## 기여 및 연락처

### 버그 리포트 및 기능 요청
- GitHub Issues를 통해 버그 리포트나 기능 요청을 남겨주세요
- 명확한 재현 단계와 예상 결과를 포함해주세요

### 개발 기여
1. Fork 후 feature 브랜치 생성
2. 코드 스타일 가이드 준수
3. 테스트 코드 작성
4. Pull Request 생성

### 라이선스
이 프로젝트는 [라이선스명]에 따라 배포됩니다.

---

**버전**: v0.2.0  
**최종 업데이트**: 2024-05-24  
**Python 버전**: 3.8+  
**FastAPI 버전**: 0.115+
