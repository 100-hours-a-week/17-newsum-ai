# NewSum AI 서비스

NewSum AI 서비스는 뉴스 기사나 사용자 의견을 기반으로 만화를 생성하는 FastAPI 기반 애플리케이션입니다. 쿼리를 처리하고, 콘텐츠를 분석하며, 창의적인 시나리오와 이미지를 생성하기 위해 LangGraph 워크플로우를 활용합니다.

## 목차

- [프로젝트 개요](#프로젝트-개요)
- [아키텍처](#아키텍처)
- [주요 기능](#주요-기능)
- [워크플로우](#워크플로우)
- [설치 방법](#설치-방법)
- [사용 방법](#사용-방법)
- [API 엔드포인트](#api-엔드포인트)
- [설정](#설정)
- [개발 가이드](#개발-가이드)
- [라이선스](#라이선스)

## 프로젝트 개요

NewSum AI 서비스는 사용자 쿼리나 뉴스 콘텐츠를 일련의 AI 기반 프로세스를 통해 흥미로운 만화 스트립으로 변환합니다. 이 서비스는 모듈식 아키텍처와 워크플로우 관리를 위한 LangGraph를 사용하여 언어 모델, 검색 도구, 이미지 생성 등 다양한 AI 서비스를 원활하게 통합합니다.

본 프로젝트의 핵심 목표는 텍스트 기반 정보를 시각적인 콘텐츠로 변환하여 정보의 접근성과 이해도를 높이는 것입니다. 특히 뉴스나 시사적인 내용을 만화라는 친숙한 형태로 재구성함으로써, 복잡한 정보도 쉽게 소화할 수 있도록 돕습니다.

## 아키텍처

시스템은 다음과 같은 구성 요소를 가진 모듈식 아키텍처로 구축되었습니다:

1. **FastAPI 애플리케이션**: 만화 생성 및 상태 확인을 위한 REST API 엔드포인트 제공
   - 비동기 처리를 통한 높은 성능 제공
   - 스웨거(Swagger) UI를 통한 API 문서 자동화

2. **LangGraph 워크플로우**: 쿼리 분석부터 이미지 생성까지 순차적인 처리 노드 조정
   - 명확한 단계별 처리로 복잡한 AI 워크플로우 관리
   - 각 노드 간 데이터 흐름 최적화

3. **서비스 구성 요소**:
   - **LLM 서비스**: 언어 모델 상호 작용 처리
     - 프롬프트 엔지니어링을 통한 최적화된 결과 생성
     - 다양한 LLM 백엔드 지원 가능
   - **이미지 서비스**: 외부 API를 통한 이미지 생성 관리
     - 텍스트 프롬프트 기반 이미지 생성
     - 생성된 이미지의 로컬 저장 및 관리
   - **Google 검색 도구**: 연구를 위한 웹 검색 수행
     - 사용자 지정 검색 사이트 설정 지원
     - 효율적인 정보 수집 알고리즘
   - **데이터베이스 클라이언트**: 워크플로우 상태 및 결과 저장
     - 워크플로우 진행 상황 실시간 추적
     - 결과물의 영구 저장소 역할
   - **추가 유틸리티 서비스**:
     - **번역 서비스**: 다국어 지원
     - **스팸 감지 서비스**: 부적절한 콘텐츠 필터링
     - **스토리지 서비스**: 파일 관리
     - **LangSmith 서비스**: 워크플로우 모니터링

## 주요 기능

- **비동기 처리**: 상태 업데이트와 함께 논블로킹 워크플로우 실행
  - FastAPI의 비동기 기능을 활용한 효율적인 리소스 사용
  - 클라이언트에 실시간에 가까운 피드백 제공

- **순차적 노드 처리**: 개별 단계를 처리하는 9개의 특수 처리 노드
  - 각 노드는 독립적으로 작동하며 명확한 책임 구분
  - 모듈식 설계로 유지보수 및 확장성 향상

- **백그라운드 작업 관리**: 장시간 실행 작업을 관리하기 위한 FastAPI 백그라운드 작업
  - 사용자 요청 처리와 리소스 집약적 작업의 분리
  - 시스템 안정성 향상 및 사용자 경험 개선

- **강력한 오류 처리**: 워크플로우 전반에 걸친 포괄적인 오류 관리 및 로깅
  - 상세한 오류 추적 및 문제 진단 용이
  - 다양한 실패 상황에 대한 체계적인 대응 메커니즘

- **유연한 설정**: 모든 서비스에 대한 환경 기반 구성
  - 개발, 테스트, 프로덕션 환경 간 쉬운 전환
  - 민감한 정보를 환경 변수로 관리하여 보안 강화

## 워크플로우

주요 워크플로우는 9개의 순차적 노드로 구성됩니다:

1. **초기화 (N01InitializeNode)**: 
   - 워크플로우 상태 설정 및 쿼리 유효성 검사
   - 고유 ID 생성 및 타임스탬프 설정
   - 기본 구성 매개변수 로드

2. **쿼리 분석 (N02AnalyzeQueryNode)**: 
   - 사용자 쿼리 심층 분석 및 컨텍스트 설정
   - 주제 분류 및 키워드 추출
   - 초기 컨텍스트 정보 수집

3. **이해 및 계획 (N03UnderstandAndPlanNode)**: 
   - 쿼리 기반 검색 전략 수립
   - 필요한 정보 유형 식별
   - 최적의 정보 소스 선정

4. **검색 실행 (N04ExecuteSearchNode)**: 
   - Google 검색 도구를 사용한 웹 검색 수행
   - 사용자 지정 사이트 기본 설정 활용
   - 검색 결과 수집 및 구조화

5. **보고서 생성 (N05ReportGenerationNode)**: 
   - 검색 결과로부터 종합 보고서 작성
   - 정보 통합 및 요약
   - HTML 형식의 구조화된 문서 생성

6. **보고서 저장 (N06SaveReportNode)**: 
   - 생성된 보고서를 향후 참조를 위해 저장
   - 파일 시스템에 문서 보관
   - 접근 경로 기록

7. **만화 아이디어 생성 (N07ComicIdeationNode)**: 
   - 보고서 기반 창의적인 만화 아이디어 생성
   - 다양한 스토리 옵션 제안
   - 가장 적합한 아이디어 선정

8. **시나리오 생성 (N08ScenarioGenerationNode)**: 
   - 선택된 아이디어를 바탕으로 상세 시나리오 개발
   - 장면별 설명 및 대화 스크립트 작성
   - 시각적 지침 포함

9. **이미지 생성 (N09ImageGenerationNode)**: 
   - 시나리오의 각 장면에 대한 이미지 생성
   - 외부 이미지 생성 API 활용
   - 생성된 이미지 저장 및 관리

## 설치 방법

### 사전 요구 사항

- Python 3.8 이상
- FastAPI
- LangGraph
- 필요한 환경 변수 (`.env.sample` 참조)
- 외부 API 접근 권한 (이미지 생성, 검색 등)

### 설정

1. 저장소 복제:
   ```bash
   git clone https://github.com/your-username/17-team-4cut.git
   cd 17-team-4cut/ai
   ```

2. 가상 환경 생성:
   ```bash
   python -m venv venv
   ```

3. 가상 환경 활성화:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Linux/macOS:
     ```bash
     source venv/bin/activate
     ```

4. 의존성 설치:
   ```bash
   pip install -r requirements.txt
   ```

5. 환경 변수 템플릿 복사 및 설정:
   ```bash
   cp .env.sample .env
   # .env 파일을 편집하여 필요한 설정 값 입력
   ```

6. 필요한 디렉토리 구조 확인:
   ```bash
   mkdir -p results/reports
   mkdir -p generated_images
   ```

## 사용 방법

### 서버 실행

FastAPI 서버 시작:

```bash
python main.py
```

기본적으로 서버는 http://0.0.0.0:8090 에서 실행됩니다.

### 개발 모드 실행

개발 중에는 코드 변경 시 자동 재시작 기능을 활용할 수 있습니다:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8090
```

### API 문서 접근

Swagger UI로 API 문서에 접근:

```
http://localhost:8090/docs
```

ReDoc으로 API 문서에 접근:

```
http://localhost:8090/redoc
```

## API 엔드포인트

### 만화 생성

```
POST /api/v1/comics
```

요청 본문 예시:
```json
{
  "writer_id": "writer1",
  "data": {
    "query": "기후 변화가 해안 도시에 미치는 영향",
    "site": {
      "news": ["bbc.com", "cnn.com", "yonhap.co.kr"],
      "research_paper": ["nature.com", "science.org"],
      "deep_dive_tech": ["medium.com", "towardsdatascience.com"]
    }
  }
}
```

응답 예시:
```json
{
  "comic_id": "9d8be988-833e-4500-8e38-8d45ca150449",
  "status": "PENDING",
  "message": "만화 생성 작업이 수락되어 백그라운드에서 시작되었습니다."
}
```

#### 매개변수 설명:

- `writer_id` (선택사항): 특정 AI 작가 스타일을 지정할 ID
- `data.query`: 만화의 주제나 내용을 설명하는 쿼리 텍스트
- `data.site` (선택사항): 카테고리별 검색 사이트 기본 설정
  - `news`: 뉴스 사이트 목록
  - `research_paper`: 연구 논문 사이트 목록
  - `deep_dive_tech`: 기술 분석 사이트 목록
  - `community`: 커뮤니티 사이트 목록

### 만화 상태 확인

```
GET /api/v1/comics/status/{comic_id}
```

응답 예시:
```json
{
  "comic_id": "9d8be988-833e-4500-8e38-8d45ca150449",
  "status": "DONE",
  "message": "워크플로우 성공적으로 완료됨.",
  "query": "기후 변화가 해안 도시에 미치는 영향",
  "writer_id": "writer1",
  "user_site_preferences_provided": true,
  "timestamp_accepted": "2024-05-07T08:30:00.000Z",
  "timestamp_start": "2024-05-07T08:30:01.123Z",
  "timestamp_end": "2024-05-07T08:32:45.678Z",
  "duration_seconds": 164.555,
  "result": {
    "trace_id": "9d8be988-833e-4500-8e38-8d45ca150449",
    "final_stage": "n09_image_generation",
    "original_query": "기후 변화가 해안 도시에 미치는 영향",
    "report_content_length": 8750,
    "saved_report_path": "results/reports/report_9d8be988.html",
    "comic_ideas_count": 3,
    "comic_ideas_titles": ["상승하는 파도", "해안 수호자들", "도시 적응 프로젝트"],
    "selected_comic_idea_title": "상승하는 파도",
    "comic_scenarios_count": 1,
    "scenario_scenes_approx": 4,
    "generated_comic_images_count": 4,
    "generated_images_summary": [
      "Scene 1: /path/to/image1.png",
      "Scene 2: /path/to/image2.png",
      "Scene 3: /path/to/image3.png",
      "Scene 4: /path/to/image4.png"
    ]
  }
}
```

#### 상태 코드 설명:

- `PENDING`: 요청이 수락되었으나 아직 처리 시작 전
- `STARTED`: 워크플로우 처리 시작됨
- `DONE`: 워크플로우가 성공적으로 완료됨
- `FAILED`: 처리 중 오류 발생
- `UNKNOWN`: 상태를 결정할 수 없음

## 설정

애플리케이션은 환경 변수를 사용하여 구성됩니다. 필요한 변수의 예시는 `.env.sample`을 참조하세요.

### 주요 설정 매개변수:

#### 이미지 생성 관련:
- `IMAGE_SERVER_URL`: 이미지 생성 서비스 URL
- `IMAGE_SERVER_API_TOKEN`: 이미지 서비스 인증 토큰
- `IMAGE_STORAGE_PATH`: 생성된 이미지 저장 로컬 경로

#### LLM 서비스 관련:
- `LLM_API_KEY`: 언어 모델 API 키
- `LLM_MODEL_NAME`: 사용할 모델 이름
- `LLM_TEMPERATURE`: 출력 다양성 조절 값
- `LLM_MAX_TOKENS`: 최대 토큰 수

#### 검색 도구 관련:
- `SEARCH_API_KEY`: 검색 API 키
- `SEARCH_ENGINE_ID`: 커스텀 검색 엔진 ID
- `MAX_SEARCH_RESULTS`: 검색 결과 최대 개수

#### 데이터베이스 관련:
- `DB_CONNECTION_STRING`: 데이터베이스 연결 문자열
- `DB_NAME`: 데이터베이스 이름
- `DB_TIMEOUT`: DB 작업 타임아웃 값

#### 로깅 관련:
- `LOG_LEVEL`: 로그 레벨 설정
- `LOG_CONFIG_PATH`: 로깅 설정 파일 경로

#### 서버 관련:
- `APP_HOST`: 서버 호스트 주소
- `APP_PORT`: 서버 포트 번호

### 로깅 설정

로깅은 `logging_config.yaml` 파일을 통해 구성됩니다. 다음 기능을 제공합니다:

- 다양한 로그 수준 지원 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- 파일 및 콘솔 로깅 동시 지원
- 로테이션 파일 핸들러로 로그 파일 관리
- 구조화된 로그 포맷팅 (타임스탬프, 레벨, 모듈, 메시지)

## 개발 가이드

프로젝트 구조는 모듈식 접근 방식을 따릅니다:

```
ai/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── endpoints.py       # API 라우트 정의
│   │       ├── schemas.py         # 요청/응답용 Pydantic 모델
│   │       └── background_tasks.py # 백그라운드 작업 처리
│   ├── config/
│   │   └── settings.py            # 애플리케이션 설정
│   ├── nodes/
│   │   ├── n01_initialize_node.py # 워크플로우 노드 1
│   │   ├── n02_analyze_query_node.py
│   │   └── ... (노드 3-9)
│   ├── services/
│   │   ├── database_client.py     # 데이터베이스 상호작용
│   │   ├── image_service.py       # 이미지 생성 서비스
│   │   ├── llm_service.py         # 언어 모델 서비스
│   │   └── ... (기타 서비스)
│   ├── tools/
│   │   └── search/
│   │       └── Google_Search_tool.py # 구글 검색 기능
│   ├── utils/
│   │   └── logger.py              # 로깅 설정
│   └── workflows/
│       ├── main_workflow.py       # LangGraph 워크플로우 정의
│       └── state.py               # 워크플로우 상태 구조
├── generated_images/              # 생성된 이미지 저장소
├── results/                       # 보고서 및 기타 출력물 저장소
├── .env                           # 환경 변수
├── .env.sample                    # 환경 변수 템플릿
├── logging_config.yaml            # 로깅 설정
└── main.py                        # 애플리케이션 진입점
```

### 새 기능 개발 가이드

#### 새로운 노드 추가:

1. `app/nodes/` 디렉토리에 새 노드 파일 생성 (예: `n10_new_feature_node.py`)
2. 기존 노드 구조를 따라 클래스 정의:
   ```python
   class N10NewFeatureNode:
       def __init__(self, required_service):
           self.service = required_service
           self.logger = get_logger(__name__)
           
       async def run(self, state: WorkflowState) -> Dict[str, Any]:
           # 노드 로직 구현
           return updated_state_dict
   ```
3. `app/workflows/main_workflow.py`에 노드 추가:
   ```python
   from app.nodes.n10_new_feature_node import N10NewFeatureNode
   
   # 노드 인스턴스 생성
   n10_new_feature = N10NewFeatureNode(some_service)
   
   # 워크플로우에 노드 추가
   workflow.add_node("n10_new_feature", n10_new_feature.run)
   
   # 엣지 연결 업데이트
   workflow.add_edge("n09_image_generation", "n10_new_feature")
   workflow.add_edge("n10_new_feature", END)
   ```
4. `app/workflows/state.py`에 새 상태 필드 추가

#### 새로운 서비스 추가:

1. `app/services/` 디렉토리에 새 서비스 파일 생성
2. 서비스 클래스 구현
3. `app/lifespan.py`에 서비스 초기화 코드 추가
4. 필요한 환경 변수를 `.env.sample`에 정의

### 테스트 가이드

단위 테스트:
```bash
pytest app/tests/unit/
```

통합 테스트:
```bash
pytest app/tests/integration/
```

전체 테스트:
```bash
pytest
```

## 라이선스

[적절한 라이선스 정보 추가]

## 문제 해결

흔히 발생하는 문제와 해결책:

1. **환경 변수 로드 실패**: `.env` 파일이 올바른 형식인지 확인하고 필수 값이 모두 설정되었는지 확인하세요.

2. **API 연결 오류**: 이미지 생성 서비스나 검색 API의 엔드포인트와 인증 정보가 올바른지 확인하세요.

3. **워크플로우 실행 문제**: 로그 파일을 확인하여 어떤 노드에서 오류가 발생했는지 파악하세요. 노드별 상태 전이를 추적할 수 있습니다.

4. **메모리 사용량 문제**: 대량의 이미지 생성 시 메모리 사용량을 모니터링하고 필요시 배치 처리를 구현하세요.

## 연락처 및 기여

이 프로젝트에 대한 질문이나 제안사항이 있으시면 다음으로 연락해주세요:
- 이메일: [이메일 주소]
- 이슈 트래커: [GitHub 이슈 링크]

## 업데이트 기록

**V0.1.0** (2024-05-07)
- 초기 버전 릴리스
- 9개 노드 워크플로우 구현
- 비동기 API 엔드포인트 구현
