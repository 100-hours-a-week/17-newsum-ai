# NewSum AI Service

## 프로젝트 개요

NewSum AI Service는 뉴스 및 의견 기반 만화 생성을 위한 AI 서비스입니다. LangGraph를 활용한 워크플로우 기반 시스템으로, 사용자의 쿼리를 기반으로 검색, 분석, 보고서 생성, 만화 아이디어 창출, 시나리오 생성, 이미지 생성까지의 전체 파이프라인을 자동화합니다.

## 주요 기능

### 🎯 핵심 워크플로우
1. **쿼리 분석 (N02)**: 사용자 입력 쿼리 분석 및 맥락 파악
2. **검색 전략 수립 (N03)**: 최적화된 검색 전략 계획
3. **검색 실행 (N04)**: 외부 데이터 소스에서 관련 정보 수집
4. **보고서 생성 (N05)**: 수집된 정보를 기반으로 종합 보고서 작성
5. **Human-in-the-Loop (HITL)**: 사용자 피드백 수집 및 반영
6. **보고서 저장 (N06)**: 로컬 및 클라우드 저장
7. **맥락 기반 요약 (N06A)**: 정제된 요약본 생성
8. **만화 아이디어 생성 (N07)**: 창의적 만화 컨셉 개발
9. **시나리오 생성 (N08)**: 상세 스토리보드 작성
10. **이미지 생성 (N09)**: AI 기반 만화 이미지 생성
11. **최종화 및 알림 (N10)**: 결과물 업로드 및 사용자 알림

### 🔄 API 버전
- **V1 API**: 비동기 만화 생성 요청 처리
  - `POST /api/v1/comics`: 만화 생성 작업 요청
  - `GET /api/v1/comics/status/{comic_id}`: 작업 상태 조회
  - LLM 큐 관리 엔드포인트 포함

- **V2 API**: 대화형 워크플로우 제어
  - `POST /api/v2/chat/workflow`: 단계별 워크플로우 실행

## 🏗️ 시스템 아키텍처

### 디렉토리 구조
```
ai/
├── app/
│   ├── api/                    # API 엔드포인트
│   │   ├── v1/                # V1 API (비동기 처리)
│   │   └── v2/                # V2 API (대화형)
│   ├── config/                # 설정 관리
│   ├── nodes/                 # 워크플로우 노드들 (N01-N10)
│   ├── nodes_v2/             # V2 노드들
│   ├── services/             # 핵심 서비스들
│   ├── tools/                # 검색, 스크래핑, 소셜 도구들
│   ├── utils/                # 유틸리티 함수들
│   ├── workers/              # 백그라운드 워커들
│   └── workflows/            # 워크플로우 관리
├── scripts/                  # 서비스 관리 스크립트
├── tests/                    # 테스트 코드
├── logs/                     # 로그 파일들
├── results/                  # 생성 결과물들
└── main.py                   # FastAPI 애플리케이션 진입점
```

### 핵심 컴포넌트

#### 1. 워크플로우 엔진
- **LangGraph 기반**: 복잡한 AI 워크플로우 오케스트레이션
- **상태 관리**: Pydantic 모델을 통한 타입 안전 상태 관리
- **노드 기반 처리**: 각 단계를 독립적인 노드로 분리

#### 2. 서비스 레이어
- **LLM Service**: 언어 모델 통신 및 관리
- **Database Client**: Redis/PostgreSQL 연결 관리
- **Storage Service**: S3/로컬 파일 저장 관리
- **Translation Service**: 다국어 번역 지원
- **Image Service**: AI 이미지 생성 서비스
- **Backend Client**: 외부 백엔드 시스템 연동

#### 3. 도구 생태계
- **검색 도구**: Google 검색, 커뮤니티 검색
- **스크래핑 도구**: 웹 콘텐츠 수집
- **소셜 도구**: 소셜 미디어 데이터 수집
- **트렌드 도구**: 실시간 트렌드 분석

## 🚀 설치 및 실행

### 1. 환경 설정
```bash
# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.sample .env
# .env 파일을 편집하여 필요한 설정값 입력
```

### 2. 데이터베이스 설정
```bash
# PostgreSQL 스키마 생성
psql -h [HOST] -U [USER] -d [DATABASE] -f scripts/chat_schema.sql
```

### 3. 서비스 실행
```bash
# 개발 모드 (로컬 실행)
python main.py

# 또는 uvicorn 직접 실행
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 워커 실행 (별도 터미널)
python run_unified_llm_worker.py
```

### 4. 서비스 관리 스크립트
```bash
# 모든 서비스 시작
bash scripts/start_services.sh

# 서비스 상태 확인
bash scripts/status_services.sh

# 서비스 중지
bash scripts/stop_services.sh

# 서비스 재시작
bash scripts/restart_services.sh
```

## 📝 주요 환경 변수

### 애플리케이션 설정
- `APP_NAME`: 애플리케이션 이름
- `APP_VERSION`: 버전 정보
- `APP_HOST`, `APP_PORT`: 서버 바인딩 정보
- `LOG_LEVEL`: 로깅 레벨

### 데이터베이스 설정
- `POSTGRES_*`: PostgreSQL 연결 정보
- `REDIS_*`: Redis 연결 정보

### AI 서비스 설정
- `LLM_API_ENDPOINT`: LLM 서비스 엔드포인트
- `DEFAULT_LLM_MODEL`: 기본 언어 모델
- `IMAGE_SERVER_URL`: 이미지 생성 서버 URL

### 외부 API 설정
- `GOOGLE_API_KEY`, `GOOGLE_CSE_ID`: Google 검색 API
- `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET`: 네이버 번역 API
- `S3_BUCKET_NAME`, `AWS_REGION`: AWS S3 설정

### LangSmith 추적 (선택사항)
- `LANGCHAIN_TRACING_V2`: LangSmith 추적 활성화
- `LANGCHAIN_API_KEY`: LangSmith API 키
- `LANGCHAIN_PROJECT`: 프로젝트 이름

## 📚 API 사용법

### V1 API - 비동기 만화 생성

#### 만화 생성 요청
```bash
curl -X POST "http://localhost:8000/api/v1/comics" \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "query": "최근 AI 기술 동향에 대한 만화를 만들어주세요",
      "site": {
        "include_dcinside": true,
        "include_fmkorea": true
      }
    },
    "writer_id": "user123"
  }'
```

#### 작업 상태 조회
```bash
curl -X GET "http://localhost:8000/api/v1/comics/status/{comic_id}"
```

### V2 API - 대화형 워크플로우

```bash
curl -X POST "http://localhost:8000/api/v2/chat/workflow" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "req123",
    "room_id": "room456",
    "user_id": "user789",
    "message": "AI 뉴스로 만화 만들기",
    "target_status": "INIT_01_02",
    "comic_id": null
  }'
```

## 🔧 개발 가이드

### 새로운 노드 추가
1. `app/nodes/` 디렉토리에 새 노드 파일 생성
2. `WorkflowState` 모델에 필요한 상태 필드 추가
3. 메인 워크플로우에 노드 연결
4. 테스트 코드 작성

### 새로운 서비스 추가
1. `app/services/` 디렉토리에 서비스 클래스 생성
2. 의존성 주입을 위해 `app/dependencies.py` 업데이트
3. 환경 변수가 필요한 경우 `.env.sample` 업데이트

### 새로운 도구 추가
1. `app/tools/` 아래 적절한 카테고리에 도구 클래스 생성
2. 필요시 새 카테고리 디렉토리 생성
3. 도구를 사용하는 노드에서 임포트하여 활용

## 🧪 테스트

```bash
# 모든 테스트 실행
pytest

# 특정 테스트 파일 실행
pytest tests/test_workflow.py

# 커버리지 포함 테스트
pytest --cov=app tests/
```

## 📊 모니터링 및 로깅

### 로깅 시스템
- 구조화된 로깅: JSON 형태의 로그 출력
- 로그 레벨별 분리: DEBUG, INFO, WARNING, ERROR
- 로그 파일 로테이션: 크기 및 시간 기반 로테이션

### 메트릭 수집
- 워크플로우 실행 시간 추적
- 각 노드별 성능 메트릭
- 오류율 및 재시도 통계

### LangSmith 통합
- 워크플로우 실행 추적
- LLM 호출 모니터링
- 성능 분석 및 디버깅

## 🔒 보안 고려사항

### API 보안
- 입력 데이터 유효성 검사 (Pydantic 스키마)
- 스팸 필터링 시스템
- 요청 제한 및 타임아웃 설정

### 데이터 보안
- 민감 정보 환경 변수 관리
- 암호화된 데이터베이스 연결
- S3 버킷 권한 관리

## 🚨 문제 해결

### 일반적인 문제

#### 1. 워크플로우 실행 실패
- 로그 파일 확인: `logs/` 디렉토리
- 환경 변수 설정 확인
- 외부 서비스 연결 상태 확인

#### 2. 이미지 생성 실패
- `IMAGE_SERVER_URL` 설정 확인
- Stable Diffusion WebUI 서비스 상태 확인
- 디스크 공간 확인

#### 3. 데이터베이스 연결 오류
- PostgreSQL/Redis 서비스 상태 확인
- 네트워크 연결 및 방화벽 설정 확인
- 연결 풀 설정 조정

### 성능 최적화

#### 1. 워크플로우 성능
- 병렬 처리 가능한 노드 식별
- 캐싱 전략 구현
- 불필요한 API 호출 최소화

#### 2. 메모리 관리
- 대용량 이미지 처리 시 스트리밍 사용
- 불필요한 객체 가비지 컬렉션
- 메모리 사용량 모니터링

## 📈 확장 계획

### 단기 계획
- 추가 이미지 생성 모델 지원
- 더 많은 검색 소스 연동
- 실시간 협업 기능 추가

### 장기 계획
- 멀티모달 콘텐츠 생성
- 사용자 맞춤형 스타일 학습
- 대규모 분산 처리 지원

## 🤝 기여 가이드

1. Fork 및 브랜치 생성
2. 기능 개발 및 테스트
3. 코드 스타일 가이드 준수
4. Pull Request 생성
5. 코드 리뷰 진행

## 📄 라이선스

이 프로젝트는 내부 사용을 위한 프로젝트입니다. 외부 배포 시 라이선스 정책을 별도 수립해야 합니다.

---

## 📞 지원 및 문의

개발팀 연락처: [개발팀 이메일]
문서 업데이트: [문서 관리자]
버그 리포트: [이슈 트래커 URL]

---

*이 README는 프로젝트 구조 분석을 통해 자동 생성되었습니다. 최신 정보는 개발팀에 문의하시기 바랍니다.*