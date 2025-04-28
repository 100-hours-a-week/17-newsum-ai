
# Newsom AI 서비스

LangGraph와 FastAPI를 사용하여 뉴스 기사로부터 4컷 만화를 생성하는 멀티스텝 AI 워크플로우

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
├── pre-test/                     # 개발 초기 단계에서의 자료 정리 및 기술 검증용 코드 저장소
│
├── main.py                       # FastAPI 서버 엔트리포인트
├── requirements.txt              # Python 패키지 의존성 목록
├── .env.sample                   # 환경설정 샘플 템플릿
├── README.md                     # 프로젝트 설명 및 아키텍처 문서
└── .gitignore                    # Git 무시 파일 설정
```

## 코드 컴포넌트 개요

- **main.py**: FastAPI 서버를 실행하는 진입점
- **pre-test/**: 개발 초기 단계에서의 자료 정리 및 기술 검증용 코드 저장소
- **app/agents/**: 뉴스 수집, 요약, 유머 추가 등 만화 생성 워크플로우의 각 단계를 담당하는 에이전트
- **app/tools/**: 요약, 번역, 후처리 등 독립적으로 사용 가능한 기능 모듈
- **app/workflows/**: StateGraph 및 상태 전이를 기반으로 한 LangGraph 워크플로우 오케스트레이션
- **app/services/**: 외부 API(Colab 서버, S3 스토리지, LangSmith 모니터링)와 통신하는 서비스 계층
- **app/api/**: BE와 통신하는 RESTful HTTP 엔드포인트(스트리밍 지원 포함)
- **app/config/**: 환경설정, API 키, 상수 등을 관리하는 모듈
- **app/utils/**: 공통 유틸리티 (로깅, 재시도 데코레이터, 성능 타이머 등)
