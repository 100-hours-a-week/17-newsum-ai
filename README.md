# Newsom AI 서비스

뉴스 기사를 기반으로 4컷 만화를 생성하는 다단계 AI 워크플로우
(LangGraph + FastAPI 기반)

## 폴더 구조

```
project_root/
│
├── app/                          # 메인 애플리케이션 코드
│   ├── agents/                   # 각 작업을 수행하는 LangGraph 에이전트
│   ├── tools/                    # 에이전트가 사용하는 기능성 도구 모듈
│   ├── workflows/                # LangGraph 그래프, 상태, 전이 정의
│   ├── services/                 # 외부 API (LLM, 이미지 생성, 저장소, LangSmith) 연동
│   ├── api/                      # FastAPI HTTP 엔드포인트 및 스키마 정의
│   ├── config/                   # 환경 설정 및 상수 관리
│   └── utils/                    # 로거, 재시도, 타이머 등 공용 유틸리티
│
├── tests/                         # 단위 테스트 및 통합 테스트
│
├── main.py                        # FastAPI 서버 실행 진입점
├── requirements.txt               # 필요한 Python 패키지 목록
├── .env                            # 비공개 환경 설정 파일
├── .env.sample                     # 환경 설정 예시 파일
├── README.md                       # 프로젝트 설명 및 아키텍처 문서
└── .gitignore                      # Git에서 제외할 파일 설정
```

## Code Components Overview

- **main.py**: FastAPI 서버를 실행하는 메인 진입점
- **app/agents/**: 뉴스 수집, 요약, 유머 삽입 등 만화 생성의 각 단계를 담당하는 에이전트 모음
- **app/tools/**: 요약, 번역, 후처리 등의 독립 기능 모듈
- **app/workflows/**: StateGraph를 기반으로 LangGraph 워크플로우를 구성하고 상태 및 전이를 관리
- **app/services/**: 외부 서비스(Colab 서버, S3 스토리지, LangSmith 모니터링 등)와 통신하는 클라이언트 모듈
- **app/api/**: BE(백엔드)와의 외부 통신을 위한 RESTful API 엔드포인트 및 스트리밍 지원을 제공
- **app/config/**: API 키, 환경 변수 등 설정 정보를 관리
- **app/utils/**: 로깅, 재시도 데코레이터, 성능 측정 타이머 등 공통 유틸리티를 제공
- **tests/**: 기능별 단위 테스트 및 전체 통합 테스트를 구성
