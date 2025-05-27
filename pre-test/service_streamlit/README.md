# SLM Chat Service

## 🎯 프로젝트 개요

SLM Chat Service는 Streamlit 기반의 AI 채팅 서비스로, 멀티스테이지 워크플로우를 지원하는 대화형 AI 플랫폼입니다. 이 서비스는 사용자가 AI와 상호작용하며 쿼리, 검색, 아이디어 생성, 시나리오 작성, 이미지 생성 등의 단계별 작업을 수행할 수 있도록 설계되었습니다.

## 🏗️ 아키텍처

### 시스템 구성요소

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Streamlit UI   │───▶│   PostgreSQL    │    │   AI Server     │
│   (Frontend)    │    │   (Database)    │    │  (External)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                ┌─────────────────▼─────────────────┐
                │        FastAPI Server            │
                │      (Reception API)              │
                └───────────────────────────────────┘
```

### 핵심 컴포넌트

1. **Streamlit Frontend** (`app.py`, `app_improved.py`)
   - 사용자 인터페이스 제공
   - 채팅방 관리
   - 워크플로우 상태 시각화
   - 실시간 폴링 시스템

2. **PostgreSQL Database**
   - 사용자, 채팅방, 메시지 데이터 저장
   - 워크플로우 상태 관리
   - JSONB 타입을 활용한 유연한 데이터 구조

3. **FastAPI Reception Server** (`reception_api.py`)
   - AI 서버로부터 응답 수신
   - 비동기 메시지 처리

4. **State Management System** (`state_v2.py`, `state_view_utils.py`)
   - Pydantic 기반 워크플로우 상태 모델
   - 마크다운 형식 상태 시각화

## 📋 주요 기능

### 🔐 사용자 관리
- 닉네임 기반 간편 로그인
- 자동 사용자 생성
- 세션 상태 관리

### 💬 채팅방 관리
- 다중 채팅방 생성/삭제
- 채팅방별 독립적 상태 관리
- 실시간 메시지 동기화

### 🔄 워크플로우 단계
1. **Query (쿼리)** - 사용자 질문 분석
2. **Search (검색)** - 관련 정보 검색 및 리포트 생성
3. **Idea (아이디어)** - 창작 아이디어 제안
4. **Scenario (시나리오)** - 스토리 시나리오 작성
5. **Image (이미지)** - 이미지 생성 예약

### 📝 추가 기능
- **노트패드**: 채팅방별 메모 기능
- **상태 시각화**: 현재 워크플로우 진행 상황 표시
- **비동기 처리**: AI 응답의 비동기 처리
- **스마트 폴링**: 효율적인 실시간 업데이트

## 🛠️ 기술 스택

### Backend
- **Python 3.12+**
- **Streamlit**: 웹 애플리케이션 프레임워크
- **FastAPI**: API 서버
- **PostgreSQL**: 데이터베이스
- **psycopg2**: PostgreSQL 어댑터
- **Pydantic**: 데이터 검증 및 모델링

### Frontend
- **Streamlit**: 사용자 인터페이스
- **Markdown**: 상태 시각화

### 통신
- **aiohttp**: 비동기 HTTP 클라이언트
- **uvicorn**: ASGI 서버

## 📁 프로젝트 구조

```
service_streamlit/
├── app.py                    # 메인 Streamlit 애플리케이션
├── app_improved.py           # 개선된 버전의 애플리케이션
├── db_utils.py              # 데이터베이스 유틸리티 함수
├── db_schema.sql            # PostgreSQL 스키마 정의
├── reception_api.py         # FastAPI 응답 수신 서버
├── state_v2.py             # 워크플로우 상태 모델 (Pydantic)
├── state_view_utils.py     # 상태 시각화 유틸리티
├── requirements.txt        # Python 의존성
├── .env                    # 환경 변수 설정
├── newsum-dev-key         # 개발용 키 파일
└── __pycache__/           # Python 컴파일된 파일
```

## 🚀 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```
- 자동 사용자 생성
- 세션 상태 관리

### 💬 채팅방 관리
- 다중 채팅방 생성/삭제
- 채팅방별 독립적 상태 관리
- 실시간 메시지 동기화

### 🔄 워크플로우 단계
1. **Query (쿼리)** - 사용자 질문 분석
2. **Search (검색)** - 관련 정보 검색 및 리포트 생성
3. **Idea (아이디어)** - 창작 아이디어 제안
4. **Scenario (시나리오)** - 스토리 시나리오 작성
5. **Image (이미지)** - 이미지 생성 예약

### 📝 추가 기능
- **노트패드**: 채팅방별 메모 기능
- **상태 시각화**: 현재 워크플로우 진행 상황 표시
- **비동기 처리**: AI 응답의 비동기 처리
- **스마트 폴링**: 효율적인 실시간 업데이트

## 🛠️ 기술 스택

### Backend
- **Python 3.12+**
- **Streamlit**: 웹 애플리케이션 프레임워크
- **FastAPI**: API 서버
- **PostgreSQL**: 데이터베이스
- **psycopg2**: PostgreSQL 어댑터
- **Pydantic**: 데이터 검증 및 모델링

### Frontend
- **Streamlit**: 사용자 인터페이스
- **Markdown**: 상태 시각화

### 통신
- **aiohttp**: 비동기 HTTP 클라이언트
- **uvicorn**: ASGI 서버

## 📁 프로젝트 구조

```
service_streamlit/
├── app.py                    # 메인 Streamlit 애플리케이션
├── app_improved.py           # 개선된 버전의 애플리케이션
├── db_utils.py              # 데이터베이스 유틸리티 함수
├── db_schema.sql            # PostgreSQL 스키마 정의
├── reception_api.py         # FastAPI 응답 수신 서버
├── state_v2.py             # 워크플로우 상태 모델 (Pydantic)
├── state_view_utils.py     # 상태 시각화 유틸리티
├── requirements.txt        # Python 의존성
├── .env                    # 환경 변수 설정
├── newsum-dev-key         # 개발용 키 파일
└── __pycache__/           # Python 컴파일된 파일
```
- 자동 사용자 생성
- 세션 상태 관리

### 💬 채팅방 관리
- 다중 채팅방 생성/삭제
- 채팅방별 독립적 상태 관리
- 실시간 메시지 동기화

### 🔄 워크플로우 단계
1. **Query (쿼리)** - 사용자 질문 분석
2. **Search (검색)** - 관련 정보 검색 및 리포트 생성
3. **Idea (아이디어)** - 창작 아이디어 제안
4. **Scenario (시나리오)** - 스토리 시나리오 작성
5. **Image (이미지)** - 이미지 생성 예약

### 📝 추가 기능
- **노트패드**: 채팅방별 메모 기능
- **상태 시각화**: 현재 워크플로우 진행 상황 표시
- **비동기 처리**: AI 응답의 비동기 처리
- **스마트 폴링**: 효율적인 실시간 업데이트

## 🛠️ 기술 스택

### Backend
- **Python 3.12+**
- **Streamlit**: 웹 애플리케이션 프레임워크
- **FastAPI**: API 서버
- **PostgreSQL**: 데이터베이스
- **psycopg2**: PostgreSQL 어댑터
- **Pydantic**: 데이터 검증 및 모델링

### Frontend
- **Streamlit**: 사용자 인터페이스
- **Markdown**: 상태 시각화

### 통신
- **aiohttp**: 비동기 HTTP 클라이언트
- **uvicorn**: ASGI 서버

## 📁 프로젝트 구조

```
service_streamlit/
├── app.py                    # 메인 Streamlit 애플리케이션
├── app_improved.py           # 개선된 버전의 애플리케이션
├── db_utils.py              # 데이터베이스 유틸리티 함수
├── db_schema.sql            # PostgreSQL 스키마 정의
├── reception_api.py         # FastAPI 응답 수신 서버
├── state_v2.py             # 워크플로우 상태 모델 (Pydantic)
├── state_view_utils.py     # 상태 시각화 유틸리티
├── requirements.txt        # Python 의존성
├── .env                    # 환경 변수 설정
├── newsum-dev-key         # 개발용 키 파일
└── __pycache__/           # Python 컴파일된 파일
```

## 🚀 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 설정

`.env` 파일을 설정하여 데이터베이스 및 API 정보를 입력:

```env
# Database configuration
DB_NAME=your_database_name
DB_USER=your_database_user  
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432

# API configuration
RECEPTION_API_PORT=9090
AI_SERVER_URL=http://localhost:8000/api/v2/chat
```

### 3. 데이터베이스 설정

PostgreSQL에서 `db_schema.sql`을 실행하여 테이블 생성:

```bash
psql -U your_user -d your_database -f db_schema.sql
```

### 4. 서비스 실행

#### Reception API 서버 시작
```bash
python reception_api.py
```

#### Streamlit 애플리케이션 시작
```bash
# 기본 버전
streamlit run app.py

# 또는 개선된 버전
streamlit run app_improved.py
```

## 📁 프로젝트 구조

```
service_streamlit/
├── app.py                    # 메인 Streamlit 애플리케이션
├── app_improved.py           # 개선된 버전의 애플리케이션
├── db_utils.py              # 데이터베이스 유틸리티 함수
├── db_schema.sql            # PostgreSQL 스키마 정의
├── reception_api.py         # FastAPI 응답 수신 서버
├── state_v2.py             # 워크플로우 상태 모델 (Pydantic)
├── state_view_utils.py     # 상태 시각화 유틸리티
├── requirements.txt        # Python 의존성
├── .env                    # 환경 변수 설정
├── newsum-dev-key         # 개발용 키 파일
└── __pycache__/           # Python 컴파일된 파일
```

## 🚀 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 설정

`.env` 파일을 설정하여 데이터베이스 및 API 정보를 입력:

```env
# Database configuration
DB_NAME=your_database_name
DB_USER=your_database_user  
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432

# API configuration
RECEPTION_API_PORT=9090
AI_SERVER_URL=http://localhost:8000/api/v2/chat
```

### 3. 데이터베이스 설정

PostgreSQL에서 `db_schema.sql`을 실행하여 테이블 생성:

```bash
psql -U your_user -d your_database -f db_schema.sql
```

## 📁 프로젝트 구조

```
service_streamlit/
├── app.py                    # 메인 Streamlit 애플리케이션
├── app_improved.py           # 개선된 버전의 애플리케이션
├── db_utils.py              # 데이터베이스 유틸리티 함수
├── db_schema.sql            # PostgreSQL 스키마 정의
├── reception_api.py         # FastAPI 응답 수신 서버
├── state_v2.py             # 워크플로우 상태 모델 (Pydantic)
├── state_view_utils.py     # 상태 시각화 유틸리티
├── requirements.txt        # Python 의존성
├── .env                    # 환경 변수 설정
├── newsum-dev-key         # 개발용 키 파일
└── __pycache__/           # Python 컴파일된 파일
```

## 🚀 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 설정

`.env` 파일을 설정하여 데이터베이스 및 API 정보를 입력:

```env
# Database configuration
DB_NAME=your_database_name
DB_USER=your_database_user  
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432

# API configuration
RECEPTION_API_PORT=9090
AI_SERVER_URL=http://localhost:8000/api/v2/chat
```

### 3. 데이터베이스 설정

PostgreSQL에서 `db_schema.sql`을 실행하여 테이블 생성:

```bash
psql -U your_user -d your_database -f db_schema.sql
```
