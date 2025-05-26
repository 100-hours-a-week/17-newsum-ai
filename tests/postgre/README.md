# PostgreSQL 테스트

## 🧪 테스트 내용
- CREATE TABLE (테이블 생성)
- INSERT & SELECT (데이터 삽입 및 조회)  
- BULK INSERT (배치 삽입)
- DROP TABLE (테이블 삭제)

## 🚀 실행 방법

### 환경 설정
```bash
# .env 파일에 PostgreSQL 연결 정보 설정 필요
POSTGRES_HOST=dev.new-sum.com
POSTGRES_USER=newsum_user
POSTGRES_PASSWORD=zkxpqn17
POSTGRES_DB=newsum
```

### 테스트 실행
```bash
# 방법 1
pytest tests/test_postgresql_service.py -v

# 방법 2  
python tests/run_tests.py

# 특정 테스트만
pytest tests/test_postgresql_service.py::TestPostgreSQLService::test_create_table -v
```

## 📋 테스트 케이스
1. `test_create_table`: 테이블 생성
2. `test_insert_and_select`: 데이터 삽입/조회
3. `test_bulk_insert`: 배치 데이터 삽입
4. `test_drop_table`: 테이블 삭제

## ⚠️ 주의사항
- 실제 PostgreSQL 연결 필요
- SSH 키 파일 위치: `app/services/newsum-dev-key`
- 테이블 생성/삭제 권한 필요
