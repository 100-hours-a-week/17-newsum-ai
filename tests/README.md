# 테스트 가이드

## 개요

이 프로젝트의 테스트 코드는 FastAPI 기반 AI 만화 생성 서비스의 주요 기능들을 검증합니다.

## 테스트 구조

```
tests/
├── conftest.py              # pytest 설정 및 공통 픽스처
├── test_api_endpoints.py    # API 엔드포인트 테스트
├── test_schemas.py          # Pydantic 스키마 테스트
├── test_background_tasks.py # 백그라운드 태스크 테스트
├── test_services.py         # 서비스 클래스 테스트
├── test_utils.py            # 유틸리티 함수 테스트
├── test_integration.py      # 통합 테스트
├── requirements-test.txt    # 테스트 의존성
├── run_tests.sh            # Linux/Mac 테스트 실행 스크립트
├── run_tests.bat           # Windows 테스트 실행 스크립트
└── README.md               # 이 파일
```

## 테스트 분류

### 단위 테스트 (Unit Tests)
- **대상**: 개별 함수, 클래스, 메서드
- **표시**: `@pytest.mark.unit`
- **특징**: 외부 의존성 최소화, 빠른 실행

### 통합 테스트 (Integration Tests)
- **대상**: 여러 컴포넌트 간 상호작용
- **표시**: `@pytest.mark.integration`
- **특징**: 실제 데이터 플로우 검증

### 느린 테스트 (Slow Tests)
- **대상**: 시간이 오래 걸리는 테스트
- **표시**: `@pytest.mark.slow`
- **특징**: CI/CD에서 별도 실행 고려

## 테스트 실행 방법

### 1. 기본 실행

```bash
# 모든 테스트 실행
pytest tests/

# 특정 파일 테스트
pytest tests/test_api_endpoints.py

# 특정 테스트 함수
pytest tests/test_api_endpoints.py::TestRequestComicGeneration::test_request_comic_generation_success
```

### 2. 스크립트를 이용한 실행

**Linux/Mac:**
```bash
# 모든 테스트 + 커버리지
./tests/run_tests.sh all yes

# 단위 테스트만
./tests/run_tests.sh unit

# 빠른 테스트만 (slow 제외)
./tests/run_tests.sh fast
```

**Windows:**
```cmd
REM 모든 테스트 + 커버리지
tests\run_tests.bat all yes

REM API 테스트만
tests\run_tests.bat api

REM 서비스 테스트만
tests\run_tests.bat services
```

### 3. 마커를 이용한 실행

```bash
# 단위 테스트만
pytest -m unit

# 통합 테스트만
pytest -m integration

# 느린 테스트 제외
pytest -m "not slow"

# API 관련 테스트만
pytest -m api
```

## 주요 테스트 케이스

### API 엔드포인트 테스트
- **POST /api/v1/comics**: 만화 생성 요청
- **GET /api/v1/comics/status/{comic_id}**: 상태 조회
- 성공/실패 시나리오, 유효성 검사, 오류 처리

### 스키마 테스트
- Pydantic 모델 유효성 검사
- 직렬화/역직렬화 검증
- 선택적 필드 처리

### 백그라운드 태스크 테스트
- 워크플로우 실행 트리거
- 상태 업데이트 로직
- 오류 처리 및 복구

### 서비스 테스트
- DatabaseClient 연결 및 작업
- Redis 연동 테스트
- 설정 관리

## 테스트 작성 가이드

### 1. 테스트 함수 명명 규칙
```python
def test_[기능]_[시나리오]_[예상결과]():
    # 예시
    def test_request_comic_generation_success():
    def test_get_comic_status_not_found():
    def test_database_client_connection_error():
```

### 2. AAA 패턴 사용
```python
def test_example():
    # Arrange - 준비
    client = DatabaseClient()
    test_data = {"key": "value"}
    
    # Act - 실행
    result = client.process(test_data)
    
    # Assert - 검증
    assert result.status == "success"
    assert result.data == test_data
```

### 3. Mock 사용 예시
```python
@pytest.mark.asyncio
async def test_with_mock(mock_db_client):
    # Given
    mock_db_client.get.return_value = {"status": "success"}
    
    # When
    result = await service.get_status("test-id")
    
    # Then
    assert result["status"] == "success"
    mock_db_client.get.assert_called_once_with("test-id")
```

## 커버리지 보고서

테스트 커버리지를 확인하려면:

```bash
# HTML 리포트 생성
pytest --cov=app --cov-report=html

# 터미널에서 확인
pytest --cov=app --cov-report=term-missing
```

생성된 HTML 리포트는 `htmlcov/index.html`에서 확인할 수 있습니다.

## CI/CD 통합

### GitHub Actions 예시
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r tests/requirements-test.txt
      - name: Run tests
        run: pytest tests/ --cov=app --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## 트러블슈팅

### 자주 발생하는 문제들

1. **ImportError: No module named 'app'**
   ```bash
   # 해결: PYTHONPATH 설정 또는 프로젝트 루트에서 실행
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   pytest tests/
   ```

2. **Redis 연결 오류**
   ```bash
   # Mock을 사용하여 실제 Redis 없이 테스트
   # conftest.py의 mock_settings 픽스처가 자동으로 처리
   ```

3. **비동기 테스트 오류**
   ```python
   # @pytest.mark.asyncio 데코레이터 필수
   @pytest.mark.asyncio
   async def test_async_function():
       result = await async_function()
       assert result is not None
   ```

## 성능 테스트

벤치마크 테스트 실행:
```bash
pytest tests/ --benchmark-only
```

## 참고 자료

- [pytest 공식 문서](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [FastAPI 테스트](https://fastapi.tiangolo.com/tutorial/testing/)
- [Pydantic 테스트](https://pydantic-docs.helpmanual.io/usage/pytest_plugin/)
