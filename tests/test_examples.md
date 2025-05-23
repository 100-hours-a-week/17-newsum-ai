# 테스트 실행 예시 및 결과 확인
# tests/test_examples.md

## 테스트 실행 예시

### 1. 기본 테스트 실행

```bash
# 프로젝트 루트 디렉토리에서 실행
cd C:\Users\xodnr\Downloads\dev\project\17-team-4cut\ai

# 모든 테스트 실행
pytest tests/ -v

# 결과 예시:
# ========================= test session starts =========================
# tests/test_api_endpoints.py::TestRequestComicGeneration::test_request_comic_generation_success PASSED
# tests/test_api_endpoints.py::TestRequestComicGeneration::test_request_comic_generation_no_site_preferences PASSED
# tests/test_schemas.py::TestAsyncComicRequest::test_async_comic_request_full PASSED
# ...
# ========================= 45 passed in 2.34s =========================
```

### 2. 특정 테스트 파일 실행

```bash
# API 엔드포인트 테스트만 실행
pytest tests/test_api_endpoints.py -v

# 스키마 테스트만 실행
pytest tests/test_schemas.py -v

# 결과 예시:
# tests/test_schemas.py::TestSitePreferencesPayload::test_site_preferences_valid_all_fields PASSED
# tests/test_schemas.py::TestSitePreferencesPayload::test_site_preferences_optional_fields PASSED
# tests/test_schemas.py::TestRequestDataPayload::test_request_data_valid_with_site PASSED
```

### 3. 마커를 이용한 테스트 실행

```bash
# 단위 테스트만 실행
pytest tests/ -m unit -v

# 통합 테스트만 실행
pytest tests/ -m integration -v

# 느린 테스트 제외하고 실행
pytest tests/ -m "not slow" -v
```

### 4. 커버리지와 함께 실행

```bash
# HTML 커버리지 리포트 생성
pytest tests/ --cov=app --cov-report=html --cov-report=term-missing

# 결과 예시:
# Name                                    Stmts   Miss  Cover   Missing
# ---------------------------------------------------------------------
# app/__init__.py                             0      0   100%
# app/api/__init__.py                         0      0   100%
# app/api/v1/__init__.py                      0      0   100%
# app/api/v1/endpoints.py                    75      8    89%   45-52, 89
# app/api/v1/schemas.py                      25      0   100%
# app/api/v1/background_tasks.py             95     12    87%   67-73, 156-162
# app/services/database_client.py            85     15    82%   134-145, 189-195
# app/utils/logger.py                        45      5    89%   78-82
# ---------------------------------------------------------------------
# TOTAL                                     325     40    88%
```

### 5. 스크립트를 이용한 실행

```bash
# Linux/Mac
./tests/run_tests.sh all yes

# Windows
tests\run_tests.bat all yes

# 결과 예시:
# 🧪 AI 프로젝트 테스트 실행 스크립트
# ==================================
# 📋 테스트 의존성 확인 중...
# ✅ 테스트 의존성 설치 완료
# 📋 테스트 유형: all
# 📋 모든 테스트 실행 중...
# ✅ 모든 테스트가 성공적으로 완료되었습니다! 🎉
```

### 6. Makefile을 이용한 실행

```bash
# Makefile이 있는 tests 디렉토리에서
cd tests/

# 도움말 확인
make help

# 모든 테스트 실행
make test

# 커버리지와 함께 실행
make test-coverage

# 빠른 테스트만 실행
make test-fast

# 결과 예시:
# 🧪 모든 테스트 실행 중...
# ========================= test session starts =========================
# collecting ... collected 45 items
# 
# tests/test_api_endpoints.py::TestRequestComicGeneration::test_request_comic_generation_success PASSED [  2%]
# tests/test_api_endpoints.py::TestRequestComicGeneration::test_request_comic_generation_no_site_preferences PASSED [  4%]
# ...
# ========================= 45 passed in 3.21s =========================
```

### 7. 성능 테스트 실행

```bash
# 벤치마크 테스트만 실행
pytest tests/test_performance.py --benchmark-only -v

# 결과 예시:
# ----------------------------- benchmark: 5 tests -----------------------------
# Name (time in us)                              Min      Max     Mean  StdDev
# ----------------------------------------------------------------------------
# test_schema_validation_performance          125.30   345.67   156.23   23.45
# test_schema_serialization_performance        89.12   234.56   112.34   18.76
# test_uuid_generation_performance             12.45    45.67    18.23    4.56
# test_helper_function_performance               2.34     8.90     3.45    0.89
# ----------------------------------------------------------------------------
```

## 예상 테스트 결과

### ✅ 성공 케이스

1. **API 엔드포인트 테스트**: 18개 테스트 모두 통과
   - POST /comics 성공/실패 시나리오
   - GET /comics/status 성공/실패 시나리오
   - 유효성 검사 오류 처리

2. **스키마 테스트**: 15개 테스트 모두 통과
   - Pydantic 모델 유효성 검사
   - 직렬화/역직렬화
   - 선택적 필드 처리

3. **백그라운드 태스크 테스트**: 8개 테스트 모두 통과
   - 헬퍼 함수 테스트
   - 워크플로우 트리거 테스트
   - 성공/실패 시나리오

4. **서비스 테스트**: 12개 테스트 모두 통과
   - DatabaseClient 초기화
   - Redis 연결 모킹
   - 오류 처리

5. **유틸리티 테스트**: 8개 테스트 모두 통과
   - ContextFilter 동작
   - 로거 설정
   - 모듈 임포트

### ⚠️ 주의사항

1. **의존성 누락 시**:
   ```
   ImportError: No module named 'app.config.settings'
   ```
   → 실제 settings 모듈이 없을 수 있음

2. **Redis 연결 오류**:
   ```
   redis.exceptions.ConnectionError
   ```
   → Mock이 제대로 적용되지 않은 경우

3. **비동기 테스트 오류**:
   ```
   RuntimeError: cannot be called from a running event loop
   ```
   → pytest-asyncio 설정 확인 필요

### 🔧 트러블슈팅

1. **PYTHONPATH 설정**:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **가상환경 활성화**:
   ```bash
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **테스트 의존성 설치**:
   ```bash
   pip install -r tests/requirements-test.txt
   ```

## 커버리지 목표

- **전체 커버리지**: 85% 이상
- **API 계층**: 90% 이상
- **서비스 계층**: 80% 이상
- **유틸리티**: 95% 이상

## CI/CD 통합 예시

```yaml
# .github/workflows/tests.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r tests/requirements-test.txt
    
    - name: Run tests with coverage
      run: |
        pytest tests/ --cov=app --cov-report=xml --cov-report=term-missing
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## 테스트 성공 확인 방법

1. **모든 테스트 통과**: `45 passed` 메시지 확인
2. **커버리지 85% 이상**: HTML 리포트에서 확인
3. **성능 기준 만족**: 벤치마크 결과가 예상 범위 내
4. **메모리 누수 없음**: 스트레스 테스트 통과
