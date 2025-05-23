@echo off
REM tests/run_tests.bat
REM Windows용 테스트 실행 스크립트

echo 🧪 AI 프로젝트 테스트 실행 스크립트 (Windows)
echo ==================================

REM 프로젝트 루트 디렉토리로 이동
cd /d "%~dp0\.."

REM 가상환경 확인 및 활성화
if not defined VIRTUAL_ENV (
    echo ⚠️  가상환경이 활성화되지 않았습니다.
    if exist "venv\Scripts\activate.bat" (
        echo 📋 venv 활성화 중...
        call venv\Scripts\activate.bat
        echo ✅ 가상환경 활성화됨
    ) else (
        echo ❌ venv 디렉토리를 찾을 수 없습니다.
    )
)

REM 테스트 의존성 설치
echo 📋 테스트 의존성 확인 중...
if exist "tests\requirements-test.txt" (
    pip install -r tests\requirements-test.txt --quiet
    echo ✅ 테스트 의존성 설치 완료
) else (
    echo ⚠️  tests\requirements-test.txt 파일이 없습니다.
)

REM 명령줄 인자 처리
set TEST_TYPE=%1
set COVERAGE=%2

if "%TEST_TYPE%"=="" set TEST_TYPE=all
if "%COVERAGE%"=="" set COVERAGE=yes

echo 📋 테스트 유형: %TEST_TYPE%

if "%TEST_TYPE%"=="unit" (
    echo 📋 단위 테스트 실행 중...
    pytest tests\ -m "unit" -v
) else if "%TEST_TYPE%"=="integration" (
    echo 📋 통합 테스트 실행 중...
    pytest tests\ -m "integration" -v
) else if "%TEST_TYPE%"=="api" (
    echo 📋 API 테스트 실행 중...
    pytest tests\test_api_endpoints.py tests\test_integration.py::TestAPIIntegration -v
) else if "%TEST_TYPE%"=="services" (
    echo 📋 서비스 테스트 실행 중...
    pytest tests\test_services.py -v
) else if "%TEST_TYPE%"=="schemas" (
    echo 📋 스키마 테스트 실행 중...
    pytest tests\test_schemas.py -v
) else if "%TEST_TYPE%"=="fast" (
    echo 📋 빠른 테스트 실행 중 (slow 테스트 제외)...
    pytest tests\ -v -m "not slow"
) else if "%TEST_TYPE%"=="slow" (
    echo 📋 느린 테스트 실행 중...
    pytest tests\ -v -m "slow"
) else if "%TEST_TYPE%"=="all" (
    echo 📋 모든 테스트 실행 중...
    if "%COVERAGE%"=="yes" (
        pytest tests\ -v --cov=app --cov-report=html --cov-report=term-missing
    ) else (
        pytest tests\ -v
    )
) else (
    echo ❌ 알 수 없는 테스트 유형: %TEST_TYPE%
    echo 사용법: %0 [unit^|integration^|api^|services^|schemas^|fast^|slow^|all] [yes^|no]
    echo 예시:
    echo   %0 unit          # 단위 테스트만 실행
    echo   %0 all yes       # 모든 테스트 + 커버리지
    echo   %0 fast no       # 빠른 테스트만, 커버리지 없이
    exit /b 1
)

if %ERRORLEVEL% equ 0 (
    echo ✅ 모든 테스트가 성공적으로 완료되었습니다! 🎉
    
    if "%TEST_TYPE%"=="all" if "%COVERAGE%"=="yes" (
        echo.
        echo 📋 커버리지 리포트가 생성되었습니다:
        echo   - HTML 리포트: htmlcov\index.html
        echo   - 터미널 리포트: 위에 표시됨
    )
) else (
    echo ❌ 일부 테스트가 실패했습니다.
    exit /b 1
)

echo.
echo 📋 테스트 실행 완료
echo ==================================
