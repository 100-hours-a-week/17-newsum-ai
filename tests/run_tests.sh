#!/bin/bash
# tests/run_tests.sh
# 테스트 실행 스크립트

set -e  # 오류 발생 시 스크립트 중단

echo "🧪 AI 프로젝트 테스트 실행 스크립트"
echo "=================================="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 함수 정의
print_step() {
    echo -e "${BLUE}📋 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 프로젝트 루트 디렉토리로 이동
cd "$(dirname "$0")/.."

# 가상환경 확인
if [[ -z "$VIRTUAL_ENV" ]]; then
    print_warning "가상환경이 활성화되지 않았습니다. venv를 활성화하는 것을 권장합니다."
    if [[ -d "venv" ]]; then
        print_step "venv 활성화 중..."
        source venv/bin/activate || source venv/Scripts/activate
        print_success "가상환경 활성화됨"
    fi
fi

# 테스트 의존성 설치 확인
print_step "테스트 의존성 확인 중..."
if [[ -f "tests/requirements-test.txt" ]]; then
    pip install -r tests/requirements-test.txt --quiet
    print_success "테스트 의존성 설치 완료"
else
    print_warning "tests/requirements-test.txt 파일이 없습니다."
fi

# 명령줄 인자 파싱
TEST_TYPE=${1:-"all"}
COVERAGE=${2:-"yes"}

print_step "테스트 유형: $TEST_TYPE"

case $TEST_TYPE in
    "unit")
        print_step "단위 테스트 실행 중..."
        pytest tests/ -m "unit" -v
        ;;
    "integration")
        print_step "통합 테스트 실행 중..."
        pytest tests/ -m "integration" -v
        ;;
    "api")
        print_step "API 테스트 실행 중..."
        pytest tests/test_api_endpoints.py tests/test_integration.py::TestAPIIntegration -v
        ;;
    "services")
        print_step "서비스 테스트 실행 중..."
        pytest tests/test_services.py -v
        ;;
    "schemas")
        print_step "스키마 테스트 실행 중..."
        pytest tests/test_schemas.py -v
        ;;
    "fast")
        print_step "빠른 테스트 실행 중 (slow 테스트 제외)..."
        pytest tests/ -v -m "not slow"
        ;;
    "slow")
        print_step "느린 테스트 실행 중..."
        pytest tests/ -v -m "slow"
        ;;
    "all")
        print_step "모든 테스트 실행 중..."
        if [[ $COVERAGE == "yes" ]]; then
            pytest tests/ -v --cov=app --cov-report=html --cov-report=term-missing
        else
            pytest tests/ -v
        fi
        ;;
    *)
        print_error "알 수 없는 테스트 유형: $TEST_TYPE"
        echo "사용법: $0 [unit|integration|api|services|schemas|fast|slow|all] [yes|no]"
        echo "예시:"
        echo "  $0 unit          # 단위 테스트만 실행"
        echo "  $0 all yes       # 모든 테스트 + 커버리지"
        echo "  $0 fast no       # 빠른 테스트만, 커버리지 없이"
        exit 1
        ;;
esac

# 테스트 결과 확인
if [[ $? -eq 0 ]]; then
    print_success "모든 테스트가 성공적으로 완료되었습니다! 🎉"
    
    if [[ $TEST_TYPE == "all" && $COVERAGE == "yes" ]]; then
        echo ""
        print_step "커버리지 리포트가 생성되었습니다:"
        echo "  - HTML 리포트: htmlcov/index.html"
        echo "  - 터미널 리포트: 위에 표시됨"
    fi
else
    print_error "일부 테스트가 실패했습니다."
    exit 1
fi

echo ""
print_step "테스트 실행 완료"
echo "=================================="
