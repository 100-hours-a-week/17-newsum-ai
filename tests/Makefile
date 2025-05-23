# Makefile for test execution
# tests/Makefile

.PHONY: help test test-unit test-integration test-api test-services test-schemas test-fast test-slow test-coverage clean install-test-deps

# 기본 Python 명령어
PYTHON := python
PIP := pip
PYTEST := pytest

# 프로젝트 디렉토리
PROJECT_DIR := ..
TEST_DIR := .

# 기본 타겟
help: ## 도움말 표시
	@echo "사용 가능한 명령어:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "예시:"
	@echo "  make test              # 모든 테스트 실행"
	@echo "  make test-coverage     # 커버리지와 함께 테스트 실행"
	@echo "  make test-fast         # 빠른 테스트만 실행"

install-test-deps: ## 테스트 의존성 설치
	@echo "📦 테스트 의존성 설치 중..."
	$(PIP) install -r requirements-test.txt
	@echo "✅ 테스트 의존성 설치 완료"

test: ## 모든 테스트 실행
	@echo "🧪 모든 테스트 실행 중..."
	cd $(PROJECT_DIR) && $(PYTEST) tests/ -v

test-unit: ## 단위 테스트만 실행
	@echo "🔬 단위 테스트 실행 중..."
	cd $(PROJECT_DIR) && $(PYTEST) tests/ -m "unit" -v

test-integration: ## 통합 테스트만 실행
	@echo "🔗 통합 테스트 실행 중..."
	cd $(PROJECT_DIR) && $(PYTEST) tests/ -m "integration" -v

test-api: ## API 테스트만 실행
	@echo "🌐 API 테스트 실행 중..."
	cd $(PROJECT_DIR) && $(PYTEST) tests/test_api_endpoints.py tests/test_integration.py::TestAPIIntegration -v

test-services: ## 서비스 테스트만 실행
	@echo "⚙️  서비스 테스트 실행 중..."
	cd $(PROJECT_DIR) && $(PYTEST) tests/test_services.py -v

test-schemas: ## 스키마 테스트만 실행
	@echo "📋 스키마 테스트 실행 중..."
	cd $(PROJECT_DIR) && $(PYTEST) tests/test_schemas.py -v

test-utils: ## 유틸리티 테스트만 실행
	@echo "🛠️  유틸리티 테스트 실행 중..."
	cd $(PROJECT_DIR) && $(PYTEST) tests/test_utils.py -v

test-background: ## 백그라운드 태스크 테스트만 실행
	@echo "⏰ 백그라운드 태스크 테스트 실행 중..."
	cd $(PROJECT_DIR) && $(PYTEST) tests/test_background_tasks.py -v

test-fast: ## 빠른 테스트만 실행 (slow 마커 제외)
	@echo "⚡ 빠른 테스트 실행 중..."
	cd $(PROJECT_DIR) && $(PYTEST) tests/ -v -m "not slow"

test-slow: ## 느린 테스트만 실행
	@echo "🐌 느린 테스트 실행 중..."
	cd $(PROJECT_DIR) && $(PYTEST) tests/ -v -m "slow"

test-coverage: ## 커버리지와 함께 테스트 실행
	@echo "📊 커버리지 테스트 실행 중..."
	cd $(PROJECT_DIR) && $(PYTEST) tests/ -v --cov=app --cov-report=html --cov-report=term-missing
	@echo "📈 커버리지 리포트: htmlcov/index.html"

test-coverage-xml: ## XML 형태의 커버리지 리포트 생성 (CI용)
	@echo "📊 XML 커버리지 리포트 생성 중..."
	cd $(PROJECT_DIR) && $(PYTEST) tests/ --cov=app --cov-report=xml --cov-report=term

test-benchmark: ## 성능 벤치마크 테스트 실행
	@echo "🏃 벤치마크 테스트 실행 중..."
	cd $(PROJECT_DIR) && $(PYTEST) tests/ --benchmark-only -v

test-parallel: ## 병렬로 테스트 실행
	@echo "🚀 병렬 테스트 실행 중..."
	cd $(PROJECT_DIR) && $(PYTEST) tests/ -v -n auto

test-verbose: ## 매우 상세한 테스트 실행
	@echo "🔍 상세 테스트 실행 중..."
	cd $(PROJECT_DIR) && $(PYTEST) tests/ -vv -s

test-failed: ## 마지막에 실패한 테스트만 재실행
	@echo "❌ 실패한 테스트 재실행 중..."
	cd $(PROJECT_DIR) && $(PYTEST) tests/ --lf -v

test-watch: ## 파일 변경 시 자동으로 테스트 실행 (pytest-watch 필요)
	@echo "👀 파일 감시 모드로 테스트 실행..."
	cd $(PROJECT_DIR) && ptw tests/ app/

clean: ## 테스트 관련 캐시 및 임시 파일 정리
	@echo "🧹 테스트 캐시 정리 중..."
	find $(PROJECT_DIR) -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find $(PROJECT_DIR) -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find $(PROJECT_DIR) -name "*.pyc" -delete 2>/dev/null || true
	rm -rf $(PROJECT_DIR)/htmlcov/ 2>/dev/null || true
	rm -f $(PROJECT_DIR)/.coverage 2>/dev/null || true
	rm -f $(PROJECT_DIR)/coverage.xml 2>/dev/null || true
	@echo "✅ 정리 완료"

lint: ## 코드 스타일 검사
	@echo "🔍 코드 스타일 검사 중..."
	cd $(PROJECT_DIR) && flake8 app/ tests/
	cd $(PROJECT_DIR) && black --check app/ tests/
	cd $(PROJECT_DIR) && isort --check-only app/ tests/

format: ## 코드 포맷팅 적용
	@echo "✨ 코드 포맷팅 적용 중..."
	cd $(PROJECT_DIR) && black app/ tests/
	cd $(PROJECT_DIR) && isort app/ tests/

type-check: ## 타입 검사
	@echo "🔍 타입 검사 중..."
	cd $(PROJECT_DIR) && mypy app/

quality: lint type-check ## 코드 품질 검사 (린트 + 타입 검사)

ci: install-test-deps test-coverage-xml lint type-check ## CI 환경에서 실행할 모든 검사

# 테스트 환경 설정
setup-test-env: ## 테스트 환경 설정
	@echo "🛠️  테스트 환경 설정 중..."
	$(PIP) install -e $(PROJECT_DIR)
	make install-test-deps
	@echo "✅ 테스트 환경 설정 완료"

# 개발 환경에서 유용한 조합 명령어
dev-test: test-fast lint ## 개발 중 빠른 검증 (빠른 테스트 + 린트)
	@echo "🎯 개발 테스트 완료"

full-check: test-coverage lint type-check ## 전체 검사 (커버리지 + 린트 + 타입)
	@echo "✅ 전체 검사 완료"
