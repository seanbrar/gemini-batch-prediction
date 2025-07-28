# Gemini Batch Processing Testing Makefile

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
PYTEST = pytest
PYTEST_ARGS = -v
COVERAGE_FAIL_UNDER ?= 40
COVERAGE_ARGS = --cov=gemini_batch --cov-report=term-missing --cov-report=html:coverage_html_report --cov-fail-under=$(COVERAGE_FAIL_UNDER)

# Default log level for pytest's console output. Can be overridden.
TEST_LOG_LEVEL ?= WARNING

# ------------------------------------------------------------------------------
# Main Commands
# ------------------------------------------------------------------------------
.PHONY: help test test-all test-coverage install-dev clean

help: ## ✨ Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install-dev: ## 📦 Install all development dependencies
	@echo "📦 Installing development dependencies..."
	pip install -e ".[dev]"
	@echo "✅ Development environment ready"

test: ## 🎯 Run the default test suite (unit + golden file regression) without coverage
	@echo "🎯 Running default test suite (unit + golden file regression)..."
	$(PYTEST) $(PYTEST_ARGS) --log-cli-level=$(TEST_LOG_LEVEL) -m "unit or golden_test"

test-coverage: ## 📊 Run all tests and generate a coverage report
	@echo "📊 Running all tests with coverage report..."
	$(PYTEST) $(PYTEST_ARGS) $(COVERAGE_ARGS) --log-cli-level=$(TEST_LOG_LEVEL) tests/
	@echo "✅ Coverage report generated in coverage_html_report/"

test-all: test test-integration test-workflows ## 🏁 Run all non-API tests
	@echo "✅ All non-API tests complete."

lint: ## ✒️ Check formatting and lint code
	@echo "✒️ Checking formatting and linting with ruff..."
	ruff format --check .
	ruff check .

clean: ## 🧹 Clean up all test and build artifacts
	@echo "🧹 Cleaning up..."
	rm -rf .pytest_cache/ coverage_html_report/ .coverage dist/ build/ *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "✅ Cleanup completed"

# ------------------------------------------------------------------------------
# Granular Test Targets
# ------------------------------------------------------------------------------
.PHONY: test-unit test-golden-files test-integration test-api test-workflows

test-unit: ## 🧪 Run all unit tests
	@echo "🧪 Running unit tests..."
	$(PYTEST) $(PYTEST_ARGS) --log-cli-level=$(TEST_LOG_LEVEL) -m "unit"

test-golden-files: ## 📸 Run golden file regression tests
	@echo "📸 Running golden file regression tests..."
	$(PYTEST) $(PYTEST_ARGS) --log-cli-level=$(TEST_LOG_LEVEL) -m "golden_test"

test-integration: .check-semantic-release ## 🔗 Run integration tests
	@echo "🔗 Running integration tests..."
	$(PYTEST) $(PYTEST_ARGS) --log-cli-level=$(TEST_LOG_LEVEL) -m "integration"

test-api: .check-api-key ## 🔑 Run API tests (requires GEMINI_API_KEY)
	@echo "🔑 Running API integration tests..."
	$(PYTEST) $(PYTEST_ARGS) --log-cli-level=$(TEST_LOG_LEVEL) -m "api"

test-workflows: ## 🔧 Run workflow configuration tests
	@echo "🔧 Running workflow configuration tests..."
	$(PYTEST) $(PYTEST_ARGS) --log-cli-level=$(TEST_LOG_LEVEL) -m "workflows"

# ------------------------------------------------------------------------------
# Prerequisite Checks (Internal)
# ------------------------------------------------------------------------------
.PHONY: .check-api-key .check-semantic-release

.check-api-key:
	@if [ -z "$$GEMINI_API_KEY" ]; then \
		echo "❌ ERROR: GEMINI_API_KEY is not set."; \
		echo "   Get a key from https://ai.dev/ and export the variable."; \
		exit 1; \
	fi

.check-semantic-release:
	@if ! command -v semantic-release >/dev/null 2>&1; then \
		echo "⚠️ WARNING: semantic-release not found, skipping integration tests."; \
		echo "   Install with: pip install python-semantic-release"; \
		exit 0; \
	fi
