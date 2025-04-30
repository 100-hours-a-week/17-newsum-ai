# Testing Framework for 17-team-4cut AI

This directory contains tests for the AI components of the 17-team-4cut project, which generates 4-panel comics from news articles.

## Test Structure

The tests are organized as follows:

- `agents/`: Unit tests for individual agent components
  - `test_sentiment_analyzer_agent.py`: Tests for the YouTube sentiment analysis agent
  - `test_humorator_agent.py`: Tests for the empathetic humor generation agent
  - `test_scenariowriter_agent.py`: Tests for the comic scenario creation agent

- Individual test scripts for manual testing:
  - `test_sentiment_analyzer.py`: Direct test of the sentiment analyzer agent
  - `test_humorator.py`: Direct test of the humorator agent
  - `test_scenariowriter.py`: Direct test of the scenario writer agent
  - `test_end_to_end.py`: End-to-end workflow test

- Support files:
  - `run_tests.py`: Script to run all or individual tests
  - `test_workflow.py`: Integration tests for the complete workflow

## Running Tests

### Running All Tests

To run all unit tests:

```bash
python tests/run_tests.py
```

### Running Individual Tests

To run a specific test file:

```bash
python tests/run_tests.py tests/agents/test_sentiment_analyzer_agent.py
```

### Running Manual Tests

These scripts test individual agents with real input data:

```bash
# Test sentiment analyzer
python tests/test_sentiment_analyzer.py

# Test humor generator
python tests/test_humorator.py

# Test scenario writer
python tests/test_scenariowriter.py

# Run end-to-end workflow test
python tests/test_end_to_end.py
```

## Test Dependencies

The tests require the following to be properly configured:

1. Environment variables set (see `.env` file)
2. LLM API endpoints configured
3. YouTube API key (for sentiment analyzer tests)

## Test Design

- **Unit Tests**: Use mock objects to test individual agent functionality without external dependencies
- **Integration Tests**: Test interactions between components
- **End-to-End Tests**: Test the complete workflow from initial query to comic generation

## Adding New Tests

When adding new agents or functionality:

1. Create unit tests in the `agents/` directory
2. Add test cases to the integration tests in `test_workflow.py`
3. Create manual test scripts for direct testing with real data
