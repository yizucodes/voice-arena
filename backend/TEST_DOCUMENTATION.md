# Test Documentation for Self-Healing Orchestrator (Step 7)

## Overview

This document describes the comprehensive test suite for the Self-Healing Orchestrator module (`healer.py`). The test suite is designed by a senior QA engineer to ensure high quality, reliability, and maintainability.

## Test Strategy

### Test Pyramid

```
        /\
       /  \     E2E Tests (Manual/API)
      /____\
     /      \   Integration Tests
    /________\
   /          \  Unit Tests
  /____________\
```

### Test Categories

1. **Unit Tests** - Test individual components in isolation
2. **Integration Tests** - Test component interactions
3. **End-to-End Tests** - Test complete workflows
4. **Performance Tests** - Test timing and resource usage
5. **Regression Tests** - Prevent previously fixed bugs

## Test Coverage

### 1. Data Classes (`TestIterationResult`, `TestHealingResult`)
- ✅ Creation and initialization
- ✅ Dictionary serialization
- ✅ Field validation
- ✅ Edge cases (None values, empty lists)

### 2. Initialization (`TestAutonomousHealerInitialization`)
- ✅ Default parameters
- ✅ Custom parameters
- ✅ Parameter bounds validation
- ✅ Client initialization

### 3. Full Healing Loop (`TestFullHealingLoop`)
- ✅ Security leak detection and healing
- ✅ Repetition loop detection and healing
- ✅ No failures scenario (immediate pass)
- ✅ Max iterations reached
- ✅ Prompt improvement across iterations

### 4. Callback Functionality (`TestCallbackFunctionality`)
- ✅ Sync callback execution
- ✅ Async callback execution
- ✅ Callback error handling (shouldn't break loop)
- ✅ Callback invocation count

### 5. Error Handling (`TestErrorHandling`)
- ✅ Conversation test failures
- ✅ Fix generation failures
- ✅ Sandbox creation failures
- ✅ Empty test input
- ✅ Very long prompts

### 6. Multi-Test Healing (`TestMultiTestHealing`)
- ✅ Sequential test execution
- ✅ Prompt carryover between tests
- ✅ Empty input list handling

### 7. Factory Functions (`TestFactoryFunctions`)
- ✅ `create_healer()` factory
- ✅ `quick_heal()` utility

### 8. Sandbox Functionality (`TestSandboxFunctionality`)
- ✅ Sandbox disabled mode
- ✅ Sandbox enabled mode (mock)

### 9. Performance (`TestPerformance`)
- ✅ Healing duration tracking
- ✅ Iteration duration tracking
- ✅ Reasonable completion times

### 10. Data Integrity (`TestDataIntegrity`)
- ✅ Session ID uniqueness
- ✅ Iteration numbering
- ✅ Final prompt consistency
- ✅ Duration consistency

### 11. Edge Cases (`TestEdgeCases`)
- ✅ Single iteration max
- ✅ Very long test inputs
- ✅ Special characters in input

### 12. Mock vs Real API (`TestMockVsReal`)
- ✅ Mock mode behavior
- ✅ Real mode behavior (requires API keys)

### 13. Regression Scenarios (`TestRegressionScenarios`)
- ✅ Multiple security leaks
- ✅ Combined failure types

### 14. Configuration Validation (`TestConfigurationValidation`)
- ✅ Default configuration
- ✅ Verbose logging

## Running Tests

### Quick Start

```bash
# Run all tests
./run_tests.sh all

# Run only unit tests
./run_tests.sh unit

# Run only integration tests
./run_tests.sh integration

# Run quick tests (exclude slow ones)
./run_tests.sh quick

# Run with coverage
./run_tests.sh coverage
```

### Using pytest directly

```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests
pytest test_healer.py -v

# Run specific test class
pytest test_healer.py::TestFullHealingLoop -v

# Run specific test
pytest test_healer.py::TestFullHealingLoop::test_security_leak_healing -v

# Run with markers
pytest test_healer.py -m "unit" -v
pytest test_healer.py -m "integration" -v

# Run with coverage
pytest test_healer.py --cov=healer --cov-report=html
```

## Test Fixtures

The test suite uses pytest fixtures for common test data:

- `sample_prompt` - Standard agent prompt
- `security_test_input` - Input that triggers security leak
- `repetition_test_input` - Input that triggers repetition loop
- `good_test_input` - Input that should pass
- `mock_conversation_result_*` - Mock conversation results
- `mock_fix_result` - Mock fix generation result

## Mock vs Real API Testing

### Mock Mode (Default)
- ✅ Fast execution
- ✅ No API costs
- ✅ Deterministic behavior
- ✅ Works without API keys

### Real Mode (Manual)
- ⚠️ Requires API keys
- ⚠️ Slower execution
- ⚠️ May incur costs
- ⚠️ Non-deterministic (API responses vary)

To run real API tests:
```bash
# Set environment variables
export OPENAI_API_KEY=sk-...
export ELEVENLABS_API_KEY=...
export DAYTONA_API_KEY=...

# Run with real APIs (tests marked with @pytest.mark.skip)
pytest test_healer.py -v -m "requires_api"
```

## Test Metrics

### Coverage Goals
- **Unit Tests**: >90% code coverage
- **Integration Tests**: All critical paths covered
- **Edge Cases**: All identified edge cases tested

### Performance Benchmarks
- Mock mode: < 5 seconds for 3 iterations
- Real mode: < 30 seconds for 3 iterations (depends on API)

## Known Limitations

1. **Real API Tests**: Marked as `@pytest.mark.skip` by default
   - Requires manual execution with API keys
   - May incur costs
   - Results may vary

2. **Sandbox Tests**: Limited in mock mode
   - Real sandbox tests require Daytona API key
   - Mock sandbox tests verify logic only

3. **Timing Tests**: May be flaky
   - Use generous timeouts
   - Focus on relative timing, not absolute

## Best Practices

1. **Isolation**: Each test should be independent
2. **Determinism**: Tests should produce consistent results
3. **Speed**: Keep tests fast (use mocks when possible)
4. **Clarity**: Test names should describe what they test
5. **Maintenance**: Update tests when code changes

## Troubleshooting

### Tests fail with import errors
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Tests timeout
- Check network connectivity (for real API tests)
- Increase timeout values if needed
- Use mock mode for faster tests

### Callback tests fail
- Ensure callbacks are properly awaited
- Check for exception handling in callbacks

## Continuous Integration

For CI/CD pipelines, use:

```yaml
# Example GitHub Actions
- name: Run tests
  run: |
    source venv/bin/activate
    pytest test_healer.py -v --tb=short
```

## Contributing

When adding new features:

1. ✅ Write tests first (TDD approach)
2. ✅ Cover happy path and error cases
3. ✅ Add edge case tests
4. ✅ Update this documentation
5. ✅ Ensure all tests pass

## Test Maintenance

- **Weekly**: Run full test suite
- **Before Release**: Run with coverage
- **After Bug Fixes**: Add regression tests
- **Monthly**: Review and update test cases
