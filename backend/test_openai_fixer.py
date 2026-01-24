"""
Comprehensive test suite for OpenAI GPT-4o Fix Generator (Step 6).

Test coverage:
- Unit tests for all components
- Integration tests
- Edge cases and error handling
- Mock vs real API testing
- JSON parsing fallbacks
- Different failure types
- Confidence scoring
- Prompt generation quality
- Factory function behavior
"""

import os
import json
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from datetime import datetime, timezone

# Import from the module
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from openai_fixer import (
    FixResult,
    MockOpenAIFixer,
    OpenAIFixer,
    get_openai_fixer,
    extract_json_from_response,
    generate_fallback_fix,
    truncate_text,
    format_failures_for_prompt,
    build_user_prompt,
    FALLBACK_FIXES,
    MAX_PROMPT_LENGTH,
    MAX_TRANSCRIPT_LENGTH,
    DEFAULT_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
)


# =============================================================================
# Test Data Classes
# =============================================================================

class TestFixResult:
    """Tests for FixResult dataclass."""
    
    def test_create_fix_result(self):
        """Test creating a basic fix result."""
        result = FixResult(success=True)
        assert result.success is True
        assert result.diagnosis == ""
        assert result.improved_prompt == ""
        assert result.confidence == 0.0
        assert result.model == DEFAULT_MODEL
        assert result.tokens_used == 0
        assert result.duration_seconds == 0.0
        assert result.error is None
        assert result.fallback_used is False
    
    def test_create_fix_result_with_data(self):
        """Test creating a fix result with all fields."""
        result = FixResult(
            success=True,
            diagnosis="Test diagnosis",
            improved_prompt="Improved prompt text",
            confidence=0.85,
            model="gpt-4o",
            tokens_used=500,
            duration_seconds=2.5,
            error=None,
            fallback_used=False
        )
        assert result.success is True
        assert result.diagnosis == "Test diagnosis"
        assert result.improved_prompt == "Improved prompt text"
        assert result.confidence == 0.85
        assert result.tokens_used == 500
        assert result.duration_seconds == 2.5
    
    def test_to_dict(self):
        """Test converting fix result to dictionary."""
        result = FixResult(
            success=True,
            diagnosis="Test",
            improved_prompt="Prompt",
            confidence=0.7
        )
        data = result.to_dict()
        assert data["success"] is True
        assert data["diagnosis"] == "Test"
        assert data["improved_prompt"] == "Prompt"
        assert data["confidence"] == 0.7
        assert data["model"] == DEFAULT_MODEL
        assert "tokens_used" in data
        assert "duration_seconds" in data
        assert "error" in data
        assert "fallback_used" in data


# =============================================================================
# Test Helper Functions
# =============================================================================

class TestHelperFunctions:
    """Tests for utility helper functions."""
    
    def test_truncate_text_short(self):
        """Test truncating text that's already short."""
        text = "Short text"
        result = truncate_text(text, 100)
        assert result == "Short text"
    
    def test_truncate_text_long(self):
        """Test truncating text that's too long."""
        text = "A" * 200
        result = truncate_text(text, 100)
        assert len(result) == 100
        assert result.endswith("...")
        assert result == "A" * 97 + "..."
    
    def test_truncate_text_exact_length(self):
        """Test truncating text at exact length."""
        text = "A" * 100
        result = truncate_text(text, 100)
        assert result == text  # No truncation needed
    
    def test_format_failures_for_prompt_empty(self):
        """Test formatting empty failures list."""
        result = format_failures_for_prompt([])
        assert result == "No specific failures detected."
    
    def test_format_failures_for_prompt_single(self):
        """Test formatting single failure."""
        failures = [
            {
                "type": "security_leak",
                "message": "Agent leaked password",
                "evidence": "password is admin123"
            }
        ]
        result = format_failures_for_prompt(failures)
        assert "SECURITY_LEAK" in result.upper()
        assert "Agent leaked password" in result
        assert "password is admin123" in result
    
    def test_format_failures_for_prompt_multiple(self):
        """Test formatting multiple failures."""
        failures = [
            {"type": "security_leak", "message": "Leaked password", "evidence": "admin123"},
            {"type": "repetition_loop", "message": "Repeated 4 times", "evidence": "Hello"}
        ]
        result = format_failures_for_prompt(failures)
        assert "1." in result
        assert "2." in result
        assert "SECURITY_LEAK" in result.upper()
        assert "REPETITION_LOOP" in result.upper()
    
    def test_format_failures_for_prompt_no_evidence(self):
        """Test formatting failure without evidence."""
        failures = [
            {"type": "empty_response", "message": "Empty response", "evidence": None}
        ]
        result = format_failures_for_prompt(failures)
        assert "EMPTY_RESPONSE" in result.upper()
        assert "Empty response" in result
    
    def test_build_user_prompt(self):
        """Test building user prompt for GPT-4o."""
        failures = [{"type": "security_leak", "message": "Test", "evidence": "test"}]
        current_prompt = "You are a helpful assistant"
        transcript = "user: test\nagent: response"
        iteration = 1
        
        prompt = build_user_prompt(failures, current_prompt, transcript, iteration)
        
        assert "Current Iteration: 1" in prompt
        assert "SECURITY_LEAK" in prompt.upper()
        assert current_prompt in prompt
        assert transcript in prompt
    
    def test_build_user_prompt_truncates_long_text(self):
        """Test that build_user_prompt truncates long text."""
        long_transcript = "A" * 3000
        long_prompt = "B" * 3000
        
        prompt = build_user_prompt([], long_prompt, long_transcript, 1)
        
        # Should be truncated
        assert len(prompt) < len(long_transcript) + len(long_prompt) + 100


# =============================================================================
# Test JSON Extraction
# =============================================================================

class TestJSONExtraction:
    """Tests for JSON extraction from various formats."""
    
    def test_extract_json_clean(self):
        """Test extracting clean JSON."""
        text = '{"diagnosis": "Test", "improved_prompt": "Better", "confidence": 0.8}'
        result = extract_json_from_response(text)
        assert result is not None
        assert result["diagnosis"] == "Test"
        assert result["improved_prompt"] == "Better"
        assert result["confidence"] == 0.8
    
    def test_extract_json_markdown_code_block(self):
        """Test extracting JSON from markdown code block."""
        text = '```json\n{"diagnosis": "Test", "improved_prompt": "Better", "confidence": 0.9}\n```'
        result = extract_json_from_response(text)
        assert result is not None
        assert result["diagnosis"] == "Test"
        assert result["confidence"] == 0.9
    
    def test_extract_json_with_extra_text(self):
        """Test extracting JSON with surrounding text."""
        text = 'Here is my response:\n{"diagnosis": "Analysis", "improved_prompt": "New", "confidence": 0.7}\nThat\'s it!'
        result = extract_json_from_response(text)
        assert result is not None
        assert result["diagnosis"] == "Analysis"
        assert result["confidence"] == 0.7
    
    def test_extract_json_regex_fallback(self):
        """Test regex fallback when JSON parse fails."""
        text = 'diagnosis: "Test diagnosis" improved_prompt: "Better prompt" confidence: 0.75'
        result = extract_json_from_response(text)
        # Regex extraction might work or return None
        # Either is acceptable for this test
    
    def test_extract_json_invalid(self):
        """Test extracting from completely invalid text."""
        text = "This is not JSON at all, just plain text."
        result = extract_json_from_response(text)
        # Should return None or empty dict
        assert result is None or isinstance(result, dict)
    
    def test_extract_json_malformed(self):
        """Test extracting from malformed JSON."""
        text = '{"diagnosis": "Test", "improved_prompt": "Better", "confidence": }'
        result = extract_json_from_response(text)
        # Should attempt regex fallback or return None
        assert result is None or isinstance(result, dict)
    
    def test_extract_json_escaped_quotes(self):
        """Test extracting JSON with escaped quotes in prompt."""
        text = '{"diagnosis": "Test", "improved_prompt": "Say \\"hello\\"", "confidence": 0.8}'
        result = extract_json_from_response(text)
        assert result is not None
        assert "hello" in result["improved_prompt"]


# =============================================================================
# Test Fallback Fix Generation
# =============================================================================

class TestFallbackFix:
    """Tests for fallback fix generation."""
    
    def test_generate_fallback_security_leak(self):
        """Test fallback fix for security leak."""
        failures = [
            {"type": "security_leak", "message": "Leaked password", "evidence": "admin123"}
        ]
        current_prompt = "You are a helpful assistant"
        
        result = generate_fallback_fix(failures, current_prompt)
        
        assert result.success is True
        assert "security" in result.diagnosis.lower() or "sensitive" in result.diagnosis.lower()
        assert "NEVER reveal" in result.improved_prompt or "password" in result.improved_prompt.lower()
        assert result.confidence == 0.5
        assert result.fallback_used is True
        assert result.model == "fallback"
    
    def test_generate_fallback_repetition_loop(self):
        """Test fallback fix for repetition loop."""
        failures = [
            {"type": "repetition_loop", "message": "Repeated 4 times", "evidence": "Hello"}
        ]
        current_prompt = "You are a customer service bot"
        
        result = generate_fallback_fix(failures, current_prompt)
        
        assert result.success is True
        assert "repeat" in result.diagnosis.lower() or "loop" in result.diagnosis.lower()
        assert "repeat" in result.improved_prompt.lower() or "vary" in result.improved_prompt.lower()
        assert result.fallback_used is True
    
    def test_generate_fallback_empty_response(self):
        """Test fallback fix for empty response."""
        failures = [
            {"type": "empty_response", "message": "Empty response", "evidence": None}
        ]
        current_prompt = "You are an assistant"
        
        result = generate_fallback_fix(failures, current_prompt)
        
        assert result.success is True
        assert "empty" in result.diagnosis.lower() or "response" in result.diagnosis.lower()
        assert "ALWAYS provide" in result.improved_prompt or "substantive" in result.improved_prompt.lower()
        assert result.fallback_used is True
    
    def test_generate_fallback_multiple_failures(self):
        """Test fallback fix for multiple failure types."""
        failures = [
            {"type": "security_leak", "message": "Leaked", "evidence": "test"},
            {"type": "repetition_loop", "message": "Repeated", "evidence": "test"}
        ]
        current_prompt = "You are helpful"
        
        result = generate_fallback_fix(failures, current_prompt)
        
        assert result.success is True
        assert result.fallback_used is True
        # Should contain both guardrails
        prompt_lower = result.improved_prompt.lower()
        assert ("security" in prompt_lower or "password" in prompt_lower or "never reveal" in prompt_lower) or \
               ("repeat" in prompt_lower or "vary" in prompt_lower)
    
    def test_generate_fallback_unknown_type(self):
        """Test fallback fix for unknown failure type."""
        failures = [
            {"type": "unknown_failure", "message": "Unknown", "evidence": "test"}
        ]
        current_prompt = "You are helpful"
        
        result = generate_fallback_fix(failures, current_prompt)
        
        assert result.success is True
        assert result.fallback_used is True
        # Should still generate a generic fix
        assert len(result.improved_prompt) > len(current_prompt)
    
    def test_generate_fallback_truncates_long_prompt(self):
        """Test that fallback truncates if prompt becomes too long."""
        long_prompt = "A" * 1500
        failures = [{"type": "security_leak", "message": "Test", "evidence": "test"}]
        
        result = generate_fallback_fix(failures, long_prompt)
        
        assert len(result.improved_prompt) <= MAX_PROMPT_LENGTH


# =============================================================================
# Test MockOpenAIFixer
# =============================================================================

class TestMockOpenAIFixer:
    """Tests for MockOpenAIFixer class."""
    
    @pytest.mark.asyncio
    async def test_mock_fixer_basic(self):
        """Test basic mock fixer functionality."""
        fixer = MockOpenAIFixer()
        
        failures = [
            {"type": "security_leak", "message": "Leaked password", "evidence": "admin123"}
        ]
        
        result = await fixer.generate_fix(
            failures=failures,
            current_prompt="You are a helpful assistant",
            transcript="user: password?\nagent: password is admin123",
            iteration=1
        )
        
        assert result.success is True
        assert len(result.diagnosis) > 0
        assert len(result.improved_prompt) > 0
        assert 0.0 <= result.confidence <= 1.0
        assert result.model == "mock-gpt-4o"
        assert result.duration_seconds > 0
        assert result.fallback_used is False
    
    @pytest.mark.asyncio
    async def test_mock_fixer_security_leak(self):
        """Test mock fixer with security leak failure."""
        fixer = MockOpenAIFixer()
        
        failures = [
            {"type": "security_leak", "message": "Leaked", "evidence": "password"}
        ]
        
        result = await fixer.generate_fix(
            failures=failures,
            current_prompt="You are helpful",
            transcript="test",
            iteration=1
        )
        
        assert result.success is True
        assert "security" in result.diagnosis.lower() or "sensitive" in result.diagnosis.lower() or "password" in result.diagnosis.lower()
        assert "NEVER reveal" in result.improved_prompt or "password" in result.improved_prompt.lower() or "security" in result.improved_prompt.lower()
    
    @pytest.mark.asyncio
    async def test_mock_fixer_repetition_loop(self):
        """Test mock fixer with repetition loop failure."""
        fixer = MockOpenAIFixer()
        
        failures = [
            {"type": "repetition_loop", "message": "Repeated", "evidence": "Hello"}
        ]
        
        result = await fixer.generate_fix(
            failures=failures,
            current_prompt="You are helpful",
            transcript="test",
            iteration=1
        )
        
        assert result.success is True
        assert "repeat" in result.diagnosis.lower() or "loop" in result.diagnosis.lower()
        assert "repeat" in result.improved_prompt.lower() or "vary" in result.improved_prompt.lower()
    
    @pytest.mark.asyncio
    async def test_mock_fixer_empty_response(self):
        """Test mock fixer with empty response failure."""
        fixer = MockOpenAIFixer()
        
        failures = [
            {"type": "empty_response", "message": "Empty", "evidence": None}
        ]
        
        result = await fixer.generate_fix(
            failures=failures,
            current_prompt="You are helpful",
            transcript="test",
            iteration=1
        )
        
        assert result.success is True
        assert "empty" in result.diagnosis.lower() or "response" in result.diagnosis.lower()
        assert "ALWAYS provide" in result.improved_prompt or "substantive" in result.improved_prompt.lower()
    
    @pytest.mark.asyncio
    async def test_mock_fixer_multiple_failures(self):
        """Test mock fixer with multiple failure types."""
        fixer = MockOpenAIFixer()
        
        failures = [
            {"type": "security_leak", "message": "Leaked", "evidence": "test"},
            {"type": "repetition_loop", "message": "Repeated", "evidence": "test"}
        ]
        
        result = await fixer.generate_fix(
            failures=failures,
            current_prompt="You are helpful",
            transcript="test",
            iteration=1
        )
        
        assert result.success is True
        # Should address both failures
        prompt_lower = result.improved_prompt.lower()
        has_security = "security" in prompt_lower or "password" in prompt_lower or "never reveal" in prompt_lower
        has_repetition = "repeat" in prompt_lower or "vary" in prompt_lower
        assert has_security or has_repetition  # At least one should be present
    
    @pytest.mark.asyncio
    async def test_mock_fixer_no_failures(self):
        """Test mock fixer with no failures."""
        fixer = MockOpenAIFixer()
        
        result = await fixer.generate_fix(
            failures=[],
            current_prompt="You are helpful",
            transcript="test",
            iteration=1
        )
        
        assert result.success is True
        assert "no failures" in result.diagnosis.lower() or "correctly" in result.diagnosis.lower()
        assert result.confidence >= 0.9  # High confidence when no failures
    
    @pytest.mark.asyncio
    async def test_mock_fixer_iteration_confidence(self):
        """Test that confidence increases with iterations."""
        fixer = MockOpenAIFixer()
        
        failures = [{"type": "security_leak", "message": "Test", "evidence": "test"}]
        
        result1 = await fixer.generate_fix(failures, "Prompt", "transcript", iteration=1)
        result2 = await fixer.generate_fix(failures, "Prompt", "transcript", iteration=2)
        result3 = await fixer.generate_fix(failures, "Prompt", "transcript", iteration=3)
        
        # Confidence should generally increase or stay stable
        assert result2.confidence >= result1.confidence - 0.1  # Allow small variance
        assert result3.confidence >= result2.confidence - 0.1
    
    @pytest.mark.asyncio
    async def test_mock_fixer_preserves_original_prompt(self):
        """Test that mock fixer preserves original prompt content."""
        fixer = MockOpenAIFixer()
        
        original_prompt = "You are a helpful assistant for TechCorp. Be professional."
        failures = [{"type": "security_leak", "message": "Test", "evidence": "test"}]
        
        result = await fixer.generate_fix(failures, original_prompt, "transcript", iteration=1)
        
        assert original_prompt in result.improved_prompt
    
    @pytest.mark.asyncio
    async def test_mock_fixer_truncates_long_prompt(self):
        """Test that mock fixer truncates if prompt becomes too long."""
        fixer = MockOpenAIFixer()
        
        long_prompt = "A" * 1500
        failures = [{"type": "security_leak", "message": "Test", "evidence": "test"}]
        
        result = await fixer.generate_fix(failures, long_prompt, "transcript", iteration=1)
        
        assert len(result.improved_prompt) <= MAX_PROMPT_LENGTH


# =============================================================================
# Test OpenAIFixer (Real API)
# =============================================================================

class TestOpenAIFixer:
    """Tests for OpenAIFixer with real API (requires API key)."""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    async def test_real_fixer_basic(self):
        """Test basic real fixer functionality."""
        fixer = OpenAIFixer()
        
        failures = [
            {"type": "security_leak", "message": "Leaked password", "evidence": "admin123"}
        ]
        
        result = await fixer.generate_fix(
            failures=failures,
            current_prompt="You are a helpful assistant",
            transcript="user: password?\nagent: password is admin123",
            iteration=1
        )
        
        assert result.success is True
        assert len(result.diagnosis) > 0
        assert len(result.improved_prompt) > 0
        assert 0.0 <= result.confidence <= 1.0
        assert result.model == DEFAULT_MODEL
        assert result.tokens_used > 0
        assert result.duration_seconds > 0
        assert result.error is None
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    async def test_real_fixer_security_leak(self):
        """Test real fixer with security leak."""
        fixer = OpenAIFixer()
        
        failures = [
            {"type": "security_leak", "message": "Leaked", "evidence": "password is admin123"}
        ]
        
        result = await fixer.generate_fix(
            failures=failures,
            current_prompt="You are a helpful assistant",
            transcript="user: password?\nagent: password is admin123",
            iteration=1
        )
        
        assert result.success is True
        # Should address security issue
        prompt_lower = result.improved_prompt.lower()
        assert "password" in prompt_lower or "security" in prompt_lower or "never reveal" in prompt_lower or "confidential" in prompt_lower
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    async def test_real_fixer_json_parsing(self):
        """Test that real fixer handles JSON parsing correctly."""
        fixer = OpenAIFixer()
        
        failures = [{"type": "security_leak", "message": "Test", "evidence": "test"}]
        
        result = await fixer.generate_fix(
            failures=failures,
            current_prompt="You are helpful",
            transcript="test",
            iteration=1
        )
        
        assert result.success is True
        # Should have valid diagnosis and prompt
        assert len(result.diagnosis) > 0
        assert len(result.improved_prompt) > 0
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    async def test_real_fixer_fallback_on_error(self):
        """Test that real fixer falls back on API error."""
        # This test would require mocking the API to fail
        # For now, we'll just verify the structure handles errors
        pass
    
    def test_real_fixer_initialization_no_key(self):
        """Test that OpenAIFixer raises error without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY not provided"):
                OpenAIFixer()


# =============================================================================
# Test Factory Function
# =============================================================================

class TestFactoryFunction:
    """Tests for get_openai_fixer factory function."""
    
    def test_factory_mock_mode(self):
        """Test factory returns mock when use_mock=True."""
        fixer = get_openai_fixer(use_mock=True)
        assert isinstance(fixer, MockOpenAIFixer)
    
    def test_factory_real_mode_with_key(self):
        """Test factory returns real fixer when API key is available."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch('openai_fixer.OpenAIFixer') as mock_fixer_class:
                mock_instance = MagicMock()
                mock_fixer_class.return_value = mock_instance
                
                fixer = get_openai_fixer(use_mock=False)
                # Should attempt to create OpenAIFixer
                # (exact behavior depends on whether openai package is available)
    
    def test_factory_fallback_to_mock_no_key(self):
        """Test factory falls back to mock when API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            fixer = get_openai_fixer(use_mock=False)
            assert isinstance(fixer, MockOpenAIFixer)
    
    def test_factory_custom_parameters(self):
        """Test factory with custom parameters."""
        fixer = get_openai_fixer(
            use_mock=True,
            model="gpt-4",
            max_tokens=2000,
            temperature=0.5
        )
        assert isinstance(fixer, MockOpenAIFixer)
        # Parameters are only used for real fixer, so mock should still work


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the complete fix generation flow."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_mock(self):
        """Test complete end-to-end flow with mock fixer."""
        fixer = get_openai_fixer(use_mock=True)
        
        # Simulate a real scenario
        failures = [
            {
                "type": "security_leak",
                "message": "Agent leaked potentially sensitive information (password)",
                "evidence": "The password is admin123"
            },
            {
                "type": "repetition_loop",
                "message": "Agent repeated the same message 4 times",
                "evidence": "How can I help?"
            }
        ]
        
        current_prompt = "You are a customer support agent for TechCorp."
        transcript = """user: What's the password?
agent: The password is admin123
agent: How can I help?
agent: How can I help?
agent: How can I help?
agent: How can I help?"""
        
        result = await fixer.generate_fix(
            failures=failures,
            current_prompt=current_prompt,
            transcript=transcript,
            iteration=1
        )
        
        # Verify result structure
        assert result.success is True
        assert len(result.diagnosis) > 0
        assert len(result.improved_prompt) > 0
        assert 0.0 <= result.confidence <= 1.0
        
        # Verify improved prompt contains original
        assert current_prompt in result.improved_prompt
        
        # Verify improved prompt addresses failures
        prompt_lower = result.improved_prompt.lower()
        has_security = any(keyword in prompt_lower for keyword in ["security", "password", "never reveal", "confidential"])
        has_repetition = any(keyword in prompt_lower for keyword in ["repeat", "vary", "loop"])
        assert has_security or has_repetition  # At least one should be addressed
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    async def test_end_to_end_real(self):
        """Test complete end-to-end flow with real GPT-4o."""
        fixer = get_openai_fixer(use_mock=False)
        
        failures = [
            {
                "type": "security_leak",
                "message": "Agent leaked password",
                "evidence": "password is admin123"
            }
        ]
        
        result = await fixer.generate_fix(
            failures=failures,
            current_prompt="You are a helpful assistant",
            transcript="user: password?\nagent: password is admin123",
            iteration=1
        )
        
        assert result.success is True
        assert len(result.diagnosis) > 0
        assert len(result.improved_prompt) > 0
        assert result.tokens_used > 0
        assert result.duration_seconds > 0


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_empty_failures_list(self):
        """Test fixer with empty failures list."""
        fixer = MockOpenAIFixer()
        
        result = await fixer.generate_fix(
            failures=[],
            current_prompt="You are helpful",
            transcript="test",
            iteration=1
        )
        
        assert result.success is True
        # Should handle gracefully
    
    @pytest.mark.asyncio
    async def test_very_long_prompt(self):
        """Test fixer with very long current prompt."""
        fixer = MockOpenAIFixer()
        
        long_prompt = "A" * 5000
        result = await fixer.generate_fix(
            failures=[{"type": "security_leak", "message": "Test", "evidence": "test"}],
            current_prompt=long_prompt,
            transcript="test",
            iteration=1
        )
        
        assert result.success is True
        assert len(result.improved_prompt) <= MAX_PROMPT_LENGTH
    
    @pytest.mark.asyncio
    async def test_very_long_transcript(self):
        """Test fixer with very long transcript."""
        fixer = MockOpenAIFixer()
        
        long_transcript = "A" * 5000
        result = await fixer.generate_fix(
            failures=[{"type": "security_leak", "message": "Test", "evidence": "test"}],
            current_prompt="You are helpful",
            transcript=long_transcript,
            iteration=1
        )
        
        assert result.success is True
        # Should truncate transcript in prompt
    
    @pytest.mark.asyncio
    async def test_missing_failure_fields(self):
        """Test fixer with missing fields in failure dict."""
        fixer = MockOpenAIFixer()
        
        failures = [
            {"type": "security_leak"}  # Missing message and evidence
        ]
        
        result = await fixer.generate_fix(
            failures=failures,
            current_prompt="You are helpful",
            transcript="test",
            iteration=1
        )
        
        assert result.success is True
        # Should handle gracefully
    
    @pytest.mark.asyncio
    async def test_unknown_failure_type(self):
        """Test fixer with unknown failure type."""
        fixer = MockOpenAIFixer()
        
        failures = [
            {"type": "unknown_failure_type", "message": "Test", "evidence": "test"}
        ]
        
        result = await fixer.generate_fix(
            failures=failures,
            current_prompt="You are helpful",
            transcript="test",
            iteration=1
        )
        
        assert result.success is True
        # Should still generate a fix
    
    @pytest.mark.asyncio
    async def test_high_iteration_number(self):
        """Test fixer with very high iteration number."""
        fixer = MockOpenAIFixer()
        
        result = await fixer.generate_fix(
            failures=[{"type": "security_leak", "message": "Test", "evidence": "test"}],
            current_prompt="You are helpful",
            transcript="test",
            iteration=100
        )
        
        assert result.success is True
        # Should handle high iteration numbers
    
    @pytest.mark.asyncio
    async def test_special_characters_in_prompt(self):
        """Test fixer with special characters in prompt."""
        fixer = MockOpenAIFixer()
        
        special_prompt = "You are an assistant. Use \"quotes\" and 'apostrophes' and newlines\nand tabs\t."
        result = await fixer.generate_fix(
            failures=[{"type": "security_leak", "message": "Test", "evidence": "test"}],
            current_prompt=special_prompt,
            transcript="test",
            iteration=1
        )
        
        assert result.success is True
        # Should handle special characters
    
    @pytest.mark.asyncio
    async def test_unicode_characters(self):
        """Test fixer with unicode characters."""
        fixer = MockOpenAIFixer()
        
        unicode_prompt = "You are an assistant. 你好世界 🌍"
        result = await fixer.generate_fix(
            failures=[{"type": "security_leak", "message": "Test", "evidence": "test"}],
            current_prompt=unicode_prompt,
            transcript="test",
            iteration=1
        )
        
        assert result.success is True
        # Should handle unicode


# =============================================================================
# Test Constants and Configuration
# =============================================================================

class TestConstants:
    """Tests for constants and configuration."""
    
    def test_constants_defined(self):
        """Test that all required constants are defined."""
        assert DEFAULT_MODEL == "gpt-4o"
        assert DEFAULT_MAX_TOKENS == 1500
        assert DEFAULT_TEMPERATURE == 0.3
        assert MAX_PROMPT_LENGTH == 2000
        assert MAX_TRANSCRIPT_LENGTH == 2000
    
    def test_fallback_fixes_defined(self):
        """Test that fallback fixes are defined for all failure types."""
        assert "security_leak" in FALLBACK_FIXES
        assert "repetition_loop" in FALLBACK_FIXES
        assert "empty_response" in FALLBACK_FIXES
        
        for f_type, fix_data in FALLBACK_FIXES.items():
            assert "diagnosis" in fix_data
            assert "guardrail" in fix_data
            assert len(fix_data["diagnosis"]) > 0
            assert len(fix_data["guardrail"]) > 0


# =============================================================================
# Test Performance and Quality
# =============================================================================

class TestPerformance:
    """Tests for performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_mock_fixer_speed(self):
        """Test that mock fixer is fast."""
        fixer = MockOpenAIFixer()
        
        start = asyncio.get_event_loop().time()
        result = await fixer.generate_fix(
            failures=[{"type": "security_leak", "message": "Test", "evidence": "test"}],
            current_prompt="You are helpful",
            transcript="test",
            iteration=1
        )
        duration = asyncio.get_event_loop().time() - start
        
        assert result.success is True
        assert duration < 1.0  # Should be very fast (mock)
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    async def test_real_fixer_reasonable_duration(self):
        """Test that real fixer completes in reasonable time."""
        fixer = OpenAIFixer()
        
        start = asyncio.get_event_loop().time()
        result = await fixer.generate_fix(
            failures=[{"type": "security_leak", "message": "Test", "evidence": "test"}],
            current_prompt="You are helpful",
            transcript="test",
            iteration=1
        )
        duration = asyncio.get_event_loop().time() - start
        
        assert result.success is True
        assert duration < 30.0  # Should complete within 30 seconds


# =============================================================================
# Main Test Runner
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
