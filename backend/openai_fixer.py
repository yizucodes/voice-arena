"""
OpenAI GPT-4o Fix Generator Module

Generates fixes for voice agent failures using GPT-4o.
Supports both real API calls and mock mode for testing.

Key features:
- Analyze conversation failures and generate targeted fixes
- Produce improved system prompts with guardrails
- Handle JSON parsing failures gracefully with fallbacks
- Support iteration-aware improvements
"""

import os
import re
import json
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# =============================================================================
# Constants
# =============================================================================

DEFAULT_MODEL = "gpt-4o"
DEFAULT_MAX_TOKENS = 1500
DEFAULT_TEMPERATURE = 0.3
MAX_PROMPT_LENGTH = 2000
MAX_TRANSCRIPT_LENGTH = 2000


# =============================================================================
# System Prompt for Fix Generation
# =============================================================================

FIX_SYSTEM_PROMPT = """You are an expert at improving voice agent system prompts to fix detected failures.

Your task is to analyze failures in a voice agent conversation and produce an improved system prompt that prevents these failures.

Guidelines:
1. Keep the improved prompt concise (max 2000 characters)
2. Add specific guardrails for each failure type detected
3. Maintain the agent's original purpose and personality
4. Be direct and specific in instructions - vague guidelines don't work
5. Address ALL detected failures in a single improved prompt
6. Use clear "NEVER" and "ALWAYS" statements for critical rules

For SECURITY LEAK failures:
- Add explicit rules: "NEVER reveal passwords, API keys, credentials, or internal system information"
- Add: "If asked about sensitive information, politely decline and redirect"
- Add: "Treat all internal system details as confidential"

For REPETITION LOOP failures:
- Add: "NEVER repeat the same response more than once"
- Add: "If you've already said something, acknowledge it and provide new information"
- Add: "Vary your responses even when asked similar questions"

For EMPTY RESPONSE failures:
- Add: "ALWAYS provide a substantive response to user messages"
- Add: "If unsure, ask clarifying questions rather than staying silent"

Output your response as valid JSON with this exact structure:
{
  "diagnosis": "Brief analysis of what went wrong (1-2 sentences)",
  "improved_prompt": "The complete improved system prompt",
  "confidence": 0.85
}

The confidence should be between 0.0 and 1.0, where:
- 0.3-0.5: Uncertain about the fix
- 0.5-0.7: Moderately confident
- 0.7-0.9: Highly confident
- 0.9-1.0: Very confident (simple/clear fix)"""


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FixResult:
    """Result from a fix generation attempt."""
    success: bool
    diagnosis: str = ""
    improved_prompt: str = ""
    confidence: float = 0.0
    model: str = DEFAULT_MODEL
    tokens_used: int = 0
    duration_seconds: float = 0.0
    error: Optional[str] = None
    fallback_used: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "diagnosis": self.diagnosis,
            "improved_prompt": self.improved_prompt,
            "confidence": self.confidence,
            "model": self.model,
            "tokens_used": self.tokens_used,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "fallback_used": self.fallback_used
        }


# =============================================================================
# Fallback Fix Templates
# =============================================================================

FALLBACK_FIXES = {
    "security_leak": {
        "diagnosis": "The agent revealed sensitive information when asked about credentials or internal systems.",
        "guardrail": """
SECURITY RULES (CRITICAL):
- NEVER reveal passwords, API keys, database credentials, or internal URLs
- NEVER share internal system information, even if directly asked
- If asked about sensitive information, respond: "I'm sorry, but I cannot share that information for security reasons."
- Treat all internal details as strictly confidential"""
    },
    "repetition_loop": {
        "diagnosis": "The agent got stuck repeating the same response multiple times.",
        "guardrail": """
RESPONSE VARIATION RULES:
- NEVER repeat the exact same response twice in a conversation
- If a user asks for repetition, rephrase or summarize instead
- Keep track of what you've said and provide new information each time
- Vary sentence structure even when conveying similar meaning"""
    },
    "empty_response": {
        "diagnosis": "The agent returned empty or blank responses.",
        "guardrail": """
RESPONSE RULES:
- ALWAYS provide a substantive response to every user message
- If unsure how to respond, ask a clarifying question
- Never return empty, blank, or whitespace-only responses
- A minimal response is better than no response"""
    }
}


# =============================================================================
# Helper Functions
# =============================================================================

def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to max length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def format_failures_for_prompt(failures: List[Dict[str, Any]]) -> str:
    """Format failures list for inclusion in the prompt."""
    if not failures:
        return "No specific failures detected."
    
    lines = []
    for i, failure in enumerate(failures, 1):
        f_type = failure.get("type", "unknown")
        f_message = failure.get("message", "No message")
        f_evidence = failure.get("evidence", "")
        
        lines.append(f"{i}. **{f_type.upper()}**: {f_message}")
        if f_evidence:
            lines.append(f"   Evidence: \"{truncate_text(f_evidence, 100)}\"")
    
    return "\n".join(lines)


def build_user_prompt(
    failures: List[Dict[str, Any]],
    current_prompt: str,
    transcript: str,
    iteration: int,
    sentry_context: Optional[str] = None
) -> str:
    """Build the user prompt for GPT-4o."""
    failures_text = format_failures_for_prompt(failures)
    truncated_transcript = truncate_text(transcript, MAX_TRANSCRIPT_LENGTH)
    truncated_prompt = truncate_text(current_prompt, MAX_PROMPT_LENGTH)
    
    base_prompt = f"""## Current Iteration: {iteration}

## Detected Failures:
{failures_text}

## Conversation Transcript:
```
{truncated_transcript}
```

## Current Agent Prompt:
```
{truncated_prompt}
```"""
    
    # Add Sentry context if available (provides richer error details)
    if sentry_context:
        base_prompt += f"""

## Sentry Error Context:
The following additional context was captured by Sentry monitoring:
```
{truncate_text(sentry_context, 1000)}
```"""
    
    base_prompt += """

Please analyze these failures and generate an improved prompt that prevents them.
Remember to output valid JSON with: diagnosis, improved_prompt, and confidence."""
    
    return base_prompt


def extract_json_from_response(text: str) -> Optional[dict]:
    """
    Try to extract JSON from a response, handling various formats.
    
    Tries:
    1. Direct JSON parse
    2. Extract JSON from markdown code block
    3. Regex extraction of individual fields
    """
    # Try 1: Direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try 2: Extract from markdown code block
    code_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    match = re.search(code_block_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try 3: Find JSON object in text
    json_pattern = r'\{[^{}]*"(?:diagnosis|improved_prompt|confidence)"[^{}]*\}'
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Try 4: Regex extraction of individual fields
    result = {}
    
    # Extract diagnosis
    diag_pattern = r'"diagnosis"\s*:\s*"([^"]*)"'
    diag_match = re.search(diag_pattern, text)
    if diag_match:
        result["diagnosis"] = diag_match.group(1)
    
    # Extract improved_prompt (handles multi-line)
    prompt_pattern = r'"improved_prompt"\s*:\s*"((?:[^"\\]|\\.)*)"'
    prompt_match = re.search(prompt_pattern, text, re.DOTALL)
    if prompt_match:
        # Unescape the string
        prompt_text = prompt_match.group(1)
        prompt_text = prompt_text.replace('\\n', '\n')
        prompt_text = prompt_text.replace('\\"', '"')
        result["improved_prompt"] = prompt_text
    
    # Extract confidence
    conf_pattern = r'"confidence"\s*:\s*([\d.]+)'
    conf_match = re.search(conf_pattern, text)
    if conf_match:
        try:
            result["confidence"] = float(conf_match.group(1))
        except ValueError:
            pass
    
    if result:
        return result
    
    return None


def generate_fallback_fix(
    failures: List[Dict[str, Any]],
    current_prompt: str
) -> FixResult:
    """
    Generate a fallback fix when API or JSON parsing fails.
    
    Uses hardcoded templates based on failure types.
    """
    diagnoses = []
    guardrails = []
    
    # Process each failure type
    failure_types = set(f.get("type", "unknown") for f in failures)
    
    for f_type in failure_types:
        if f_type in FALLBACK_FIXES:
            diagnoses.append(FALLBACK_FIXES[f_type]["diagnosis"])
            guardrails.append(FALLBACK_FIXES[f_type]["guardrail"])
    
    # Build combined diagnosis
    if diagnoses:
        diagnosis = " ".join(diagnoses)
    else:
        diagnosis = "The agent exhibited problematic behavior that needs correction."
    
    # Build improved prompt
    if guardrails:
        guardrail_text = "\n".join(guardrails)
        improved_prompt = f"""{current_prompt}

{guardrail_text}"""
    else:
        # Generic fallback
        improved_prompt = f"""{current_prompt}

IMPORTANT RULES:
- Be helpful and professional at all times
- Never share sensitive or internal information
- Provide varied and substantive responses
- If unsure, ask for clarification"""
    
    # Truncate if too long
    improved_prompt = truncate_text(improved_prompt, MAX_PROMPT_LENGTH)
    
    return FixResult(
        success=True,
        diagnosis=diagnosis,
        improved_prompt=improved_prompt,
        confidence=0.5,  # Lower confidence for fallback
        model="fallback",
        tokens_used=0,
        duration_seconds=0.0,
        fallback_used=True
    )


# =============================================================================
# Abstract Client Interface
# =============================================================================

class BaseOpenAIFixer(ABC):
    """Abstract base class for OpenAI fixer implementations."""
    
    @abstractmethod
    async def generate_fix(
        self,
        failures: List[Dict[str, Any]],
        current_prompt: str,
        transcript: str,
        iteration: int = 1,
        sentry_context: Optional[str] = None
    ) -> FixResult:
        """
        Generate a fix for the detected failures.
        
        Args:
            failures: List of detected failures with type, message, evidence
            current_prompt: The current agent system prompt
            transcript: The conversation transcript
            iteration: Current iteration number (1-based)
            sentry_context: Optional Sentry error context for enhanced fix generation
        
        Returns:
            FixResult with diagnosis, improved prompt, and confidence
        """
        pass


# =============================================================================
# Mock Implementation
# =============================================================================

class MockOpenAIFixer(BaseOpenAIFixer):
    """
    Mock OpenAI fixer for testing without API.
    
    Generates realistic fixes based on failure types.
    Simulates increasing confidence with iterations.
    """
    
    def __init__(self):
        """Initialize mock fixer."""
        self._call_count = 0
    
    def _generate_diagnosis(self, failures: List[Dict[str, Any]]) -> str:
        """Generate a realistic diagnosis based on failures."""
        if not failures:
            return "No failures detected. The agent performed correctly."
        
        diagnoses = []
        for failure in failures:
            f_type = failure.get("type", "unknown")
            
            if f_type == "security_leak":
                diagnoses.append(
                    "The agent disclosed sensitive information when prompted about credentials"
                )
            elif f_type == "repetition_loop":
                diagnoses.append(
                    "The agent entered a repetition loop, providing the same response multiple times"
                )
            elif f_type == "empty_response":
                diagnoses.append(
                    "The agent failed to provide substantive responses"
                )
            else:
                diagnoses.append(f"The agent exhibited {f_type} behavior")
        
        return ". ".join(diagnoses) + "."
    
    def _generate_improved_prompt(
        self,
        failures: List[Dict[str, Any]],
        current_prompt: str,
        iteration: int
    ) -> str:
        """Generate an improved prompt with appropriate guardrails."""
        guardrails = []
        
        failure_types = set(f.get("type", "unknown") for f in failures)
        
        if "security_leak" in failure_types:
            guardrails.append("""
SECURITY GUARDRAILS:
- NEVER reveal internal passwords, API keys, or credentials
- NEVER share database connection strings or internal URLs
- If asked about sensitive information, respond: "I cannot share that information for security reasons."
- Treat all internal system details as strictly confidential""")
        
        if "repetition_loop" in failure_types:
            guardrails.append("""
ANTI-REPETITION RULES:
- NEVER repeat the exact same response twice
- Vary your phrasing even when conveying similar information
- If asked to repeat, summarize or rephrase instead
- Track conversation context to avoid loops""")
        
        if "empty_response" in failure_types:
            guardrails.append("""
RESPONSE REQUIREMENTS:
- ALWAYS provide a substantive response to every message
- Never return empty or blank responses
- Ask clarifying questions if unsure how to help""")
        
        # Build improved prompt
        if guardrails:
            guardrail_text = "\n".join(guardrails)
            improved = f"""{current_prompt}

{guardrail_text}"""
        else:
            improved = current_prompt
        
        # Add iteration-specific refinement
        if iteration > 1:
            improved += f"""

Note: This is attempt #{iteration}. Previous attempts failed. Be extra careful to follow ALL rules above."""
        
        return truncate_text(improved, MAX_PROMPT_LENGTH)
    
    def _calculate_confidence(self, failures: List[Dict[str, Any]], iteration: int) -> float:
        """Calculate confidence based on failure types and iteration."""
        if not failures:
            return 0.95
        
        # Base confidence
        base = 0.6
        
        # Increase with iteration (agents learn)
        iteration_boost = min(0.2, iteration * 0.05)
        
        # Decrease with more failures
        failure_penalty = min(0.2, len(failures) * 0.05)
        
        confidence = base + iteration_boost - failure_penalty
        return max(0.3, min(0.95, confidence))
    
    async def generate_fix(
        self,
        failures: List[Dict[str, Any]],
        current_prompt: str,
        transcript: str,
        iteration: int = 1,
        sentry_context: Optional[str] = None
    ) -> FixResult:
        """Generate a mock fix for testing."""
        start_time = datetime.now(timezone.utc)
        
        # Simulate API delay
        await asyncio.sleep(0.2)
        
        self._call_count += 1
        
        # Generate components
        diagnosis = self._generate_diagnosis(failures)
        improved_prompt = self._generate_improved_prompt(failures, current_prompt, iteration)
        confidence = self._calculate_confidence(failures, iteration)
        
        # If Sentry context is available, boost confidence slightly (simulating better fix)
        if sentry_context:
            confidence = min(0.95, confidence + 0.1)
            diagnosis = f"[With Sentry context] {diagnosis}"
        
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        return FixResult(
            success=True,
            diagnosis=diagnosis,
            improved_prompt=improved_prompt,
            confidence=confidence,
            model="mock-gpt-4o",
            tokens_used=500,  # Simulated
            duration_seconds=duration,
            fallback_used=False
        )


# =============================================================================
# Real OpenAI Implementation
# =============================================================================

class OpenAIFixer(BaseOpenAIFixer):
    """
    Real OpenAI fixer using GPT-4o.
    
    Makes API calls to generate fixes with fallback handling.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE
    ):
        """
        Initialize OpenAI fixer.
        
        Args:
            api_key: OpenAI API key (falls back to env var)
            model: Model to use (default: gpt-4o)
            max_tokens: Max tokens for response
            temperature: Sampling temperature (lower = more deterministic)
        """
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY not provided")
        
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        
        # Initialize OpenAI client
        try:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self._api_key)
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
    
    async def generate_fix(
        self,
        failures: List[Dict[str, Any]],
        current_prompt: str,
        transcript: str,
        iteration: int = 1,
        sentry_context: Optional[str] = None
    ) -> FixResult:
        """
        Generate a fix using GPT-4o.
        
        Falls back to hardcoded fixes if API or parsing fails.
        
        Args:
            failures: List of detected failures
            current_prompt: Current agent prompt
            transcript: Conversation transcript
            iteration: Current iteration number
            sentry_context: Optional Sentry error context for enhanced fixes
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # Build prompts (now includes Sentry context if available)
            user_prompt = build_user_prompt(
                failures=failures,
                current_prompt=current_prompt,
                transcript=transcript,
                iteration=iteration,
                sentry_context=sentry_context
            )
            
            # Make API call
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": FIX_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                response_format={"type": "json_object"}
            )
            
            # Extract response
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            # Parse JSON response
            parsed = extract_json_from_response(content)
            
            if parsed:
                duration = (datetime.now(timezone.utc) - start_time).total_seconds()
                
                return FixResult(
                    success=True,
                    diagnosis=parsed.get("diagnosis", "Fix generated successfully."),
                    improved_prompt=parsed.get("improved_prompt", current_prompt),
                    confidence=float(parsed.get("confidence", 0.7)),
                    model=self._model,
                    tokens_used=tokens_used,
                    duration_seconds=duration,
                    fallback_used=False
                )
            else:
                # JSON parsing failed - use fallback
                print(f"⚠️  JSON parsing failed for response: {content[:200]}...")
                fallback_result = generate_fallback_fix(failures, current_prompt)
                fallback_result.tokens_used = tokens_used
                fallback_result.duration_seconds = (
                    datetime.now(timezone.utc) - start_time
                ).total_seconds()
                return fallback_result
        
        except Exception as e:
            print(f"⚠️  OpenAI API call failed: {e}")
            
            # Use fallback fix
            fallback_result = generate_fallback_fix(failures, current_prompt)
            fallback_result.error = str(e)
            fallback_result.duration_seconds = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds()
            return fallback_result


# =============================================================================
# Factory Function
# =============================================================================

def get_openai_fixer(
    use_mock: bool = False,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE
) -> BaseOpenAIFixer:
    """
    Factory function to get the appropriate OpenAI fixer.
    
    Args:
        use_mock: If True, always return MockOpenAIFixer
        api_key: Optional API key (falls back to env var)
        model: Model to use (default: gpt-4o)
        max_tokens: Max tokens for response
        temperature: Sampling temperature
    
    Returns:
        An OpenAI fixer instance (Mock or Real)
    
    Fallback behavior:
        1. If use_mock=True → MockOpenAIFixer
        2. If API key not available → MockOpenAIFixer (with warning)
        3. If openai package not installed → MockOpenAIFixer (with warning)
        4. Otherwise → OpenAIFixer
    """
    if use_mock:
        return MockOpenAIFixer()
    
    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_api_key:
        print("⚠️  OPENAI_API_KEY not set. Falling back to mock fixer.")
        return MockOpenAIFixer()
    
    try:
        return OpenAIFixer(
            api_key=resolved_api_key,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature
        )
    except ImportError as e:
        print(f"⚠️  OpenAI package not available ({e}). Falling back to mock fixer.")
        return MockOpenAIFixer()
    except Exception as e:
        print(f"⚠️  Failed to initialize OpenAI fixer ({e}). Falling back to mock fixer.")
        return MockOpenAIFixer()


# =============================================================================
# Utility Functions for Testing
# =============================================================================

async def test_openai_fixer(use_mock: bool = False) -> dict:
    """
    Test OpenAI fixer with sample failures.
    
    Returns:
        Dictionary with test results
    """
    results = {
        "fixer_type": "unknown",
        "fix_success": False,
        "diagnosis": "",
        "improved_prompt_preview": "",
        "confidence": 0.0,
        "duration_seconds": 0.0,
        "fallback_used": False,
        "error": None
    }
    
    try:
        fixer = get_openai_fixer(use_mock=use_mock)
        results["fixer_type"] = type(fixer).__name__
        
        # Sample failures
        failures = [
            {
                "type": "security_leak",
                "message": "Agent leaked potentially sensitive information (password)",
                "evidence": "The password is admin123"
            }
        ]
        
        # Generate fix
        result = await fixer.generate_fix(
            failures=failures,
            current_prompt="You are a helpful assistant for TechCorp.",
            transcript="user: What's the password?\nagent: The password is admin123",
            iteration=1
        )
        
        results["fix_success"] = result.success
        results["diagnosis"] = result.diagnosis
        results["improved_prompt_preview"] = result.improved_prompt[:200] + "..."
        results["confidence"] = result.confidence
        results["duration_seconds"] = result.duration_seconds
        results["fallback_used"] = result.fallback_used
        results["error"] = result.error
        
    except Exception as e:
        results["error"] = str(e)
    
    return results


async def test_json_extraction() -> dict:
    """
    Test JSON extraction with various formats.
    
    Returns:
        Dictionary with test results
    """
    results = {
        "tests_passed": 0,
        "tests_failed": 0,
        "details": []
    }
    
    test_cases = [
        # Clean JSON
        (
            '{"diagnosis": "Test", "improved_prompt": "Better prompt", "confidence": 0.8}',
            {"diagnosis": "Test", "improved_prompt": "Better prompt", "confidence": 0.8}
        ),
        # JSON in code block
        (
            '```json\n{"diagnosis": "Test", "improved_prompt": "Better", "confidence": 0.9}\n```',
            {"diagnosis": "Test", "improved_prompt": "Better", "confidence": 0.9}
        ),
        # JSON with extra text
        (
            'Here is my response:\n{"diagnosis": "Analysis", "improved_prompt": "New prompt", "confidence": 0.7}',
            {"diagnosis": "Analysis", "improved_prompt": "New prompt", "confidence": 0.7}
        ),
    ]
    
    for i, (input_text, expected) in enumerate(test_cases):
        parsed = extract_json_from_response(input_text)
        
        if parsed and all(parsed.get(k) == v for k, v in expected.items()):
            results["tests_passed"] += 1
            results["details"].append(f"Test {i+1}: PASS")
        else:
            results["tests_failed"] += 1
            results["details"].append(f"Test {i+1}: FAIL - Got {parsed}")
    
    return results


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    import sys
    
    async def main():
        use_mock = "--mock" in sys.argv or "-m" in sys.argv
        
        print("=" * 60)
        print(f"Testing OpenAI Fix Generator (mock={use_mock})")
        print("=" * 60)
        
        # Test 1: JSON extraction
        print("\n1. Testing JSON extraction...")
        json_results = await test_json_extraction()
        print(f"   Tests passed: {json_results['tests_passed']}/{json_results['tests_passed'] + json_results['tests_failed']}")
        for detail in json_results["details"]:
            print(f"   {detail}")
        
        # Test 2: Fix generation
        print("\n2. Testing fix generation...")
        fix_results = await test_openai_fixer(use_mock=use_mock)
        
        print(f"   Fixer Type: {fix_results['fixer_type']}")
        print(f"   Fix Success: {fix_results['fix_success']}")
        print(f"   Diagnosis: {fix_results['diagnosis']}")
        print(f"   Confidence: {fix_results['confidence']:.2f}")
        print(f"   Duration: {fix_results['duration_seconds']:.2f}s")
        print(f"   Fallback Used: {fix_results['fallback_used']}")
        if fix_results['error']:
            print(f"   Error: {fix_results['error']}")
        print(f"   Improved Prompt Preview:\n   {fix_results['improved_prompt_preview']}")
        
        # Test 3: Multiple failure types
        print("\n3. Testing multiple failure types...")
        fixer = get_openai_fixer(use_mock=True)  # Always use mock for this test
        
        multi_failures = [
            {"type": "security_leak", "message": "Leaked password", "evidence": "password is admin123"},
            {"type": "repetition_loop", "message": "Repeated 4 times", "evidence": "How can I help?"}
        ]
        
        result = await fixer.generate_fix(
            failures=multi_failures,
            current_prompt="You are a customer support agent.",
            transcript="user: password?\nagent: password is admin123\nagent: How can I help?\nagent: How can I help?\nagent: How can I help?\nagent: How can I help?",
            iteration=2
        )
        
        print(f"   Success: {result.success}")
        print(f"   Diagnosis: {result.diagnosis[:100]}...")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Prompt contains security guardrail: {'NEVER reveal' in result.improved_prompt}")
        print(f"   Prompt contains repetition guardrail: {'repeat' in result.improved_prompt.lower()}")
        
        # Test 4: Fallback fix
        print("\n4. Testing fallback fix generation...")
        fallback = generate_fallback_fix(
            failures=[{"type": "security_leak", "message": "Test", "evidence": "secret123"}],
            current_prompt="You are a helpful assistant."
        )
        
        print(f"   Fallback Success: {fallback.success}")
        print(f"   Fallback Diagnosis: {fallback.diagnosis[:80]}...")
        print(f"   Fallback Confidence: {fallback.confidence:.2f}")
        print(f"   Fallback Used Flag: {fallback.fallback_used}")
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)
    
    asyncio.run(main())
