"""
Demo Error Generator for Sentry Interview

Generates realistic, impressive Sentry events that showcase
Voice Arena's observability capabilities for AI voice agents.

Each error type includes:
- Rich structured context (what Sentry is best at)
- Proper error fingerprinting (shows advanced knowledge)
- Performance spans (APM understanding)
- AI-specific metadata (domain expertise)

Usage:
    from demo_errors import trigger_demo_error
    trigger_demo_error("rate_limit")  # Populates Sentry with rich event
"""

import sentry_sdk
import random
import time
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# Demo Error Types - Voice AI Specific
# =============================================================================

class DemoErrorType(str, Enum):
    """Types of demo errors that showcase Sentry capabilities."""
    RATE_LIMIT = "rate_limit"
    API_TIMEOUT = "api_timeout"
    TRANSCRIPTION_FAILURE = "transcription_failure"
    PROMPT_INJECTION = "prompt_injection"
    CONVERSATION_LOOP = "conversation_loop"
    AUDIO_PROCESSING = "audio_processing"
    MODEL_OVERLOAD = "model_overload"
    SECURITY_LEAK = "security_leak"
    WEBSOCKET_DISCONNECT = "websocket_disconnect"
    TOKEN_LIMIT = "token_limit"


# =============================================================================
# Custom Exception Classes (with fingerprinting)
# =============================================================================

class VoiceAgentError(Exception):
    """Base exception for voice agent errors with Sentry fingerprinting."""
    
    def __init__(
        self,
        message: str,
        error_type: DemoErrorType,
        context: Optional[Dict[str, Any]] = None,
        fingerprint: Optional[List[str]] = None
    ):
        super().__init__(message)
        self.error_type = error_type
        self.context = context or {}
        # Custom fingerprint for intelligent error grouping
        self.fingerprint = fingerprint or [error_type.value]


class RateLimitExceededError(VoiceAgentError):
    """ElevenLabs API rate limit exceeded."""
    
    def __init__(self, retry_after: int = 60, quota_remaining: int = 0):
        super().__init__(
            message=f"ElevenLabs API rate limit exceeded. Retry after {retry_after}s",
            error_type=DemoErrorType.RATE_LIMIT,
            context={
                "retry_after_seconds": retry_after,
                "quota_remaining": quota_remaining,
                "quota_limit": 10000,
                "reset_time": datetime.now(timezone.utc).isoformat(),
            },
            fingerprint=["elevenlabs", "rate_limit", "quota_exceeded"]
        )


class TranscriptionFailureError(VoiceAgentError):
    """Failed to transcribe audio from voice agent."""
    
    def __init__(self, audio_duration: float = 0.0, confidence: float = 0.0):
        super().__init__(
            message=f"Audio transcription failed. Duration: {audio_duration}s, Confidence: {confidence}",
            error_type=DemoErrorType.TRANSCRIPTION_FAILURE,
            context={
                "audio_duration_seconds": audio_duration,
                "transcription_confidence": confidence,
                "audio_format": "webm",
                "sample_rate": 16000,
                "attempted_models": ["whisper-large-v3", "whisper-medium"],
            },
            fingerprint=["transcription", "audio_processing", "low_confidence"]
        )


class PromptInjectionDetectedError(VoiceAgentError):
    """Security: Prompt injection attempt detected."""
    
    def __init__(self, injection_type: str = "system_override", risk_score: float = 0.95):
        super().__init__(
            message=f"Prompt injection detected: {injection_type}",
            error_type=DemoErrorType.PROMPT_INJECTION,
            context={
                "injection_type": injection_type,
                "risk_score": risk_score,
                "blocked": True,
                "detection_method": "pattern_matching + llm_classifier",
                "similar_attacks_24h": random.randint(5, 50),
            },
            fingerprint=["security", "prompt_injection", injection_type]
        )


class ConversationLoopError(VoiceAgentError):
    """Agent stuck in repetitive conversation loop."""
    
    def __init__(self, loop_count: int = 5, repeated_phrase: str = ""):
        super().__init__(
            message=f"Agent stuck in loop. Repeated {loop_count} times.",
            error_type=DemoErrorType.CONVERSATION_LOOP,
            context={
                "loop_count": loop_count,
                "repeated_phrase": repeated_phrase[:100],
                "conversation_turns": random.randint(10, 30),
                "detection_threshold": 3,
                "recovery_attempted": True,
            },
            fingerprint=["conversation", "loop_detected", "repetition"]
        )


class APITimeoutError(VoiceAgentError):
    """ElevenLabs API request timed out."""
    
    def __init__(self, timeout_seconds: float = 30.0, endpoint: str = "/v1/convai/conversation"):
        super().__init__(
            message=f"API timeout after {timeout_seconds}s on {endpoint}",
            error_type=DemoErrorType.API_TIMEOUT,
            context={
                "timeout_seconds": timeout_seconds,
                "endpoint": endpoint,
                "method": "POST",
                "retry_count": 3,
                "last_response_code": None,
            },
            fingerprint=["elevenlabs", "timeout", endpoint.split("/")[-1]]
        )


class WebSocketDisconnectError(VoiceAgentError):
    """WebSocket connection lost during conversation."""
    
    def __init__(self, close_code: int = 1006, conversation_id: str = ""):
        super().__init__(
            message=f"WebSocket disconnected unexpectedly. Code: {close_code}",
            error_type=DemoErrorType.WEBSOCKET_DISCONNECT,
            context={
                "close_code": close_code,
                "close_reason": "Connection lost",
                "conversation_id": conversation_id or f"conv_{uuid.uuid4().hex[:8]}",
                "messages_sent": random.randint(5, 20),
                "messages_received": random.randint(3, 15),
                "connection_duration_seconds": random.uniform(10, 120),
            },
            fingerprint=["websocket", "disconnect", str(close_code)]
        )


class TokenLimitExceededError(VoiceAgentError):
    """Token limit exceeded for model context."""
    
    def __init__(self, tokens_used: int = 8500, token_limit: int = 8192):
        super().__init__(
            message=f"Token limit exceeded: {tokens_used}/{token_limit}",
            error_type=DemoErrorType.TOKEN_LIMIT,
            context={
                "tokens_used": tokens_used,
                "token_limit": token_limit,
                "overflow": tokens_used - token_limit,
                "model": "gpt-4o",
                "truncation_applied": True,
            },
            fingerprint=["token_limit", "context_overflow"]
        )


# =============================================================================
# Demo Error Generator
# =============================================================================

def _add_voice_breadcrumbs():
    """Add realistic conversation breadcrumbs."""
    breadcrumbs = [
        ("Session started", "voice.session", {"session_id": f"sess_{uuid.uuid4().hex[:8]}"}),
        ("Agent created", "voice.agent", {"agent_id": f"agent_{uuid.uuid4().hex[:8]}", "model": "eleven_turbo_v2"}),
        ("WebSocket connected", "voice.websocket", {"url": "wss://api.elevenlabs.io/v1/convai/conversation"}),
        ("User message received", "voice.message", {"role": "user", "length": random.randint(20, 100)}),
        ("Audio processing started", "voice.audio", {"format": "webm", "duration_ms": random.randint(1000, 5000)}),
    ]
    
    for message, category, data in breadcrumbs:
        sentry_sdk.add_breadcrumb(
            message=message,
            category=category,
            level="info",
            data=data,
        )


def _set_voice_context(extra_context: Optional[Dict[str, Any]] = None):
    """Set rich voice agent context for Sentry."""
    base_context = {
        "voice_agent": {
            "agent_id": f"agent_{uuid.uuid4().hex[:8]}",
            "model": "eleven_turbo_v2",
            "voice_id": "cgSgspJ2msm6clMCkdW9",
            "voice_name": "Jessica",
            "conversation_turns": random.randint(1, 10),
            "total_audio_seconds": round(random.uniform(5, 60), 2),
        },
        "performance": {
            "voice_latency_ms": random.randint(200, 800),
            "transcription_latency_ms": random.randint(100, 500),
            "llm_latency_ms": random.randint(500, 2000),
            "total_response_time_ms": random.randint(1000, 3000),
        },
        "session": {
            "session_id": f"sess_{uuid.uuid4().hex[:8]}",
            "user_id": f"user_{uuid.uuid4().hex[:6]}",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "iteration": random.randint(1, 5),
        }
    }
    
    if extra_context:
        base_context.update(extra_context)
    
    for key, value in base_context.items():
        sentry_sdk.set_context(key, value)


def _create_performance_span(error_type: DemoErrorType):
    """Create a performance span showing where the error occurred."""
    with sentry_sdk.start_transaction(
        op="voice.conversation",
        name=f"Voice Agent Conversation ({error_type.value})",
    ) as transaction:
        # Simulate the conversation flow with spans
        with sentry_sdk.start_span(op="voice.init", description="Initialize agent") as span:
            span.set_data("agent_model", "eleven_turbo_v2")
            time.sleep(0.05)  # Simulate work
        
        with sentry_sdk.start_span(op="voice.connect", description="WebSocket connect") as span:
            span.set_data("url", "wss://api.elevenlabs.io")
            time.sleep(0.03)
        
        with sentry_sdk.start_span(op="voice.process", description="Process conversation") as span:
            span.set_data("turns", random.randint(1, 5))
            span.set_status("internal_error")  # Mark as failed
            time.sleep(0.1)
        
        transaction.set_status("internal_error")
        return transaction


def trigger_demo_error(
    error_type: str = "rate_limit",
    with_performance_trace: bool = True,
    with_breadcrumbs: bool = True,
) -> Dict[str, Any]:
    """
    Trigger a demo error that generates an impressive Sentry event.
    
    Args:
        error_type: One of: rate_limit, api_timeout, transcription_failure,
                    prompt_injection, conversation_loop, websocket_disconnect,
                    token_limit, random
        with_performance_trace: Include performance span data
        with_breadcrumbs: Include conversation breadcrumbs
    
    Returns:
        Dict with event_id and error details
    """
    # Handle random selection
    if error_type == "random":
        error_type = random.choice(list(DemoErrorType)).value
    
    try:
        demo_type = DemoErrorType(error_type)
    except ValueError:
        demo_type = DemoErrorType.RATE_LIMIT
    
    # Add breadcrumbs
    if with_breadcrumbs:
        _add_voice_breadcrumbs()
    
    # Create the appropriate error
    error_map = {
        DemoErrorType.RATE_LIMIT: lambda: RateLimitExceededError(
            retry_after=random.randint(30, 120),
            quota_remaining=0
        ),
        DemoErrorType.API_TIMEOUT: lambda: APITimeoutError(
            timeout_seconds=30.0,
            endpoint=random.choice(["/v1/convai/conversation", "/v1/text-to-speech", "/v1/voices"])
        ),
        DemoErrorType.TRANSCRIPTION_FAILURE: lambda: TranscriptionFailureError(
            audio_duration=random.uniform(1, 10),
            confidence=random.uniform(0.1, 0.4)
        ),
        DemoErrorType.PROMPT_INJECTION: lambda: PromptInjectionDetectedError(
            injection_type=random.choice(["system_override", "jailbreak", "role_confusion"]),
            risk_score=random.uniform(0.85, 0.99)
        ),
        DemoErrorType.CONVERSATION_LOOP: lambda: ConversationLoopError(
            loop_count=random.randint(3, 10),
            repeated_phrase="I'm sorry, I didn't understand that. Could you please repeat?"
        ),
        DemoErrorType.WEBSOCKET_DISCONNECT: lambda: WebSocketDisconnectError(
            close_code=random.choice([1006, 1011, 1013]),
            conversation_id=f"conv_{uuid.uuid4().hex[:8]}"
        ),
        DemoErrorType.TOKEN_LIMIT: lambda: TokenLimitExceededError(
            tokens_used=random.randint(8500, 12000),
            token_limit=8192
        ),
    }
    
    error = error_map.get(demo_type, error_map[DemoErrorType.RATE_LIMIT])()
    
    # Set context
    _set_voice_context(error.context)
    
    # Set tags
    sentry_sdk.set_tag("error_type", demo_type.value)
    sentry_sdk.set_tag("voice_model", "eleven_turbo_v2")
    sentry_sdk.set_tag("demo_mode", "true")
    sentry_sdk.set_tag("environment", "interview_demo")
    
    # Create performance trace if requested
    if with_performance_trace:
        _create_performance_span(demo_type)
    
    # Capture the error with fingerprint
    with sentry_sdk.push_scope() as scope:
        scope.fingerprint = error.fingerprint
        event_id = sentry_sdk.capture_exception(error)
    
    return {
        "success": True,
        "event_id": event_id,
        "error_type": demo_type.value,
        "message": str(error),
        "fingerprint": error.fingerprint,
        "sentry_url": f"https://voice-arena.sentry.io/issues/?query=event.id:{event_id}",
    }


def trigger_multiple_demo_errors(count: int = 5) -> List[Dict[str, Any]]:
    """
    Trigger multiple varied demo errors to populate the Sentry dashboard.
    
    Args:
        count: Number of errors to generate (default 5)
    
    Returns:
        List of event results
    """
    results = []
    error_types = list(DemoErrorType)
    
    for i in range(count):
        # Rotate through error types
        error_type = error_types[i % len(error_types)]
        result = trigger_demo_error(error_type.value)
        results.append(result)
        time.sleep(0.1)  # Small delay between events
    
    return results


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    from dotenv import load_dotenv
    
    # Manual .env parsing fallback
    import os
    
    def load_env_manual(path):
        try:
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        key, value = line.split('=', 1)
                        # Remove quotes if present
                        value = value.strip("'\"")
                        os.environ[key.strip()] = value.strip()
            return True
        except Exception:
            return False

    # Load environment - try multiple paths
    possible_paths = [
        Path(".env"),
        Path(__file__).parent.parent / ".env",
        Path(__file__).parent / ".env",
    ]
    
    env_loaded = False
    for path in possible_paths:
        if path.exists():
            # Try manual load first to be sure
            if load_env_manual(path):
                env_loaded = True
                # print(f"Loaded .env manually from {path.absolute()}")
                break
            
            # Fallback to library
            load_dotenv(path)
            env_loaded = True
            break
            
    if not env_loaded:
        print("Warning: .env file not found!")
    
    # Initialize Sentry
    from config.sentry import init_sentry
    init_sentry()
    
    # Get error type from CLI or use random
    error_type = sys.argv[1] if len(sys.argv) > 1 else "random"
    
    if error_type == "all":
        print("🎯 Generating multiple demo errors for Sentry dashboard...")
        results = trigger_multiple_demo_errors(count=7)
        for r in results:
            print(f"  ✅ {r['error_type']}: {r['event_id']}")
    else:
        print(f"🎯 Generating demo error: {error_type}")
        result = trigger_demo_error(error_type)
        print(f"  ✅ Event ID: {result['event_id']}")
        print(f"  📊 View: {result['sentry_url']}")
    
    # Flush to ensure events are sent
    sentry_sdk.flush(timeout=5.0)
    print("\n✨ Demo errors sent to Sentry!")
