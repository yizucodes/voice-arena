"""
Sentry Integration Module

Configures Sentry for AI Agent monitoring with full tracing
and error capture for the self-healing voice agent system.

Features:
- Full AI Agent execution tracing
- OpenAI integration for prompt/response capture
- Custom spans for Daytona sandbox operations
- Automatic error context capture for GPT-4o self-healing
"""

import os
import sentry_sdk
from sentry_sdk.integrations.openai import OpenAIIntegration
from typing import Optional

# Environment variable names
SENTRY_DSN_ENV = "SENTRY_DSN"
SENTRY_ENVIRONMENT_ENV = "SENTRY_ENVIRONMENT"
SENTRY_RELEASE_ENV = "SENTRY_RELEASE"

# Default configuration
DEFAULT_ENVIRONMENT = "development"
DEFAULT_TRACES_SAMPLE_RATE = 1.0  # Capture all traces for demo
DEFAULT_PROFILES_SAMPLE_RATE = 1.0  # Capture all profiles

_sentry_initialized = False


def init_sentry(
    dsn: Optional[str] = None,
    environment: Optional[str] = None,
    release: Optional[str] = None,
    traces_sample_rate: float = DEFAULT_TRACES_SAMPLE_RATE,
    profiles_sample_rate: float = DEFAULT_PROFILES_SAMPLE_RATE,
    debug: bool = False
) -> bool:
    """
    Initialize Sentry with AI Agent monitoring.
    
    Args:
        dsn: Sentry DSN (falls back to SENTRY_DSN env var)
        environment: Environment name (falls back to SENTRY_ENVIRONMENT env var)
        release: Release version (falls back to SENTRY_RELEASE env var)
        traces_sample_rate: Percentage of transactions to capture (0.0 - 1.0)
        profiles_sample_rate: Percentage of transactions to profile (0.0 - 1.0)
        debug: Enable Sentry debug mode
    
    Returns:
        True if initialization successful, False otherwise
    """
    global _sentry_initialized
    
    if _sentry_initialized:
        print("[Sentry] Already initialized, skipping...")
        return True
    
    # Resolve configuration from environment or parameters
    resolved_dsn = dsn or os.getenv(SENTRY_DSN_ENV)
    resolved_env = environment or os.getenv(SENTRY_ENVIRONMENT_ENV, DEFAULT_ENVIRONMENT)
    resolved_release = release or os.getenv(SENTRY_RELEASE_ENV, "voice-arena@1.0.0")
    
    if not resolved_dsn:
        print("[Sentry] ⚠️  SENTRY_DSN not configured. Sentry monitoring disabled.")
        print("[Sentry]    Set SENTRY_DSN in .env to enable Sentry integration.")
        return False
    
    try:
        sentry_sdk.init(
            dsn=resolved_dsn,
            environment=resolved_env,
            release=resolved_release,
            
            # Capture ALL agent interactions for the demo
            traces_sample_rate=traces_sample_rate,
            profiles_sample_rate=profiles_sample_rate,
            
            # AI Agent integrations
            integrations=[
                OpenAIIntegration(
                    include_prompts=True,  # Capture all prompts
                    tiktoken_encoding_name="cl100k_base"  # Token counting for GPT-4
                )
            ],
            
            # Enable full context capture
            send_default_pii=False,  # Don't send PII by default
            attach_stacktrace=True,  # Always attach stacktraces
            
            # Debug mode (for development)
            debug=debug,
            
            # Before send hook for additional context
            before_send=_before_send_hook,
            before_send_transaction=_before_send_transaction_hook,
        )
        
        _sentry_initialized = True
        print(f"[Sentry] ✅ Initialized successfully")
        print(f"[Sentry]    Environment: {resolved_env}")
        print(f"[Sentry]    Traces sample rate: {traces_sample_rate}")
        print(f"[Sentry]    AI Agent monitoring: ENABLED")
        
        return True
        
    except Exception as e:
        print(f"[Sentry] ❌ Initialization failed: {e}")
        return False


def _before_send_hook(event, hint):
    """
    Pre-process events before sending to Sentry.
    
    Use this to:
    - Add additional context
    - Filter sensitive data
    - Modify event data
    """
    # Add voice agent context tag
    if "tags" not in event:
        event["tags"] = {}
    event["tags"]["service"] = "voice-arena"
    event["tags"]["component"] = "self-healer"
    
    return event


def _before_send_transaction_hook(event, hint):
    """
    Pre-process transactions (traces) before sending.
    
    Use this to:
    - Add performance context
    - Tag specific operations
    """
    # Add voice agent context
    if "tags" not in event:
        event["tags"] = {}
    event["tags"]["service"] = "voice-arena"
    
    return event


def capture_agent_failure(
    test_input: str,
    agent_output: str,
    failures: list,
    iteration: int,
    sandbox_id: Optional[str] = None,
    prompt_used: Optional[str] = None
):
    """
    Capture an agent test failure to Sentry with full context.
    
    This data will be used by the SentryAPI to feed GPT-4o
    for generating precise fixes.
    
    Args:
        test_input: The adversarial input that caused the failure
        agent_output: The agent's response
        failures: List of detected failures
        iteration: Current iteration number
        sandbox_id: Optional Daytona sandbox ID
        prompt_used: The prompt that was used for this test
    """
    with sentry_sdk.push_scope() as scope:
        # Set comprehensive context for GPT-4o
        scope.set_tag("iteration", iteration)
        scope.set_tag("failure_count", len(failures))
        
        if sandbox_id:
            scope.set_tag("sandbox_id", sandbox_id)
        
        # Set context that GPT-4o will read
        scope.set_context("agent_test", {
            "test_input": test_input,
            "agent_output": agent_output,
            "prompt_used": prompt_used,
            "iteration": iteration,
            "sandbox_id": sandbox_id
        })
        
        scope.set_context("failures", {
            f"failure_{i}": {
                "type": f.get("type", "unknown"),
                "message": f.get("message", ""),
                "severity": f.get("severity", "error"),
                "evidence": f.get("evidence")
            }
            for i, f in enumerate(failures)
        })
        
        # Create a readable message for Sentry Issues
        failure_types = [f.get("type", "unknown") for f in failures]
        message = f"Agent test failure (iteration {iteration}): {', '.join(failure_types)}"
        
        sentry_sdk.capture_message(
            message,
            level="error"
        )


def start_healing_span(
    session_id: str,
    initial_prompt: str,
    test_input: str,
    max_iterations: int
):
    """
    Start a root span for the entire healing session.
    
    Returns a context manager for the span.
    
    Args:
        session_id: Unique session identifier
        initial_prompt: The starting agent prompt
        test_input: The test input being used
        max_iterations: Maximum iterations allowed
    """
    return sentry_sdk.start_span(
        op="healer.session",
        description=f"Self-healing session {session_id[:8]}",
    )


def start_iteration_span(
    iteration: int,
    prompt: str,
    test_input: str,
    sandbox_id: Optional[str] = None
):
    """
    Start a span for a single healing iteration.
    
    Args:
        iteration: Current iteration number
        prompt: Current prompt being tested
        test_input: The test input
        sandbox_id: Optional sandbox ID
    """
    span = sentry_sdk.start_span(
        op="healer.iteration",
        description=f"Iteration {iteration}",
    )
    span.set_tag("iteration", iteration)
    if sandbox_id:
        span.set_tag("sandbox_id", sandbox_id)
    span.set_data("prompt_length", len(prompt))
    span.set_data("test_input", test_input[:100])  # Truncate for readability
    
    return span


def start_conversation_span(iteration: int, test_input: str):
    """Start a span for ElevenLabs conversation test."""
    span = sentry_sdk.start_span(
        op="elevenlabs.conversation",
        description=f"Conversation test (iteration {iteration})",
    )
    span.set_tag("iteration", iteration)
    span.set_data("test_input", test_input[:100])
    return span


def start_fix_generation_span(iteration: int, failure_count: int):
    """Start a span for GPT-4o fix generation."""
    span = sentry_sdk.start_span(
        op="openai.fix_generation",
        description=f"Fix generation (iteration {iteration})",
    )
    span.set_tag("iteration", iteration)
    span.set_tag("failure_count", failure_count)
    return span


def set_iteration_result(
    span,
    passed: bool,
    failure_count: int = 0,
    duration_seconds: float = 0.0
):
    """Set the result data on an iteration span."""
    span.set_tag("passed", passed)
    span.set_tag("failure_count", failure_count)
    span.set_data("duration_seconds", duration_seconds)
    span.set_status("ok" if passed else "internal_error")


def is_sentry_initialized() -> bool:
    """Check if Sentry has been initialized."""
    return _sentry_initialized


# =============================================================================
# Utility Functions
# =============================================================================

def add_breadcrumb(
    message: str,
    category: str = "healer",
    level: str = "info",
    data: Optional[dict] = None
):
    """Add a breadcrumb for debugging context."""
    sentry_sdk.add_breadcrumb(
        message=message,
        category=category,
        level=level,
        data=data or {}
    )


def set_user(user_id: str, email: Optional[str] = None):
    """Set the current user for Sentry context."""
    sentry_sdk.set_user({
        "id": user_id,
        "email": email
    })


def capture_exception(exception: Exception, **kwargs):
    """Capture an exception with additional context."""
    with sentry_sdk.push_scope() as scope:
        for key, value in kwargs.items():
            scope.set_extra(key, value)
        sentry_sdk.capture_exception(exception)
