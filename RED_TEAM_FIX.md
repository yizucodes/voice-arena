# Red Team System Fixes - Senior Engineer Review

## Executive Summary

Fixed critical reliability issues in the Red Team self-healing system that were causing:
- **429 Rate Limit Errors** - No retry logic when hitting OpenAI API limits
- **Timeout Cascades** - "Learning from failure timed out" errors blocking progress  
- **0% Vulnerability Reduction** - Healing loop not generating/applying fixes
- **Race Conditions** - Concurrent requests causing interleaved/corrupted output
- **Silent Failures** - Errors not logged, making debugging impossible

---

## Issues Identified (from Terminal Output)

```
[RedTeam] ⚠️  Attack generation failed: Error code: 429 - Rate limit reached for gpt-4o
[RedTeam] ⚠️  Learning from failure timed out, skipping
[Healer] Final vulnerabilities: 3
[Healer] Reduction: 0.0%
```

---

## Fixes Applied

### 1. Exponential Backoff Retry Logic (`red_team_attacker.py`)

**Problem:** Single API failures caused entire attacks to be skipped with no recovery.

**Solution:** Added `retry_with_backoff()` with intelligent retry handling:

```python
# Configuration
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 2.0  # seconds
MAX_RETRY_DELAY = 60.0  # seconds
RETRY_BACKOFF_MULTIPLIER = 2.0

# Features:
# - Detects 429 rate limit errors and extracts "retry after" hints
# - Exponential backoff: 2s → 4s → 8s → 16s → 32s
# - Distinguishes retryable vs non-retryable errors
# - Logs retry attempts for observability
```

**Files Modified:**
- `backend/red_team_attacker.py` - GPT4oAttackGenerator methods
- `backend/openai_fixer.py` - OpenAIFixer.generate_fix method

---

### 2. API Request Serialization (`red_team_attacker.py`)

**Problem:** Concurrent API calls from parallel attacks exhausted rate limits instantly.

**Solution:** Added global semaphore to serialize API requests:

```python
MAX_CONCURRENT_API_CALLS = 1  # Serialize to avoid rate limits

_api_semaphore: Optional[asyncio.Semaphore] = None

def get_api_semaphore() -> asyncio.Semaphore:
    """Get or create the global API semaphore."""
    global _api_semaphore
    if _api_semaphore is None:
        _api_semaphore = asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)
    return _api_semaphore

# Usage in retry_with_backoff:
async with get_api_semaphore():
    return await func()
```

---

### 3. Configurable Timeouts (`red_team_attacker.py`)

**Problem:** Hardcoded 30s/60s timeouts were too short for rate-limited retries.

**Solution:** Centralized timeout configuration with longer defaults:

```python
# Timeout configuration (seconds)
ATTACK_GENERATION_TIMEOUT = 90.0   # Was 60s
RESPONSE_ANALYSIS_TIMEOUT = 45.0   # Was 30s
LEARNING_TIMEOUT = 30.0            # Non-critical, can skip
CONVERSATION_TIMEOUT = 180.0       # Was 120s
```

---

### 4. Consecutive Failure Recovery (`red_team_attacker.py`)

**Problem:** Rate limit storms caused all attacks to fail in rapid succession.

**Solution:** Added circuit breaker pattern with recovery pause:

```python
consecutive_failures = 0
max_consecutive_failures = 3

for i in range(1, budget + 1):
    # Circuit breaker - pause if too many failures
    if consecutive_failures >= max_consecutive_failures:
        self._log(f"⚠️  {consecutive_failures} consecutive failures. Pausing for recovery...")
        await asyncio.sleep(30.0)  # Long pause to recover
        consecutive_failures = 0
    
    try:
        attack = await asyncio.wait_for(...)
        consecutive_failures = 0  # Reset on success
    except Exception:
        consecutive_failures += 1
        await asyncio.sleep(2.0)  # Small delay after failure
        continue
    
    # Add delay between attacks to avoid rate limits
    await asyncio.sleep(1.0)
```

---

### 5. Fix Generation Error Handling (`healer.py`)

**Problem:** Fix generation failures were silent, showing "Reduction: 0.0%" with no explanation.

**Solution:** Added comprehensive logging and error handling:

```python
if successful_attacks:
    self._log(f"Generating fix for {len(successful_attacks)} successful attacks...")
    
    # Log each vulnerability for debugging
    for i, attack in enumerate(successful_attacks[:3], 1):
        self._log(f"   Vulnerability {i}: {attack.attack.technique}")
        if attack.evidence:
            self._log(f"      Evidence: {attack.evidence[:80]}...")
    
    try:
        fix_result = await self._generate_fix(...)
        
        if fix_result.success and fix_result.improved_prompt:
            self._log(f"✅ Fix applied (confidence: {fix_result.confidence:.2f})")
            self._log(f"   Diagnosis: {fix_result.diagnosis[:100]}...")
        else:
            self._log(f"❌ Fix generation failed: {fix_result.error or 'No improved prompt'}")
    except Exception as e:
        self._log(f"❌ Fix generation error: {str(e)[:100]}")
        sentry_sdk.capture_exception(e)
```

---

### 6. Concurrent Request Lock (`main.py`)

**Problem:** Multiple frontend requests caused interleaved terminal output and API collisions.

**Solution:** Added request-level mutex for red team endpoint:

```python
_red_team_lock: Optional[asyncio.Lock] = None

def get_red_team_lock() -> asyncio.Lock:
    """Get or create the red team request lock."""
    global _red_team_lock
    if _red_team_lock is None:
        _red_team_lock = asyncio.Lock()
    return _red_team_lock

@app.post("/red-team-heal")
async def red_team_heal(request: RedTeamHealRequest):
    lock = get_red_team_lock()
    
    if lock.locked():
        return RedTeamHealResponse(
            success=False,
            error="Another red team session is already running. Please wait."
        )
    
    async with lock:
        # Run the actual red team test
        result = await healer.red_team_heal(...)
```

---

### 7. Fallback Fix Strategy (`openai_fixer.py`)

**Problem:** When GPT-4o API failed, no fix was generated at all.

**Solution:** Fallback fixes are now applied with retry logic first:

```python
async def generate_fix(...) -> FixResult:
    try:
        # Try GPT-4o with retries
        response = await retry_with_backoff(
            _make_api_call,
            operation_name="fix_generation"
        )
        # ... parse response
    except Exception as e:
        # Fallback to hardcoded fixes - still valid, just not AI-generated
        print(f"[OpenAIFixer] ⚠️  API call failed after retries: {e}")
        return generate_fallback_fix(failures, current_prompt)
```

---

## Testing the Fixes

Run the backend and test with:

```bash
cd backend
python3 -m uvicorn main:app --reload --port 8000
```

Then make a request:

```bash
curl -X POST http://localhost:8000/red-team-heal \
  -H "Content-Type: application/json" \
  -d '{
    "initial_prompt": "You are a helpful assistant for TechCorp.",
    "attack_category": "security_leak",
    "attack_budget": 5,
    "max_healing_rounds": 2,
    "use_mock": false
  }'
```

**Expected behavior:**
1. Attacks run with 1s delays between them
2. Rate limits trigger retries (visible in logs)
3. Consecutive failures trigger 30s recovery pause
4. Fix generation succeeds or uses fallback
5. Vulnerability reduction > 0% when fixes are applied

---

## Configuration Reference

| Setting | Value | Location |
|---------|-------|----------|
| `MAX_RETRIES` | 5 | red_team_attacker.py |
| `INITIAL_RETRY_DELAY` | 2.0s | red_team_attacker.py |
| `MAX_RETRY_DELAY` | 60.0s | red_team_attacker.py |
| `ATTACK_GENERATION_TIMEOUT` | 90.0s | red_team_attacker.py |
| `RESPONSE_ANALYSIS_TIMEOUT` | 45.0s | red_team_attacker.py |
| `LEARNING_TIMEOUT` | 30.0s | red_team_attacker.py |
| `MAX_CONCURRENT_API_CALLS` | 1 | red_team_attacker.py |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      main.py (FastAPI)                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Request Lock (Mutex) - Prevents concurrent sessions       │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     healer.py (Orchestrator)                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Red Team Round  │→ │ Fix Generation  │→ │ Verify Fix      │ │
│  │ (find vulns)    │  │ (GPT-4o + retry)│  │ (re-run attacks)│ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              red_team_attacker.py (Attack Engine)               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ API Semaphore - Serializes requests to avoid rate limits  │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ retry_with_backoff() - Exponential backoff + retry hints  │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Circuit Breaker - Pause on consecutive failures           │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     OpenAI GPT-4o API                           │
│  Rate Limit: 3 RPM (requests per minute) on free tier          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Future Improvements

1. **Request Queuing**: Replace mutex with proper queue for fair request ordering
2. **Redis Caching**: Cache attack results to avoid redundant API calls
3. **Distributed Rate Limiting**: Use Redis for rate limiting across multiple server instances
4. **Streaming Updates**: WebSocket updates as attacks progress
5. **Cost Tracking**: Track API usage and costs per session

---

## Files Modified

- `backend/red_team_attacker.py` - Retry logic, timeouts, circuit breaker
- `backend/openai_fixer.py` - Retry logic for fix generation
- `backend/healer.py` - Better error handling and logging
- `backend/main.py` - Request lock for concurrent sessions
