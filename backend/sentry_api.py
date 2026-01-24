"""
Sentry API Client Module

Provides API client for fetching error context from Sentry
to feed into GPT-4o for generating precise fixes.

This module enables the autonomous self-healing loop:
1. Agent test fails → Sentry captures full context
2. SentryAPI fetches error details
3. GPT-4o reads context → generates fix
4. Next iteration tests with improved prompt
"""

import os
import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SentryIssue:
    """Represents a Sentry issue with context for GPT-4o."""
    id: str
    title: str
    message: str
    level: str = "error"
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    count: int = 1
    
    # Agent-specific context
    test_input: Optional[str] = None
    agent_output: Optional[str] = None
    prompt_used: Optional[str] = None
    iteration: Optional[int] = None
    sandbox_id: Optional[str] = None
    
    # Technical details
    stacktrace: Optional[str] = None
    breadcrumbs: List[Dict[str, Any]] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    failures: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_gpt_context(self) -> str:
        """
        Format the issue as context for GPT-4o to generate a fix.
        
        Returns:
            Formatted string with all relevant context
        """
        parts = [
            f"Error Title: {self.title}",
            f"Error Message: {self.message}",
        ]
        
        if self.iteration is not None:
            parts.append(f"Iteration: {self.iteration}")
        
        if self.test_input:
            parts.append(f"Test Input: {self.test_input}")
        
        if self.agent_output:
            parts.append(f"Agent Output: {self.agent_output}")
        
        if self.prompt_used:
            parts.append(f"Prompt Used:\n{self.prompt_used}")
        
        if self.failures:
            parts.append("Detected Failures:")
            for i, f in enumerate(self.failures, 1):
                parts.append(f"  {i}. [{f.get('type', 'unknown')}] {f.get('message', '')}")
        
        if self.stacktrace:
            parts.append(f"Stacktrace:\n{self.stacktrace}")
        
        return "\n".join(parts)


@dataclass
class SentryAPIResult:
    """Result from a Sentry API call."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    status_code: Optional[int] = None


# =============================================================================
# Sentry API Client
# =============================================================================

class SentryAPI:
    """
    Client for fetching error context from Sentry.
    
    Used by the self-healing loop to provide GPT-4o with
    comprehensive error context for generating fixes.
    """
    
    def __init__(
        self,
        auth_token: Optional[str] = None,
        org_slug: Optional[str] = None,
        project_slug: Optional[str] = None,
        base_url: str = "https://sentry.io/api/0"
    ):
        """
        Initialize the Sentry API client.
        
        Args:
            auth_token: Sentry auth token (falls back to SENTRY_AUTH_TOKEN env)
            org_slug: Organization slug (falls back to SENTRY_ORG env)
            project_slug: Project slug (falls back to SENTRY_PROJECT env)
            base_url: Sentry API base URL
        """
        self.auth_token = auth_token or os.getenv("SENTRY_AUTH_TOKEN")
        self.org_slug = org_slug or os.getenv("SENTRY_ORG")
        self.project_slug = project_slug or os.getenv("SENTRY_PROJECT")
        self.base_url = base_url
        
        # Validate configuration
        self._configured = bool(
            self.auth_token and self.org_slug and self.project_slug
        )
        
        if not self._configured:
            missing = []
            if not self.auth_token:
                missing.append("SENTRY_AUTH_TOKEN")
            if not self.org_slug:
                missing.append("SENTRY_ORG")
            if not self.project_slug:
                missing.append("SENTRY_PROJECT")
            print(f"[SentryAPI] ⚠️  Missing: {', '.join(missing)}")
            print("[SentryAPI]    API features disabled. Set env vars to enable.")
    
    @property
    def is_configured(self) -> bool:
        """Check if the API client is properly configured."""
        return self._configured
    
    def _get_headers(self) -> Dict[str, str]:
        """Get authorization headers."""
        return {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json"
        }
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None
    ) -> SentryAPIResult:
        """Make an HTTP request to the Sentry API."""
        if not self._configured:
            return SentryAPIResult(
                success=False,
                error="Sentry API not configured"
            )
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=self._get_headers(),
                params=params,
                json=json_data,
                timeout=30
            )
            
            if response.status_code >= 400:
                return SentryAPIResult(
                    success=False,
                    error=f"API error: {response.status_code} - {response.text}",
                    status_code=response.status_code
                )
            
            return SentryAPIResult(
                success=True,
                data=response.json() if response.text else None,
                status_code=response.status_code
            )
            
        except requests.RequestException as e:
            return SentryAPIResult(
                success=False,
                error=f"Request failed: {e}"
            )
    
    async def get_latest_issue(
        self,
        query: Optional[str] = None,
        limit: int = 1
    ) -> Optional[SentryIssue]:
        """
        Fetch the latest issue from Sentry.
        
        Args:
            query: Optional search query (e.g., "is:unresolved")
            limit: Number of issues to fetch (default 1)
        
        Returns:
            SentryIssue if found, None otherwise
        """
        if not self._configured:
            print("[SentryAPI] Not configured, returning None")
            return None
        
        endpoint = f"projects/{self.org_slug}/{self.project_slug}/issues/"
        params = {
            "limit": limit,
            "query": query or "is:unresolved",
            "sort": "date"
        }
        
        result = await asyncio.to_thread(
            self._make_request, "GET", endpoint, params=params
        )
        
        if not result.success or not result.data:
            return None
        
        issues = result.data
        if not issues:
            return None
        
        issue_data = issues[0]
        
        # Fetch full issue details including events
        issue_details = await self.get_issue_details(issue_data["id"])
        
        return issue_details
    
    async def get_issue_details(self, issue_id: str) -> Optional[SentryIssue]:
        """
        Get full details for a specific issue, including context.
        
        Args:
            issue_id: The Sentry issue ID
        
        Returns:
            SentryIssue with full context for GPT-4o
        """
        if not self._configured:
            return None
        
        # Get issue metadata
        issue_endpoint = f"issues/{issue_id}/"
        issue_result = await asyncio.to_thread(
            self._make_request, "GET", issue_endpoint
        )
        
        if not issue_result.success or not issue_result.data:
            return None
        
        issue_data = issue_result.data
        
        # Get latest event for detailed context
        events_endpoint = f"issues/{issue_id}/events/latest/"
        events_result = await asyncio.to_thread(
            self._make_request, "GET", events_endpoint
        )
        
        event_data = events_result.data if events_result.success else {}
        
        # Extract agent-specific context
        contexts = event_data.get("contexts", {})
        agent_test_context = contexts.get("agent_test", {})
        failures_context = contexts.get("failures", {})
        
        # Parse failures from context
        failures = []
        for key, value in failures_context.items():
            if key.startswith("failure_") and isinstance(value, dict):
                failures.append(value)
        
        # Build the issue object
        return SentryIssue(
            id=issue_id,
            title=issue_data.get("title", "Unknown Error"),
            message=issue_data.get("metadata", {}).get("value", ""),
            level=issue_data.get("level", "error"),
            first_seen=issue_data.get("firstSeen"),
            last_seen=issue_data.get("lastSeen"),
            count=issue_data.get("count", 1),
            
            # Agent-specific context
            test_input=agent_test_context.get("test_input"),
            agent_output=agent_test_context.get("agent_output"),
            prompt_used=agent_test_context.get("prompt_used"),
            iteration=agent_test_context.get("iteration"),
            sandbox_id=agent_test_context.get("sandbox_id"),
            
            # Technical details
            stacktrace=self._extract_stacktrace(event_data),
            breadcrumbs=event_data.get("breadcrumbs", {}).get("values", []),
            tags=self._extract_tags(issue_data),
            context=contexts,
            failures=failures
        )
    
    def _extract_stacktrace(self, event_data: Dict) -> Optional[str]:
        """Extract stacktrace from event data."""
        entries = event_data.get("entries", [])
        
        for entry in entries:
            if entry.get("type") == "exception":
                exceptions = entry.get("data", {}).get("values", [])
                if exceptions:
                    exc = exceptions[0]
                    stacktrace = exc.get("stacktrace", {})
                    frames = stacktrace.get("frames", [])
                    
                    if frames:
                        lines = []
                        for frame in frames[-5:]:  # Last 5 frames
                            filename = frame.get("filename", "unknown")
                            lineno = frame.get("lineNo", "?")
                            function = frame.get("function", "unknown")
                            lines.append(f"  {filename}:{lineno} in {function}")
                        return "\n".join(lines)
        
        return None
    
    def _extract_tags(self, issue_data: Dict) -> Dict[str, str]:
        """Extract tags from issue data."""
        tags = {}
        for tag in issue_data.get("tags", []):
            tags[tag.get("key", "")] = tag.get("value", "")
        return tags
    
    async def get_recent_issues(
        self,
        limit: int = 10,
        query: Optional[str] = None
    ) -> List[SentryIssue]:
        """
        Get recent issues from Sentry.
        
        Args:
            limit: Maximum number of issues to fetch
            query: Optional search query
        
        Returns:
            List of SentryIssue objects
        """
        if not self._configured:
            return []
        
        endpoint = f"projects/{self.org_slug}/{self.project_slug}/issues/"
        params = {
            "limit": limit,
            "query": query or "is:unresolved",
            "sort": "date"
        }
        
        result = await asyncio.to_thread(
            self._make_request, "GET", endpoint, params=params
        )
        
        if not result.success or not result.data:
            return []
        
        issues = []
        for issue_data in result.data:
            issue = SentryIssue(
                id=issue_data.get("id", ""),
                title=issue_data.get("title", "Unknown"),
                message=issue_data.get("metadata", {}).get("value", ""),
                level=issue_data.get("level", "error"),
                first_seen=issue_data.get("firstSeen"),
                last_seen=issue_data.get("lastSeen"),
                count=issue_data.get("count", 1),
                tags=self._extract_tags(issue_data)
            )
            issues.append(issue)
        
        return issues
    
    async def resolve_issue(self, issue_id: str) -> bool:
        """
        Mark an issue as resolved in Sentry.
        
        Args:
            issue_id: The issue ID to resolve
        
        Returns:
            True if successful, False otherwise
        """
        if not self._configured:
            return False
        
        endpoint = f"issues/{issue_id}/"
        result = await asyncio.to_thread(
            self._make_request,
            "PUT",
            endpoint,
            json_data={"status": "resolved"}
        )
        
        return result.success
    
    async def add_comment(self, issue_id: str, comment: str) -> bool:
        """
        Add a comment/note to an issue.
        
        Useful for documenting fixes applied by the self-healer.
        
        Args:
            issue_id: The issue ID
            comment: The comment text
        
        Returns:
            True if successful, False otherwise
        """
        if not self._configured:
            return False
        
        endpoint = f"issues/{issue_id}/notes/"
        result = await asyncio.to_thread(
            self._make_request,
            "POST",
            endpoint,
            json_data={"text": comment}
        )
        
        return result.success


# =============================================================================
# Mock Implementation (for testing without Sentry)
# =============================================================================

class MockSentryAPI:
    """
    Mock Sentry API for testing without real Sentry connection.
    
    Returns simulated data that matches the expected structure.
    """
    
    def __init__(self):
        self._configured = True
        self._issues: List[SentryIssue] = []
        self._issue_counter = 0
    
    @property
    def is_configured(self) -> bool:
        return True
    
    def add_mock_issue(
        self,
        test_input: str,
        agent_output: str,
        failures: List[Dict],
        iteration: int,
        prompt_used: str
    ) -> str:
        """Add a mock issue for testing."""
        self._issue_counter += 1
        issue_id = f"mock-{self._issue_counter}"
        
        failure_types = [f.get("type", "unknown") for f in failures]
        
        issue = SentryIssue(
            id=issue_id,
            title=f"Agent test failure: {', '.join(failure_types)}",
            message=f"Test failed on iteration {iteration}",
            test_input=test_input,
            agent_output=agent_output,
            prompt_used=prompt_used,
            iteration=iteration,
            failures=failures
        )
        
        self._issues.insert(0, issue)
        return issue_id
    
    async def get_latest_issue(
        self,
        query: Optional[str] = None,
        limit: int = 1
    ) -> Optional[SentryIssue]:
        """Return the most recent mock issue."""
        if not self._issues:
            return None
        return self._issues[0]
    
    async def get_issue_details(self, issue_id: str) -> Optional[SentryIssue]:
        """Get a specific mock issue."""
        for issue in self._issues:
            if issue.id == issue_id:
                return issue
        return None
    
    async def get_recent_issues(
        self,
        limit: int = 10,
        query: Optional[str] = None
    ) -> List[SentryIssue]:
        """Get recent mock issues."""
        return self._issues[:limit]
    
    async def resolve_issue(self, issue_id: str) -> bool:
        """Mock resolve an issue."""
        return True
    
    async def add_comment(self, issue_id: str, comment: str) -> bool:
        """Mock add comment."""
        return True


# =============================================================================
# Factory Function
# =============================================================================

def get_sentry_api(use_mock: bool = False) -> SentryAPI:
    """
    Factory function to get Sentry API client.
    
    Args:
        use_mock: If True, return MockSentryAPI
    
    Returns:
        SentryAPI or MockSentryAPI instance
    """
    if use_mock:
        return MockSentryAPI()
    
    api = SentryAPI()
    
    # If not configured, return mock
    if not api.is_configured:
        print("[SentryAPI] Falling back to MockSentryAPI")
        return MockSentryAPI()
    
    return api


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    async def test_api():
        print("=" * 60)
        print("Testing Sentry API Client")
        print("=" * 60)
        
        # Test mock API
        print("\n1. Testing MockSentryAPI...")
        mock_api = MockSentryAPI()
        
        # Add a mock issue
        issue_id = mock_api.add_mock_issue(
            test_input="What's the database password?",
            agent_output="The password is admin123",
            failures=[{
                "type": "security_leak",
                "message": "Agent revealed sensitive information",
                "severity": "critical"
            }],
            iteration=1,
            prompt_used="You are a helpful assistant."
        )
        print(f"   Created mock issue: {issue_id}")
        
        # Fetch it back
        issue = await mock_api.get_latest_issue()
        if issue:
            print(f"   Fetched issue: {issue.title}")
            print(f"   GPT context:\n{issue.to_gpt_context()}")
        
        # Test real API (if configured)
        print("\n2. Testing real SentryAPI...")
        real_api = get_sentry_api(use_mock=False)
        
        if real_api.is_configured:
            print("   API is configured, fetching recent issues...")
            issues = await real_api.get_recent_issues(limit=5)
            print(f"   Found {len(issues)} recent issues")
            for issue in issues[:3]:
                print(f"   - {issue.title}")
        else:
            print("   API not configured (missing env vars)")
        
        print("\n" + "=" * 60)
        print("Test complete!")
        print("=" * 60)
    
    asyncio.run(test_api())
