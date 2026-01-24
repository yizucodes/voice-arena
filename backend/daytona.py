"""
Daytona Sandbox Integration Module

Provides sandbox isolation for running voice agent tests.
Supports both real Daytona API and mock mode for testing.
"""

import os
import uuid
import asyncio
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CommandResult:
    """Result from executing a command in a sandbox."""
    success: bool
    output: str
    exit_code: int
    error: Optional[str] = None
    duration_seconds: float = 0.0


@dataclass
class CodeResult:
    """Result from executing code in a sandbox."""
    success: bool
    output: str
    exit_code: int
    error: Optional[str] = None
    duration_seconds: float = 0.0


# =============================================================================
# Abstract Sandbox Interface
# =============================================================================

class BaseSandbox(ABC):
    """Abstract base class for sandbox implementations."""
    
    def __init__(self, sandbox_id: str, name: str = ""):
        self.id = sandbox_id
        self.name = name
        self.created_at = datetime.now(timezone.utc)
        self._cleaned_up = False
    
    @abstractmethod
    async def run_command(self, command: str, cwd: Optional[str] = None, 
                          timeout: Optional[int] = None) -> CommandResult:
        """Execute a shell command in the sandbox."""
        pass
    
    @abstractmethod
    async def run_code(self, code: str, language: str = "python") -> CodeResult:
        """Execute code directly in the sandbox."""
        pass
    
    @abstractmethod
    async def install_dependencies(self, packages: list[str]) -> CommandResult:
        """Install Python packages in the sandbox."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> bool:
        """Stop and remove the sandbox."""
        pass
    
    @property
    def is_active(self) -> bool:
        """Check if sandbox is still active."""
        return not self._cleaned_up


# =============================================================================
# Abstract Client Interface
# =============================================================================

class BaseDaytonaClient(ABC):
    """Abstract base class for Daytona client implementations."""
    
    @abstractmethod
    async def create_sandbox(self, name: Optional[str] = None) -> BaseSandbox:
        """Create a new isolated sandbox environment."""
        pass
    
    @abstractmethod
    async def get_sandbox(self, sandbox_id: str) -> Optional[BaseSandbox]:
        """Get an existing sandbox by ID."""
        pass
    
    @abstractmethod
    async def list_sandboxes(self) -> list[BaseSandbox]:
        """List all active sandboxes."""
        pass


# =============================================================================
# Mock Implementation (for testing without API)
# =============================================================================

class MockSandbox(BaseSandbox):
    """Mock sandbox that runs commands locally for testing."""
    
    def __init__(self, sandbox_id: str, name: str = ""):
        super().__init__(sandbox_id, name)
        self._command_history: list[str] = []
    
    async def run_command(self, command: str, cwd: Optional[str] = None,
                          timeout: Optional[int] = None) -> CommandResult:
        """Execute command locally (mock behavior)."""
        if self._cleaned_up:
            return CommandResult(
                success=False,
                output="",
                exit_code=-1,
                error="Sandbox has been cleaned up"
            )
        
        start_time = datetime.now(timezone.utc)
        self._command_history.append(command)
        
        try:
            # Run command locally using subprocess
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout or 30
                )
            except asyncio.TimeoutError:
                process.kill()
                return CommandResult(
                    success=False,
                    output="",
                    exit_code=-1,
                    error=f"Command timed out after {timeout} seconds"
                )
            
            output = stdout.decode('utf-8', errors='replace')
            error_output = stderr.decode('utf-8', errors='replace')
            
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            return CommandResult(
                success=process.returncode == 0,
                output=output + error_output if error_output else output,
                exit_code=process.returncode or 0,
                error=error_output if process.returncode != 0 else None,
                duration_seconds=duration
            )
            
        except Exception as e:
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            return CommandResult(
                success=False,
                output="",
                exit_code=-1,
                error=str(e),
                duration_seconds=duration
            )
    
    async def run_code(self, code: str, language: str = "python") -> CodeResult:
        """Execute code locally (mock behavior)."""
        if self._cleaned_up:
            return CodeResult(
                success=False,
                output="",
                exit_code=-1,
                error="Sandbox has been cleaned up"
            )
        
        start_time = datetime.now(timezone.utc)
        
        if language != "python":
            return CodeResult(
                success=False,
                output="",
                exit_code=-1,
                error=f"Mock sandbox only supports Python, got: {language}"
            )
        
        try:
            # Execute Python code
            process = await asyncio.create_subprocess_exec(
                "python", "-c", code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=30
            )
            
            output = stdout.decode('utf-8', errors='replace')
            error_output = stderr.decode('utf-8', errors='replace')
            
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            return CodeResult(
                success=process.returncode == 0,
                output=output + error_output if error_output else output,
                exit_code=process.returncode or 0,
                error=error_output if process.returncode != 0 else None,
                duration_seconds=duration
            )
            
        except asyncio.TimeoutError:
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            return CodeResult(
                success=False,
                output="",
                exit_code=-1,
                error="Code execution timed out",
                duration_seconds=duration
            )
        except Exception as e:
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            return CodeResult(
                success=False,
                output="",
                exit_code=-1,
                error=str(e),
                duration_seconds=duration
            )
    
    async def install_dependencies(self, packages: list[str]) -> CommandResult:
        """Mock dependency installation."""
        if not packages:
            return CommandResult(
                success=True,
                output="No packages to install",
                exit_code=0
            )
        
        # In mock mode, just simulate success
        package_list = " ".join(packages)
        return CommandResult(
            success=True,
            output=f"[Mock] Successfully installed: {package_list}",
            exit_code=0,
            duration_seconds=0.1
        )
    
    async def cleanup(self) -> bool:
        """Mark sandbox as cleaned up."""
        self._cleaned_up = True
        return True


class MockDaytonaClient(BaseDaytonaClient):
    """Mock Daytona client for testing without API."""
    
    def __init__(self):
        self._sandboxes: Dict[str, MockSandbox] = {}
    
    async def create_sandbox(self, name: Optional[str] = None) -> MockSandbox:
        """Create a mock sandbox."""
        sandbox_id = f"mock-{uuid.uuid4().hex[:8]}"
        sandbox_name = name or f"sandbox-{sandbox_id}"
        
        sandbox = MockSandbox(sandbox_id, sandbox_name)
        self._sandboxes[sandbox_id] = sandbox
        
        return sandbox
    
    async def get_sandbox(self, sandbox_id: str) -> Optional[MockSandbox]:
        """Get an existing mock sandbox."""
        return self._sandboxes.get(sandbox_id)
    
    async def list_sandboxes(self) -> list[MockSandbox]:
        """List all mock sandboxes."""
        return [s for s in self._sandboxes.values() if s.is_active]


# =============================================================================
# Real Daytona Implementation
# =============================================================================

class DaytonaSandbox(BaseSandbox):
    """Real Daytona sandbox wrapper."""
    
    def __init__(self, sandbox_id: str, name: str, daytona_sandbox: Any):
        super().__init__(sandbox_id, name)
        self._daytona_sandbox = daytona_sandbox
    
    async def run_command(self, command: str, cwd: Optional[str] = None,
                          timeout: Optional[int] = None) -> CommandResult:
        """Execute command in real Daytona sandbox."""
        if self._cleaned_up:
            return CommandResult(
                success=False,
                output="",
                exit_code=-1,
                error="Sandbox has been cleaned up"
            )
        
        start_time = datetime.now(timezone.utc)
        
        try:
            # Use process.exec from Daytona SDK
            response = await asyncio.to_thread(
                self._daytona_sandbox.process.exec,
                command,
                cwd=cwd,
                timeout=timeout
            )
            
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            return CommandResult(
                success=response.exit_code == 0,
                output=response.result or "",
                exit_code=response.exit_code,
                error=None if response.exit_code == 0 else response.result,
                duration_seconds=duration
            )
            
        except Exception as e:
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            return CommandResult(
                success=False,
                output="",
                exit_code=-1,
                error=str(e),
                duration_seconds=duration
            )
    
    async def run_code(self, code: str, language: str = "python") -> CodeResult:
        """Execute code in real Daytona sandbox."""
        if self._cleaned_up:
            return CodeResult(
                success=False,
                output="",
                exit_code=-1,
                error="Sandbox has been cleaned up"
            )
        
        start_time = datetime.now(timezone.utc)
        
        try:
            # Use process.code_run from Daytona SDK
            response = await asyncio.to_thread(
                self._daytona_sandbox.process.code_run,
                code
            )
            
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            return CodeResult(
                success=response.exit_code == 0,
                output=response.result or "",
                exit_code=response.exit_code,
                error=None if response.exit_code == 0 else response.result,
                duration_seconds=duration
            )
            
        except Exception as e:
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            return CodeResult(
                success=False,
                output="",
                exit_code=-1,
                error=str(e),
                duration_seconds=duration
            )
    
    async def install_dependencies(self, packages: list[str]) -> CommandResult:
        """Install Python packages in the sandbox."""
        if not packages:
            return CommandResult(
                success=True,
                output="No packages to install",
                exit_code=0
            )
        
        package_list = " ".join(packages)
        return await self.run_command(f"pip install {package_list}")
    
    async def cleanup(self) -> bool:
        """Stop and remove the sandbox."""
        if self._cleaned_up:
            return True
        
        try:
            await asyncio.to_thread(self._daytona_sandbox.delete)
            self._cleaned_up = True
            return True
        except Exception as e:
            print(f"Error cleaning up sandbox {self.id}: {e}")
            self._cleaned_up = True  # Mark as cleaned up anyway
            return False


class DaytonaClient(BaseDaytonaClient):
    """Real Daytona client using the SDK."""
    
    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        self._api_key = api_key or os.getenv("DAYTONA_API_KEY")
        self._api_url = api_url or os.getenv("DAYTONA_API_URL", "https://app.daytona.io/api")
        self._daytona = None
        self._sandboxes: Dict[str, DaytonaSandbox] = {}
        
        # Initialize Daytona SDK
        self._init_sdk()
    
    def _init_sdk(self):
        """Initialize the Daytona SDK."""
        try:
            from daytona_sdk import Daytona, DaytonaConfig
            
            config = DaytonaConfig(
                api_key=self._api_key,
                server_url=self._api_url
            )
            self._daytona = Daytona(config)
        except ImportError:
            # Try alternate import path
            try:
                from daytona import Daytona, DaytonaConfig
                
                config = DaytonaConfig(
                    api_key=self._api_key,
                    server_url=self._api_url
                )
                self._daytona = Daytona(config)
            except ImportError:
                raise ImportError(
                    "Daytona SDK not found. Install with: pip install daytona-sdk"
                )
    
    async def create_sandbox(self, name: Optional[str] = None) -> DaytonaSandbox:
        """Create a new Daytona sandbox."""
        try:
            # Create sandbox using the SDK
            sandbox_name = name or f"voice-agent-{uuid.uuid4().hex[:8]}"
            
            daytona_sandbox = await asyncio.to_thread(
                self._daytona.create,
                labels={"purpose": "voice-agent-test", "name": sandbox_name}
            )
            
            sandbox = DaytonaSandbox(
                sandbox_id=daytona_sandbox.id,
                name=sandbox_name,
                daytona_sandbox=daytona_sandbox
            )
            
            self._sandboxes[sandbox.id] = sandbox
            return sandbox
            
        except Exception as e:
            raise RuntimeError(f"Failed to create Daytona sandbox: {e}")
    
    async def get_sandbox(self, sandbox_id: str) -> Optional[DaytonaSandbox]:
        """Get an existing sandbox by ID."""
        return self._sandboxes.get(sandbox_id)
    
    async def list_sandboxes(self) -> list[DaytonaSandbox]:
        """List all active sandboxes."""
        return [s for s in self._sandboxes.values() if s.is_active]


# =============================================================================
# Async Daytona Implementation
# =============================================================================

class AsyncDaytonaSandbox(BaseSandbox):
    """Async Daytona sandbox wrapper using AsyncDaytona."""
    
    def __init__(self, sandbox_id: str, name: str, async_sandbox: Any):
        super().__init__(sandbox_id, name)
        self._async_sandbox = async_sandbox
    
    async def run_command(self, command: str, cwd: Optional[str] = None,
                          timeout: Optional[int] = None) -> CommandResult:
        """Execute command in async Daytona sandbox."""
        if self._cleaned_up:
            return CommandResult(
                success=False,
                output="",
                exit_code=-1,
                error="Sandbox has been cleaned up"
            )
        
        start_time = datetime.now(timezone.utc)
        
        try:
            response = await self._async_sandbox.process.exec(
                command,
                cwd=cwd,
                timeout=timeout
            )
            
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            return CommandResult(
                success=response.exit_code == 0,
                output=response.result or "",
                exit_code=response.exit_code,
                error=None if response.exit_code == 0 else response.result,
                duration_seconds=duration
            )
            
        except Exception as e:
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            return CommandResult(
                success=False,
                output="",
                exit_code=-1,
                error=str(e),
                duration_seconds=duration
            )
    
    async def run_code(self, code: str, language: str = "python") -> CodeResult:
        """Execute code in async Daytona sandbox."""
        if self._cleaned_up:
            return CodeResult(
                success=False,
                output="",
                exit_code=-1,
                error="Sandbox has been cleaned up"
            )
        
        start_time = datetime.now(timezone.utc)
        
        try:
            response = await self._async_sandbox.process.code_run(code)
            
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            return CodeResult(
                success=response.exit_code == 0,
                output=response.result or "",
                exit_code=response.exit_code,
                error=None if response.exit_code == 0 else response.result,
                duration_seconds=duration
            )
            
        except Exception as e:
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            return CodeResult(
                success=False,
                output="",
                exit_code=-1,
                error=str(e),
                duration_seconds=duration
            )
    
    async def install_dependencies(self, packages: list[str]) -> CommandResult:
        """Install Python packages in the sandbox."""
        if not packages:
            return CommandResult(
                success=True,
                output="No packages to install",
                exit_code=0
            )
        
        package_list = " ".join(packages)
        return await self.run_command(f"pip install {package_list}")
    
    async def cleanup(self) -> bool:
        """Stop and remove the sandbox."""
        if self._cleaned_up:
            return True
        
        try:
            await self._async_sandbox.delete()
            self._cleaned_up = True
            return True
        except Exception as e:
            print(f"Error cleaning up async sandbox {self.id}: {e}")
            self._cleaned_up = True
            return False


class AsyncDaytonaClient(BaseDaytonaClient):
    """Async Daytona client using AsyncDaytona SDK."""
    
    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        self._api_key = api_key or os.getenv("DAYTONA_API_KEY")
        self._api_url = api_url or os.getenv("DAYTONA_API_URL", "https://app.daytona.io/api")
        self._async_daytona = None
        self._sandboxes: Dict[str, AsyncDaytonaSandbox] = {}
        self._initialized = False
    
    async def _init_sdk(self):
        """Initialize the async Daytona SDK."""
        if self._initialized:
            return
        
        try:
            from daytona_sdk import AsyncDaytona, DaytonaConfig
            
            config = DaytonaConfig(
                api_key=self._api_key,
                server_url=self._api_url
            )
            self._async_daytona = AsyncDaytona(config)
            self._initialized = True
        except ImportError:
            try:
                from daytona import AsyncDaytona, DaytonaConfig
                
                config = DaytonaConfig(
                    api_key=self._api_key,
                    server_url=self._api_url
                )
                self._async_daytona = AsyncDaytona(config)
                self._initialized = True
            except ImportError:
                raise ImportError(
                    "Daytona SDK not found. Install with: pip install daytona-sdk"
                )
    
    async def create_sandbox(self, name: Optional[str] = None) -> AsyncDaytonaSandbox:
        """Create a new async Daytona sandbox."""
        await self._init_sdk()
        
        try:
            sandbox_name = name or f"voice-agent-{uuid.uuid4().hex[:8]}"
            
            async_sandbox = await self._async_daytona.create(
                labels={"purpose": "voice-agent-test", "name": sandbox_name}
            )
            
            sandbox = AsyncDaytonaSandbox(
                sandbox_id=async_sandbox.id,
                name=sandbox_name,
                async_sandbox=async_sandbox
            )
            
            self._sandboxes[sandbox.id] = sandbox
            return sandbox
            
        except Exception as e:
            raise RuntimeError(f"Failed to create async Daytona sandbox: {e}")
    
    async def get_sandbox(self, sandbox_id: str) -> Optional[AsyncDaytonaSandbox]:
        """Get an existing sandbox by ID."""
        return self._sandboxes.get(sandbox_id)
    
    async def list_sandboxes(self) -> list[AsyncDaytonaSandbox]:
        """List all active sandboxes."""
        return [s for s in self._sandboxes.values() if s.is_active]


# =============================================================================
# Factory Function
# =============================================================================

def _is_daytona_sdk_available() -> bool:
    """Check if the Daytona SDK is installed."""
    try:
        from daytona_sdk import Daytona, DaytonaConfig
        return True
    except ImportError:
        try:
            # Try alternate import path (in case the module is named differently)
            import importlib
            importlib.import_module('daytona_sdk')
            return True
        except ImportError:
            return False


def get_daytona_client(
    use_mock: bool = False,
    use_async: bool = True,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None
) -> BaseDaytonaClient:
    """
    Factory function to get the appropriate Daytona client.
    
    Args:
        use_mock: If True, always return MockDaytonaClient
        use_async: If True and not mocking, prefer AsyncDaytonaClient
        api_key: Optional API key (falls back to env var)
        api_url: Optional API URL (falls back to env var)
    
    Returns:
        A Daytona client instance (Mock, Async, or Sync)
    
    Fallback behavior:
        1. If use_mock=True → MockDaytonaClient
        2. If API key not available → MockDaytonaClient (with warning)
        3. If SDK not installed → MockDaytonaClient (with warning)
        4. Otherwise → AsyncDaytonaClient or DaytonaClient
    """
    # Always use mock if explicitly requested
    if use_mock:
        return MockDaytonaClient()
    
    # Check for API key
    resolved_api_key = api_key or os.getenv("DAYTONA_API_KEY")
    if not resolved_api_key:
        print("⚠️  DAYTONA_API_KEY not set. Falling back to mock client.")
        return MockDaytonaClient()
    
    # Check if SDK is available before trying to create real client
    if not _is_daytona_sdk_available():
        print("⚠️  Daytona SDK not installed. Falling back to mock client.")
        print("   Install with: pip install daytona-sdk")
        return MockDaytonaClient()
    
    # Try to create real client
    try:
        if use_async:
            return AsyncDaytonaClient(api_key=resolved_api_key, api_url=api_url)
        else:
            return DaytonaClient(api_key=resolved_api_key, api_url=api_url)
    except ImportError as e:
        print(f"⚠️  Daytona SDK import error ({e}). Falling back to mock client.")
        return MockDaytonaClient()
    except Exception as e:
        print(f"⚠️  Failed to initialize Daytona client ({e}). Falling back to mock client.")
        return MockDaytonaClient()


# =============================================================================
# Utility Functions
# =============================================================================

async def test_daytona_connection(use_mock: bool = False) -> dict:
    """
    Test Daytona connection by creating a sandbox and running a simple command.
    
    Returns:
        Dictionary with test results
    """
    results = {
        "client_type": "unknown",
        "sandbox_created": False,
        "command_executed": False,
        "output": "",
        "cleanup_success": False,
        "error": None
    }
    
    try:
        client = get_daytona_client(use_mock=use_mock)
        results["client_type"] = type(client).__name__
        
        # Create sandbox
        sandbox = await client.create_sandbox(name="test-connection")
        results["sandbox_created"] = True
        results["sandbox_id"] = sandbox.id
        
        # Run test command
        result = await sandbox.run_command("echo 'Hello from Daytona!'")
        results["command_executed"] = result.success
        results["output"] = result.output.strip()
        
        # Cleanup
        cleanup_success = await sandbox.cleanup()
        results["cleanup_success"] = cleanup_success
        
    except Exception as e:
        results["error"] = str(e)
    
    return results


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    import sys
    
    async def main():
        use_mock = "--mock" in sys.argv or "-m" in sys.argv
        
        print("=" * 60)
        print(f"Testing Daytona Integration (mock={use_mock})")
        print("=" * 60)
        
        results = await test_daytona_connection(use_mock=use_mock)
        
        print(f"\nClient Type: {results['client_type']}")
        print(f"Sandbox Created: {results['sandbox_created']}")
        print(f"Command Executed: {results['command_executed']}")
        print(f"Output: {results['output']}")
        print(f"Cleanup Success: {results['cleanup_success']}")
        
        if results['error']:
            print(f"Error: {results['error']}")
        
        # Additional tests
        print("\n" + "=" * 60)
        print("Running additional tests...")
        print("=" * 60)
        
        client = get_daytona_client(use_mock=use_mock)
        sandbox = await client.create_sandbox(name="test-additional")
        
        # Test multiple commands
        print("\n1. Testing echo command:")
        result = await sandbox.run_command("echo 'Hello World'")
        print(f"   Output: {result.output.strip()}")
        print(f"   Success: {result.success}")
        
        print("\n2. Testing Python code execution:")
        code_result = await sandbox.run_code("print('Python says hello!')")
        print(f"   Output: {code_result.output.strip()}")
        print(f"   Success: {code_result.success}")
        
        print("\n3. Testing dependency installation (mock):")
        dep_result = await sandbox.install_dependencies(["requests", "openai"])
        print(f"   Output: {dep_result.output}")
        print(f"   Success: {dep_result.success}")
        
        # Cleanup
        print("\n4. Cleaning up sandbox...")
        await sandbox.cleanup()
        print("   Sandbox cleaned up!")
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)
    
    asyncio.run(main())
