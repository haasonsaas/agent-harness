"""Tests for error handling"""

import pytest

from agent_harness import (
    HarnessError,
    ProviderError,
    RateLimitError,
    SchemaError,
    TimeoutError,
    ToolError,
)


def test_harness_error():
    """Test base HarnessError"""
    error = HarnessError("Test error")
    assert str(error) == "Test error"
    assert isinstance(error, Exception)


def test_provider_error():
    """Test ProviderError"""
    error = ProviderError("Provider failed", "openai", retryable=True, request_id="123")

    assert str(error) == "Provider failed"
    assert error.provider == "openai"
    assert error.retryable is True
    assert error.context["request_id"] == "123"


def test_rate_limit_error():
    """Test RateLimitError is retryable"""
    error = RateLimitError("Rate limited", "claude", request_id="456")

    assert error.retryable is True
    assert error.provider == "claude"
    assert error.context["request_id"] == "456"


def test_timeout_error():
    """Test TimeoutError is retryable"""
    error = TimeoutError("Request timeout", "openai")

    assert error.retryable is True
    assert error.provider == "openai"


def test_tool_error():
    """Test ToolError"""
    error = ToolError("Tool registration failed")
    assert str(error) == "Tool registration failed"
    assert isinstance(error, HarnessError)


def test_schema_error():
    """Test SchemaError"""
    error = SchemaError("Invalid schema")
    assert str(error) == "Invalid schema"
    assert isinstance(error, HarnessError)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
