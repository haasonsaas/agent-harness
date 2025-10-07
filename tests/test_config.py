"""Tests for HarnessConfig"""

import pytest
from agent_harness import HarnessConfig


def test_config_defaults():
    """Test default configuration values"""
    config = HarnessConfig()
    
    assert config.system_prompt == "You are a helpful assistant."
    assert config.model is None
    assert config.max_turns == 10
    assert config.temperature is None
    assert config.timeout_sec == 30.0
    assert config.retry_attempts == 3
    assert config.request_id is not None


def test_config_validation_max_turns():
    """Test max_turns validation"""
    with pytest.raises(ValueError, match="max_turns must be positive"):
        HarnessConfig(max_turns=0)
    
    with pytest.raises(ValueError, match="max_turns must be positive"):
        HarnessConfig(max_turns=-1)


def test_config_validation_temperature():
    """Test temperature validation"""
    with pytest.raises(ValueError, match="temperature must be between"):
        HarnessConfig(temperature=-0.1)
    
    with pytest.raises(ValueError, match="temperature must be between"):
        HarnessConfig(temperature=2.1)
    
    # Valid temperatures
    config = HarnessConfig(temperature=0.0)
    assert config.temperature == 0.0
    
    config = HarnessConfig(temperature=1.0)
    assert config.temperature == 1.0
    
    config = HarnessConfig(temperature=2.0)
    assert config.temperature == 2.0


def test_config_validation_timeout():
    """Test timeout validation"""
    with pytest.raises(ValueError, match="timeout_sec must be positive"):
        HarnessConfig(timeout_sec=0)
    
    with pytest.raises(ValueError, match="timeout_sec must be positive"):
        HarnessConfig(timeout_sec=-1)


def test_config_validation_retry():
    """Test retry_attempts validation"""
    with pytest.raises(ValueError, match="retry_attempts must be non-negative"):
        HarnessConfig(retry_attempts=-1)
    
    # Zero retries should be valid
    config = HarnessConfig(retry_attempts=0)
    assert config.retry_attempts == 0


def test_config_custom_request_id():
    """Test custom request_id"""
    config = HarnessConfig(request_id="custom-id-123")
    assert config.request_id == "custom-id-123"


def test_config_auto_request_id():
    """Test auto-generated request_id"""
    config1 = HarnessConfig()
    config2 = HarnessConfig()
    
    assert config1.request_id != config2.request_id
    assert len(config1.request_id) > 0


def test_config_provider_options():
    """Test provider_options"""
    config = HarnessConfig(
        provider_options={"key1": "value1", "key2": 42}
    )
    
    assert config.provider_options["key1"] == "value1"
    assert config.provider_options["key2"] == 42


def test_config_all_fields():
    """Test configuration with all fields set"""
    config = HarnessConfig(
        system_prompt="Custom prompt",
        model="gpt-4o",
        max_turns=20,
        temperature=0.7,
        timeout_sec=60.0,
        max_output_tokens=1000,
        top_p=0.9,
        stop_sequences=["STOP"],
        tool_names=["tool1", "tool2"],
        retry_attempts=5,
        retry_backoff=2.0,
        request_id="test-123",
        provider_options={"custom": True}
    )
    
    assert config.system_prompt == "Custom prompt"
    assert config.model == "gpt-4o"
    assert config.max_turns == 20
    assert config.temperature == 0.7
    assert config.timeout_sec == 60.0
    assert config.max_output_tokens == 1000
    assert config.top_p == 0.9
    assert config.stop_sequences == ["STOP"]
    assert config.tool_names == ["tool1", "tool2"]
    assert config.retry_attempts == 5
    assert config.retry_backoff == 2.0
    assert config.request_id == "test-123"
    assert config.provider_options["custom"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
