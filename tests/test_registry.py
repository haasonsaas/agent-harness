"""Tests for ToolRegistry"""

import pytest
import asyncio
from typing import Optional
from agent_harness import (
    ToolRegistry,
    register_tool,
    get_registry,
    python_type_to_json_schema
)


def test_python_type_to_json_schema():
    """Test JSON schema generation for Python types"""
    assert python_type_to_json_schema(str) == {"type": "string"}
    assert python_type_to_json_schema(int) == {"type": "integer"}
    assert python_type_to_json_schema(float) == {"type": "number"}
    assert python_type_to_json_schema(bool) == {"type": "boolean"}
    assert python_type_to_json_schema(list) == {"type": "array"}
    assert python_type_to_json_schema(dict) == {"type": "object"}


def test_python_type_to_json_schema_optional():
    """Test Optional type handling"""
    from typing import Optional
    schema = python_type_to_json_schema(Optional[str])
    assert schema == {"type": "string"}


def test_python_type_to_json_schema_list():
    """Test List[T] handling"""
    from typing import List
    schema = python_type_to_json_schema(List[str])
    assert schema == {"type": "array", "items": {"type": "string"}}


def test_registry_register():
    """Test tool registration"""
    registry = ToolRegistry()
    
    def sample_tool(x: int) -> int:
        return x * 2
    
    registry.register("sample", "A sample tool", sample_tool)
    
    tool = registry.get("sample")
    assert tool is not None
    assert tool.name == "sample"
    assert tool.description == "A sample tool"
    assert tool.callable(5) == 10


def test_registry_register_async():
    """Test async tool registration"""
    registry = ToolRegistry()
    
    async def async_tool(x: int) -> int:
        await asyncio.sleep(0.01)
        return x * 2
    
    registry.register("async_sample", "Async sample", async_tool)
    
    tool = registry.get("async_sample")
    assert tool is not None
    assert tool.is_async is True


def test_registry_json_schema():
    """Test JSON schema generation"""
    registry = ToolRegistry()
    
    def tool_with_params(name: str, age: int, score: float = 0.0) -> str:
        """A tool with parameters"""
        return f"{name} is {age} years old"
    
    registry.register("parameterized", "Tool with params", tool_with_params)
    
    tool = registry.get("parameterized")
    schema = tool.json_schema
    
    assert schema["type"] == "object"
    assert "properties" in schema
    assert "name" in schema["properties"]
    assert "age" in schema["properties"]
    assert "score" in schema["properties"]
    assert schema["properties"]["name"]["type"] == "string"
    assert schema["properties"]["age"]["type"] == "integer"
    assert schema["properties"]["score"]["type"] == "number"
    assert set(schema["required"]) == {"name", "age"}


def test_registry_decorator():
    """Test @register_tool decorator"""
    # Clear global registry
    get_registry().clear()
    
    @register_tool(description="Add two numbers")
    def add(a: float, b: float) -> float:
        return a + b
    
    tool = get_registry().get("add")
    assert tool is not None
    assert tool.description == "Add two numbers"
    assert tool.callable(2, 3) == 5


def test_registry_unregister():
    """Test tool unregistration"""
    registry = ToolRegistry()
    
    def sample(x: int) -> int:
        return x
    
    registry.register("sample", "Sample", sample)
    assert registry.get("sample") is not None
    
    result = registry.unregister("sample")
    assert result is True
    assert registry.get("sample") is None
    
    result = registry.unregister("nonexistent")
    assert result is False


def test_registry_clear():
    """Test clearing registry"""
    registry = ToolRegistry()
    
    registry.register("tool1", "Tool 1", lambda x: x)
    registry.register("tool2", "Tool 2", lambda x: x)
    
    assert len(registry.get_all()) == 2
    
    registry.clear()
    assert len(registry.get_all()) == 0


def test_registry_thread_safety():
    """Test concurrent access to registry"""
    import threading
    registry = ToolRegistry()
    
    def register_tools(prefix: str):
        for i in range(10):
            registry.register(
                f"{prefix}_{i}",
                f"Tool {prefix} {i}",
                lambda x: x
            )
    
    threads = [
        threading.Thread(target=register_tools, args=(f"thread{i}",))
        for i in range(5)
    ]
    
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # Should have 50 tools (5 threads * 10 tools each)
    assert len(registry.get_all()) == 50


def test_registry_overwrite_warning(caplog):
    """Test warning on tool overwrite"""
    import logging
    registry = ToolRegistry()
    
    registry.register("duplicate", "First", lambda x: x)
    
    with caplog.at_level(logging.WARNING):
        registry.register("duplicate", "Second", lambda x: x * 2)
    
    assert "already registered" in caplog.text
    
    tool = registry.get("duplicate")
    assert tool.description == "Second"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
