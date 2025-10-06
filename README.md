# Agent Harness - Unified OpenAI & Anthropic SDK Interface

A production-ready harness for **hot-swapping** between OpenAI Agents SDK and Anthropic Claude Agent SDK with a unified tool registry and common API.

## Key Features

✅ **Unified Tool Registry** - Register tools once, use with any provider  
✅ **Hot-Swapping** - Switch providers at runtime without code changes  
✅ **Lazy Loading** - SDKs imported only when needed  
✅ **Streaming Support** - Real-time response streaming for both providers  
✅ **Provider Comparison** - Run same prompt on multiple providers  
✅ **Type-Safe** - Full type hints and parameter extraction  

## Architecture

```
┌─────────────────────────────────────────┐
│         AgentHarness                    │
│  • switch_provider()                    │
│  • run() / stream()                     │
│  • compare_providers()                  │
└─────────────┬───────────────────────────┘
              │
       ┌──────┴──────┐
       │             │
┌──────▼──────┐ ┌───▼──────────┐
│BaseHarness  │ │ToolRegistry  │
│(Abstract)   │ │@register_tool│
└──────┬──────┘ └──────────────┘
       │
  ┌────┴────┐
  │         │
┌─▼─────┐ ┌─▼────────┐
│OpenAI │ │Claude    │
│Harness│ │Harness   │
└───┬───┘ └───┬──────┘
    │         │
┌───▼───┐ ┌───▼──────┐
│OpenAI │ │Claude    │
│Agents │ │Agent SDK │
│SDK    │ │+ MCP     │
└───────┘ └──────────┘
```

## Quick Start

### 1. Installation

```bash
# Install both SDKs
pip install openai-agents claude-agent-sdk

# Or with uv
uv add openai-agents claude-agent-sdk
```

### 2. Set API Keys

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. Basic Usage

```python
import asyncio
from agent_harness import AgentHarness, HarnessConfig, register_tool


# Register a tool once - works with both providers
@register_tool(description="Get weather for a city")
def get_weather(city: str) -> str:
    return f"Weather in {city}: sunny, 72°F"


async def main():
    config = HarnessConfig(
        system_prompt="You are a helpful assistant",
        tool_names=["get_weather"],
        max_turns=5
    )
    
    # Start with OpenAI
    harness = AgentHarness(provider="openai", config=config)
    result = await harness.run("What's the weather in Tokyo?")
    print(f"OpenAI: {result.final_output}")
    
    # Hot-swap to Claude
    harness.switch_provider("claude")
    result = await harness.run("What's the weather in Tokyo?")
    print(f"Claude: {result.final_output}")


asyncio.run(main())
```

## Core Concepts

### Tool Registry

The `@register_tool` decorator adds tools to a global registry accessible by all providers:

```python
from agent_harness import register_tool

@register_tool(description="Add two numbers")
def add(a: float, b: float) -> float:
    """Adds two numbers together"""
    return a + b

@register_tool(description="Calculate factorial")
def factorial(n: int) -> int:
    """Calculate n!"""
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

Tools are automatically:
- Extracted with parameter types via introspection
- Wrapped for OpenAI using `function_tool`
- Wrapped for Claude using `@tool` + in-process MCP server
- Available to both providers without duplication

### Configuration

`HarnessConfig` provides unified configuration:

```python
from agent_harness import HarnessConfig

config = HarnessConfig(
    system_prompt="You are an expert data analyst",
    model="gpt-4o",              # Provider-specific model
    max_turns=10,                # Max conversation turns
    temperature=0.7,             # LLM temperature
    tool_names=["add", "multiply"],  # Specific tools to use
    provider_options={           # Provider-specific options
        "permission_mode": "acceptEdits"  # Claude-specific
    }
)
```

### Base Harness Interface

All providers implement `BaseHarness`:

```python
class BaseHarness(ABC):
    @abstractmethod
    async def run(self, prompt: str) -> AgentResponse:
        """Run agent and return final response"""
        pass
    
    @abstractmethod
    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Stream responses in real-time"""
        pass
```

## Advanced Usage

### Streaming Responses

```python
harness = AgentHarness(provider="openai", config=config)

async for chunk in harness.stream("Write a story about AI"):
    print(chunk, end="", flush=True)
```

### Provider Comparison

Compare both providers side-by-side:

```python
results = await harness.compare_providers(
    "Explain quantum computing in simple terms"
)

for provider, response in results.items():
    print(f"\n{provider.upper()}:")
    print(response.final_output)
```

### Dynamic Tool Registration

Register tools at runtime:

```python
from agent_harness import register_tool, get_registry

@register_tool(description="Convert miles to kilometers")
def miles_to_km(miles: float) -> float:
    return miles * 1.60934

print(f"Registered: {list(get_registry().get_all().keys())}")
```

### Provider-Specific Features

#### OpenAI: Guardrails & Handoffs

```python
config = HarnessConfig(
    system_prompt="You are an assistant",
    provider_options={
        "guardrails": [my_guardrail],
        "handoffs": [other_agent]
    }
)
```

#### Claude: MCP Servers & Hooks

```python
config = HarnessConfig(
    system_prompt="You are a coding assistant",
    provider_options={
        "allowed_tools": ["Read", "Write", "Bash"],
        "permission_mode": "acceptEdits",
        "cwd": "/path/to/project",
        "hooks": {
            "PreToolUse": [check_bash_hook]
        }
    }
)
```

## API Reference

### AgentHarness

Main class for hot-swapping providers.

```python
class AgentHarness:
    def __init__(
        self,
        provider: str = "openai",
        config: Optional[HarnessConfig] = None,
        api_key: Optional[str] = None
    )
    
    def switch_provider(self, provider: str) -> None
    async def run(self, prompt: str) -> AgentResponse
    async def stream(self, prompt: str) -> AsyncIterator[str]
    async def compare_providers(
        self,
        prompt: str,
        providers: Optional[list[str]] = None
    ) -> dict[str, AgentResponse]
```

**Providers:** `"openai"`, `"claude"`, `"anthropic"`

### HarnessConfig

```python
@dataclass
class HarnessConfig:
    system_prompt: str = "You are a helpful assistant."
    model: Optional[str] = None
    max_turns: int = 10
    temperature: float = 1.0
    tool_names: Optional[list[str]] = None
    provider_options: dict[str, Any] = field(default_factory=dict)
```

### AgentResponse

```python
@dataclass
class AgentResponse:
    final_output: str
    messages: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
```

### Tool Registry

```python
def register_tool(
    name: Optional[str] = None,
    description: Optional[str] = None
) -> Callable

def get_registry() -> ToolRegistry
```

## Implementation Details

### OpenAI Harness

- Lazily imports `agents` module
- Wraps tools using `function_tool` decorator
- Creates `Agent` with `model_config`
- Uses `Runner.run()` for execution
- Streams via `Runner.run_streamed()`

### Claude Harness

- Lazily imports `claude_agent_sdk`
- Wraps tools with `@tool` decorator
- Creates in-process MCP server via `create_sdk_mcp_server()`
- Uses `ClaudeAgentOptions` for configuration
- Executes via `query()` or `ClaudeSDKClient`
- Extracts `TextBlock` content from responses

## Examples

See [example_usage.py](example_usage.py) for comprehensive examples:

- ✅ Basic hot-swapping
- ✅ Tool usage with both providers
- ✅ Streaming responses
- ✅ Provider comparison
- ✅ Dynamic tool registration
- ✅ Provider-specific options

Run examples:

```bash
python example_usage.py
```

## Design Principles

1. **Single Responsibility**: Each provider adapter handles only its SDK
2. **Lazy Loading**: SDKs imported only when provider is selected
3. **Don't Repeat Yourself**: Tools registered once, used everywhere
4. **Open/Closed**: Easy to add new providers by extending `BaseHarness`
5. **Dependency Inversion**: Code depends on `BaseHarness` abstraction

## Comparison: OpenAI vs Claude

| Feature | OpenAI Agents SDK | Claude Agent SDK |
|---------|------------------|------------------|
| **Release** | March 2025 | 2025 |
| **Core API** | Responses API | Model Context Protocol (MCP) |
| **Tool Format** | `function_tool` decorator | `@tool` + MCP server |
| **Streaming** | `run_streamed()` | `query()` iterator |
| **Built-in Tools** | Web search, file search, code interpreter | Read, Write, Bash, etc. |
| **Handoffs** | Native support | Programmatic subagents |
| **Guardrails** | Built-in | Hooks (PreToolUse, PostToolUse) |
| **Sessions** | SQLite, Redis | Session forking |

## Extending the Harness

To add a new provider:

```python
class MyProviderHarness(BaseHarness):
    async def run(self, prompt: str) -> AgentResponse:
        # Import SDK lazily
        from my_sdk import Agent
        
        # Get registered tools
        tools = self._get_registered_tools()
        
        # Wrap and execute
        # ...
        
        return AgentResponse(...)
    
    async def stream(self, prompt: str) -> AsyncIterator[str]:
        # Implement streaming
        # ...

# Register in AgentHarness.PROVIDERS
AgentHarness.PROVIDERS["myprovider"] = MyProviderHarness
```

## Testing

```bash
# Run examples
python example_usage.py

# Test with both providers
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
python example_usage.py
```

## Repository Structure

```
agent_sdk/
├── agent_harness.py              # Core harness implementation
├── example_usage.py              # Comprehensive examples
├── README.md                     # This file
├── openai-agents-python/         # OpenAI SDK (cloned)
└── claude-agent-sdk-python/      # Claude SDK (cloned)
```

## License

MIT
