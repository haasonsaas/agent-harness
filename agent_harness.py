"""
Unified Agent Harness for OpenAI and Anthropic Agent SDKs

Provides a common interface with hot-swapping capability between providers.
Uses a unified tool registry to avoid duplicating tool definitions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Optional
import inspect
import os


@dataclass
class ToolMetadata:
    """Metadata for a registered tool"""
    name: str
    description: str
    callable: Callable
    parameters: dict[str, type]


class ToolRegistry:
    """Global registry for tools that can be used by any provider"""
    
    def __init__(self):
        self._tools: dict[str, ToolMetadata] = {}
    
    def register(
        self,
        name: str,
        description: str,
        callable: Callable,
        parameters: Optional[dict[str, type]] = None
    ) -> None:
        """Register a tool"""
        if parameters is None:
            parameters = self._extract_parameters(callable)
        
        self._tools[name] = ToolMetadata(
            name=name,
            description=description,
            callable=callable,
            parameters=parameters
        )
    
    def get(self, name: str) -> Optional[ToolMetadata]:
        """Get tool metadata by name"""
        return self._tools.get(name)
    
    def get_all(self) -> dict[str, ToolMetadata]:
        """Get all registered tools"""
        return self._tools.copy()
    
    def clear(self) -> None:
        """Clear all registered tools"""
        self._tools.clear()
    
    @staticmethod
    def _extract_parameters(func: Callable) -> dict[str, type]:
        """Extract parameter types from function signature"""
        sig = inspect.signature(func)
        params = {}
        for name, param in sig.parameters.items():
            if param.annotation != inspect.Parameter.empty:
                params[name] = param.annotation
            else:
                params[name] = str
        return params


_global_registry = ToolRegistry()


def register_tool(name: Optional[str] = None, description: Optional[str] = None):
    """
    Decorator to register a tool in the global registry.
    
    Usage:
        @register_tool(description="Get weather for a city")
        def get_weather(city: str) -> str:
            return f"Weather in {city}"
    """
    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"Tool: {tool_name}"
        
        _global_registry.register(
            name=tool_name,
            description=tool_description,
            callable=func
        )
        
        return func
    
    return decorator


def get_registry() -> ToolRegistry:
    """Get the global tool registry"""
    return _global_registry


@dataclass
class HarnessConfig:
    """Configuration for the harness"""
    system_prompt: str = "You are a helpful assistant."
    model: Optional[str] = None
    max_turns: int = 10
    temperature: float = 1.0
    tool_names: Optional[list[str]] = None
    provider_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Unified response from agent execution"""
    final_output: str
    messages: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseHarness(ABC):
    """Abstract base class for agent harnesses"""
    
    def __init__(self, config: Optional[HarnessConfig] = None, api_key: Optional[str] = None):
        self.config = config or HarnessConfig()
        self.api_key = api_key
        self.registry = get_registry()
    
    @abstractmethod
    async def run(self, prompt: str) -> AgentResponse:
        """
        Run the agent with a prompt and return the final response.
        
        Args:
            prompt: User input prompt
            
        Returns:
            AgentResponse with final output and metadata
        """
        pass
    
    @abstractmethod
    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """
        Stream agent responses in real-time.
        
        Args:
            prompt: User input prompt
            
        Yields:
            Text chunks as they become available
        """
        pass
    
    def _get_registered_tools(self) -> list[ToolMetadata]:
        """Get tools specified in config or all registered tools"""
        if self.config.tool_names:
            return [
                self.registry.get(name)
                for name in self.config.tool_names
                if self.registry.get(name)
            ]
        return list(self.registry.get_all().values())


class OpenAIHarness(BaseHarness):
    """Harness for OpenAI Agents SDK"""
    
    def __init__(self, config: Optional[HarnessConfig] = None, api_key: Optional[str] = None):
        super().__init__(config, api_key)
        self._agent = None
        self._setup_api_key()
    
    def _setup_api_key(self):
        """Set up API key from constructor or environment"""
        if self.api_key:
            os.environ["OPENAI_API_KEY"] = self.api_key
    
    def _create_agent(self):
        """Lazily create OpenAI agent with registered tools"""
        from agents import Agent, function_tool
        
        tools = []
        for tool_meta in self._get_registered_tools():
            wrapped = function_tool(tool_meta.callable)
            tools.append(wrapped)
        
        model_config = {}
        if self.config.model:
            model_config["model"] = self.config.model
        if self.config.temperature:
            model_config["temperature"] = self.config.temperature
        
        self._agent = Agent(
            name="Agent",
            instructions=self.config.system_prompt,
            tools=tools,
            model_config=model_config if model_config else None,
        )
    
    async def run(self, prompt: str) -> AgentResponse:
        """Run OpenAI agent and return final response"""
        from agents import Runner
        
        if not self._agent:
            self._create_agent()
        
        result = await Runner.run(
            self._agent,
            input=prompt,
            max_turns=self.config.max_turns,
        )
        
        return AgentResponse(
            final_output=result.final_output,
            messages=result.messages,
            metadata={
                "provider": "openai",
                "agent_name": result.active_agent.name,
                "turns": len(result.messages),
            }
        )
    
    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Stream OpenAI agent responses"""
        from agents import Runner
        
        if not self._agent:
            self._create_agent()
        
        async for event in Runner.run_streamed(
            self._agent,
            input=prompt,
            max_turns=self.config.max_turns
        ):
            if hasattr(event, 'delta') and event.delta:
                yield event.delta


class ClaudeHarness(BaseHarness):
    """Harness for Anthropic Claude Agent SDK"""
    
    def __init__(self, config: Optional[HarnessConfig] = None, api_key: Optional[str] = None):
        super().__init__(config, api_key)
        self._mcp_server = None
        self._client_options = None
        self._setup_api_key()
    
    def _setup_api_key(self):
        """Set up API key from constructor or environment"""
        if self.api_key:
            os.environ["ANTHROPIC_API_KEY"] = self.api_key
    
    def _create_mcp_server(self):
        """Create in-process MCP server with registered tools"""
        from claude_agent_sdk import tool, create_sdk_mcp_server
        
        tool_functions = []
        
        for tool_meta in self._get_registered_tools():
            input_schema = {
                param_name: param_type
                for param_name, param_type in tool_meta.parameters.items()
            }
            
            wrapped_tool = tool(
                tool_meta.name,
                tool_meta.description,
                input_schema
            )(tool_meta.callable)
            
            tool_functions.append(wrapped_tool)
        
        if tool_functions:
            self._mcp_server = create_sdk_mcp_server(
                name="harness-tools",
                version="1.0.0",
                tools=tool_functions
            )
    
    def _create_client_options(self):
        """Create Claude client options"""
        from claude_agent_sdk import ClaudeAgentOptions
        
        if not self._mcp_server:
            self._create_mcp_server()
        
        options_dict = {
            "system_prompt": self.config.system_prompt,
            "max_turns": self.config.max_turns,
        }
        
        if self.config.model:
            options_dict["model"] = self.config.model
        
        if self._mcp_server:
            options_dict["mcp_servers"] = {"harness": self._mcp_server}
            
            tool_names = [
                f"mcp__harness__{tool_meta.name}"
                for tool_meta in self._get_registered_tools()
            ]
            if tool_names:
                options_dict["allowed_tools"] = tool_names
        
        options_dict.update(self.config.provider_options)
        
        self._client_options = ClaudeAgentOptions(**options_dict)
    
    async def run(self, prompt: str) -> AgentResponse:
        """Run Claude agent and return final response"""
        from claude_agent_sdk import query, AssistantMessage, TextBlock, ResultMessage
        
        if not self._client_options:
            self._create_client_options()
        
        messages = []
        final_text = ""
        
        async for message in query(prompt=prompt, options=self._client_options):
            messages.append(message)
            
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        final_text = block.text
            elif isinstance(message, ResultMessage):
                final_text = str(message.result)
        
        return AgentResponse(
            final_output=final_text,
            messages=messages,
            metadata={
                "provider": "claude",
                "message_count": len(messages),
            }
        )
    
    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Stream Claude agent responses"""
        from claude_agent_sdk import query, AssistantMessage, TextBlock
        
        if not self._client_options:
            self._create_client_options()
        
        async for message in query(prompt=prompt, options=self._client_options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        yield block.text


class AgentHarness:
    """
    High-level harness that supports hot-swapping between providers.
    
    Usage:
        harness = AgentHarness(provider="openai", config=config)
        result = await harness.run("Hello")
        
        # Hot-swap to Claude
        harness.switch_provider("claude")
        result = await harness.run("Hello")
    """
    
    PROVIDERS = {
        "openai": OpenAIHarness,
        "claude": ClaudeHarness,
        "anthropic": ClaudeHarness,
    }
    
    def __init__(
        self,
        provider: str = "openai",
        config: Optional[HarnessConfig] = None,
        api_key: Optional[str] = None
    ):
        self.provider_name = provider.lower()
        self.config = config or HarnessConfig()
        self.api_key = api_key
        
        if self.provider_name not in self.PROVIDERS:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Available: {list(self.PROVIDERS.keys())}"
            )
        
        self._harness = self._create_harness()
    
    def _create_harness(self) -> BaseHarness:
        """Create provider-specific harness"""
        harness_class = self.PROVIDERS[self.provider_name]
        return harness_class(config=self.config, api_key=self.api_key)
    
    def switch_provider(self, provider: str) -> None:
        """
        Hot-swap to a different provider.
        
        Args:
            provider: Provider name ("openai", "claude", or "anthropic")
        """
        provider = provider.lower()
        if provider not in self.PROVIDERS:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Available: {list(self.PROVIDERS.keys())}"
            )
        
        self.provider_name = provider
        self._harness = self._create_harness()
        print(f"✓ Switched to {provider} provider")
    
    async def run(self, prompt: str) -> AgentResponse:
        """Run agent with current provider"""
        return await self._harness.run(prompt)
    
    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Stream responses with current provider"""
        async for chunk in self._harness.stream(prompt):
            yield chunk
    
    async def compare_providers(
        self,
        prompt: str,
        providers: Optional[list[str]] = None
    ) -> dict[str, AgentResponse]:
        """
        Run the same prompt on multiple providers for comparison.
        
        Args:
            prompt: User prompt
            providers: List of provider names (defaults to all)
            
        Returns:
            Dictionary mapping provider name to response
        """
        if providers is None:
            providers = ["openai", "claude"]
        
        original_provider = self.provider_name
        results = {}
        
        for provider in providers:
            try:
                self.switch_provider(provider)
                results[provider] = await self.run(prompt)
            except Exception as e:
                results[provider] = AgentResponse(
                    final_output=f"Error: {str(e)}",
                    metadata={"error": True, "provider": provider}
                )
        
        self.switch_provider(original_provider)
        return results
