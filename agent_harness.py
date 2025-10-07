"""
Unified Agent Harness for OpenAI and Anthropic Agent SDKs

Provides a common interface with hot-swapping capability between providers.
Uses a unified tool registry to avoid duplicating tool definitions.
"""

import asyncio
import inspect
import logging
import os
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Callable, Optional, Union, get_args, get_origin

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Supported provider types"""

    OPENAI = "openai"
    CLAUDE = "claude"
    ANTHROPIC = "anthropic"


class HarnessError(Exception):
    """Base exception for harness errors"""

    pass


class ProviderError(HarnessError):
    """Error from provider SDK"""

    def __init__(self, message: str, provider: str, retryable: bool = False, **context):
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable
        self.context = context


class ToolError(HarnessError):
    """Error in tool registration or execution"""

    pass


class SchemaError(HarnessError):
    """Error in schema generation"""

    pass


class RateLimitError(ProviderError):
    """Provider rate limit exceeded"""

    def __init__(self, message: str, provider: str, **context):
        super().__init__(message, provider, retryable=True, **context)


class TimeoutError(ProviderError):
    """Request timeout"""

    def __init__(self, message: str, provider: str, **context):
        super().__init__(message, provider, retryable=True, **context)


def python_type_to_json_schema(param_type: type) -> dict:
    """
    Convert Python type annotations to JSON Schema.

    Supports: str, int, float, bool, list, dict, Optional, Union, Literal
    """
    origin = get_origin(param_type)

    # Handle Optional[T] -> Union[T, None]
    if origin is Union:
        args = get_args(param_type)
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return python_type_to_json_schema(non_none_args[0])
        # Multiple types (not Optional) - use first for now
        return python_type_to_json_schema(non_none_args[0])

    # Handle List[T]
    if origin is list:
        args = get_args(param_type)
        item_schema = python_type_to_json_schema(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item_schema}

    # Handle Dict[str, T]
    if origin is dict:
        return {"type": "object"}

    # Basic types
    type_map = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
    }

    return type_map.get(param_type, {"type": "string"})


@dataclass
class ToolMetadata:
    """Metadata for a registered tool"""

    name: str
    description: str
    callable: Callable
    parameters: dict[str, type]
    json_schema: dict[str, Any]
    is_async: bool = False


class ToolRegistry:
    """Global registry for tools with thread-safe operations"""

    def __init__(self):
        self._tools: dict[str, ToolMetadata] = {}
        self._lock = threading.RLock()

    def register(
        self,
        name: str,
        description: str,
        callable: Callable,
        parameters: Optional[dict[str, type]] = None,
    ) -> None:
        """Register a tool with JSON schema generation"""
        with self._lock:
            if name in self._tools:
                logger.warning(f"Tool '{name}' already registered, overwriting")

            if parameters is None:
                parameters = self._extract_parameters(callable)

            json_schema = self._build_json_schema(callable, parameters, description)
            is_async = asyncio.iscoroutinefunction(callable)

            self._tools[name] = ToolMetadata(
                name=name,
                description=description,
                callable=callable,
                parameters=parameters,
                json_schema=json_schema,
                is_async=is_async,
            )

            logger.debug(f"Registered tool: {name} (async={is_async})")

    def get(self, name: str) -> Optional[ToolMetadata]:
        """Get tool metadata by name (thread-safe)"""
        with self._lock:
            return self._tools.get(name)

    def get_all(self) -> dict[str, ToolMetadata]:
        """Get all registered tools (returns a copy)"""
        with self._lock:
            return self._tools.copy()

    def unregister(self, name: str) -> bool:
        """Remove a tool from the registry"""
        with self._lock:
            if name in self._tools:
                del self._tools[name]
                logger.debug(f"Unregistered tool: {name}")
                return True
            return False

    def clear(self) -> None:
        """Clear all registered tools"""
        with self._lock:
            self._tools.clear()
            logger.debug("Cleared all registered tools")

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

    @staticmethod
    def _build_json_schema(func: Callable, parameters: dict[str, type], description: str) -> dict:
        """Build JSON schema for function parameters"""
        sig = inspect.signature(func)
        properties = {}
        required = []

        for param_name, param_type in parameters.items():
            param_obj = sig.parameters.get(param_name)

            # Build schema for this parameter
            param_schema = python_type_to_json_schema(param_type)

            # Add description from docstring if available
            if func.__doc__:
                # Simple extraction - could be enhanced with docstring parsing
                param_schema["description"] = f"Parameter: {param_name}"

            properties[param_name] = param_schema

            # Check if required (no default value)
            if param_obj and param_obj.default == inspect.Parameter.empty:
                required.append(param_name)

        schema = {
            "type": "object",
            "properties": properties,
            "required": required,
            "description": description,
        }

        return schema


_global_registry = ToolRegistry()


def register_tool(name: Optional[str] = None, description: Optional[str] = None):
    """
    Decorator to register a tool in the global registry.

    Usage:
        @register_tool(description="Get weather for a city")
        def get_weather(city: str) -> str:
            return f"Weather in {city}"

        @register_tool(description="Async calculation")
        async def calculate(x: float, y: float) -> float:
            await asyncio.sleep(0.1)
            return x + y
    """

    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"Tool: {tool_name}"

        _global_registry.register(name=tool_name, description=tool_description, callable=func)

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
    temperature: Optional[float] = None
    timeout_sec: Optional[float] = 30.0
    max_output_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stop_sequences: Optional[list[str]] = None
    tool_names: Optional[list[str]] = None
    retry_attempts: int = 3
    retry_backoff: float = 1.0
    request_id: Optional[str] = None
    provider_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration"""
        if self.max_turns <= 0:
            raise ValueError("max_turns must be positive")

        if self.temperature is not None and not (0.0 <= self.temperature <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")

        if self.timeout_sec is not None and self.timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")

        if self.retry_attempts < 0:
            raise ValueError("retry_attempts must be non-negative")

        if not self.request_id:
            self.request_id = str(uuid.uuid4())


@dataclass
class AgentResponse:
    """Unified response from agent execution"""

    final_output: str
    provider: str
    request_id: str
    messages: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def latency_ms(self) -> Optional[float]:
        """Get latency in milliseconds"""
        return self.metadata.get("latency_ms")

    @property
    def error(self) -> bool:
        """Check if response contains an error"""
        return self.metadata.get("error", False)


class BaseHarness(ABC):
    """Abstract base class for agent harnesses"""

    def __init__(self, config: Optional[HarnessConfig] = None, api_key: Optional[str] = None):
        self.config = config or HarnessConfig()
        self.api_key = api_key
        self.registry = get_registry()
        self._closed = False

    @abstractmethod
    async def run(
        self, prompt: str, config_override: Optional[HarnessConfig] = None
    ) -> AgentResponse:
        """
        Run the agent with a prompt and return the final response.

        Args:
            prompt: User input prompt
            config_override: Optional config to override defaults for this call

        Returns:
            AgentResponse with final output and metadata
        """
        pass

    @abstractmethod
    async def stream(
        self, prompt: str, config_override: Optional[HarnessConfig] = None
    ) -> AsyncIterator[str]:
        """
        Stream agent responses in real-time (incremental deltas).

        Args:
            prompt: User input prompt
            config_override: Optional config to override defaults for this call

        Yields:
            Text deltas as they become available
        """
        pass

    async def aclose(self) -> None:
        """Clean up resources"""
        self._closed = True
        logger.debug(f"{self.__class__.__name__} closed")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()

    def _get_registered_tools(self) -> list[ToolMetadata]:
        """Get tools specified in config or all registered tools"""
        if self.config.tool_names:
            return [
                self.registry.get(name)
                for name in self.config.tool_names
                if self.registry.get(name)
            ]
        return list(self.registry.get_all().values())

    def _merge_config(self, override: Optional[HarnessConfig]) -> HarnessConfig:
        """Merge base config with override"""
        if not override:
            return self.config

        # Create new config with overrides
        merged = HarnessConfig(
            system_prompt=(
                override.system_prompt
                if override.system_prompt != "You are a helpful assistant."
                else self.config.system_prompt
            ),
            model=override.model or self.config.model,
            max_turns=override.max_turns if override.max_turns != 10 else self.config.max_turns,
            temperature=(
                override.temperature
                if override.temperature is not None
                else self.config.temperature
            ),
            timeout_sec=override.timeout_sec or self.config.timeout_sec,
            max_output_tokens=override.max_output_tokens or self.config.max_output_tokens,
            top_p=override.top_p or self.config.top_p,
            stop_sequences=override.stop_sequences or self.config.stop_sequences,
            tool_names=override.tool_names or self.config.tool_names,
            retry_attempts=(
                override.retry_attempts
                if override.retry_attempts != 3
                else self.config.retry_attempts
            ),
            retry_backoff=(
                override.retry_backoff
                if override.retry_backoff != 1.0
                else self.config.retry_backoff
            ),
            request_id=override.request_id or self.config.request_id,
            provider_options={**self.config.provider_options, **override.provider_options},
        )
        return merged


async def retry_with_backoff(
    func: Callable,
    max_attempts: int,
    backoff: float,
    retryable_errors: tuple = (RateLimitError, TimeoutError),
):
    """Retry a function with exponential backoff"""
    for attempt in range(max_attempts):
        try:
            return await func()
        except retryable_errors as e:
            if attempt == max_attempts - 1:
                raise

            wait_time = backoff * (2**attempt)
            logger.warning(
                f"Retryable error: {e}. Retrying in {wait_time}s (attempt {attempt + 1}/{max_attempts})"
            )
            await asyncio.sleep(wait_time)
        except Exception:
            raise


class OpenAIHarness(BaseHarness):
    """Harness for OpenAI Agents SDK"""

    def __init__(self, config: Optional[HarnessConfig] = None, api_key: Optional[str] = None):
        super().__init__(config, api_key)
        self._agent = None
        self._tool_cache = {}
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

    def _create_agent(self, config: HarnessConfig):
        """Lazily create OpenAI agent with registered tools"""
        from agents import Agent, function_tool

        # Use cached wrapped tools
        tools = []
        for tool_meta in self._get_registered_tools():
            if tool_meta.name not in self._tool_cache:
                self._tool_cache[tool_meta.name] = function_tool(tool_meta.callable)
            tools.append(self._tool_cache[tool_meta.name])

        # Build model config
        model_config = {}
        if config.model:
            model_config["model"] = config.model
        if config.temperature is not None:
            model_config["temperature"] = config.temperature
        if config.max_output_tokens:
            model_config["max_tokens"] = config.max_output_tokens
        if config.top_p is not None:
            model_config["top_p"] = config.top_p
        if config.stop_sequences:
            model_config["stop"] = config.stop_sequences

        agent = Agent(
            name="Agent",
            instructions=config.system_prompt,
            tools=tools,
            model_config=model_config if model_config else None,
        )

        return agent

    async def run(
        self, prompt: str, config_override: Optional[HarnessConfig] = None
    ) -> AgentResponse:
        """Run OpenAI agent and return final response"""
        from agents import Runner

        config = self._merge_config(config_override)
        start_time = datetime.now()

        logger.info(
            "openai.run.start",
            extra={
                "request_id": config.request_id,
                "model": config.model,
                "max_turns": config.max_turns,
            },
        )

        try:
            agent = self._create_agent(config)

            async def _run():
                return await Runner.run(
                    agent,
                    input=prompt,
                    max_turns=config.max_turns,
                )

            # Apply timeout if configured
            if config.timeout_sec:
                result = await asyncio.wait_for(_run(), timeout=config.timeout_sec)
            else:
                result = await _run()

            end_time = datetime.now()
            latency_ms = (end_time - start_time).total_seconds() * 1000

            metadata = {
                "provider": "openai",
                "agent_name": result.active_agent.name,
                "turns": len(result.messages),
                "latency_ms": latency_ms,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
            }

            logger.info(
                "openai.run.complete",
                extra={
                    "request_id": config.request_id,
                    "latency_ms": latency_ms,
                    "turns": len(result.messages),
                },
            )

            return AgentResponse(
                final_output=result.final_output,
                provider="openai",
                request_id=config.request_id,
                messages=result.messages,
                metadata=metadata,
            )

        except asyncio.TimeoutError:
            logger.error("openai.run.timeout", extra={"request_id": config.request_id})
            raise TimeoutError(
                f"OpenAI request timed out after {config.timeout_sec}s",
                "openai",
                request_id=config.request_id,
            ) from None

        except Exception as e:
            logger.error(
                "openai.run.error",
                extra={"request_id": config.request_id, "error": str(e)},
                exc_info=True,
            )
            raise ProviderError(
                f"OpenAI error: {str(e)}", "openai", request_id=config.request_id
            ) from e

    async def stream(
        self, prompt: str, config_override: Optional[HarnessConfig] = None
    ) -> AsyncIterator[str]:
        """Stream OpenAI agent responses (incremental deltas)"""
        from agents import Runner

        config = self._merge_config(config_override)
        agent = self._create_agent(config)

        logger.info("openai.stream.start", extra={"request_id": config.request_id})

        try:
            async for event in Runner.run_streamed(agent, input=prompt, max_turns=config.max_turns):
                # Yield only text deltas
                if hasattr(event, "delta") and event.delta:
                    yield event.delta
        except Exception as e:
            logger.error(
                "openai.stream.error", extra={"request_id": config.request_id, "error": str(e)}
            )
            raise ProviderError(f"OpenAI streaming error: {str(e)}", "openai") from e


class ClaudeHarness(BaseHarness):
    """Harness for Anthropic Claude Agent SDK"""

    def __init__(self, config: Optional[HarnessConfig] = None, api_key: Optional[str] = None):
        super().__init__(config, api_key)
        self._mcp_server = None
        self._tool_name_mapping = {}
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key

    def _create_mcp_server(self):
        """Create in-process MCP server with registered tools"""
        from claude_agent_sdk import create_sdk_mcp_server, tool

        tool_functions = []
        self._tool_name_mapping = {}

        for tool_meta in self._get_registered_tools():
            # Use JSON schema for input_schema
            input_schema = tool_meta.json_schema.get("properties", {})

            # Wrap the tool
            wrapped_tool = tool(tool_meta.name, tool_meta.description, input_schema)(
                tool_meta.callable
            )

            tool_functions.append(wrapped_tool)
            # Store mapping for later use
            self._tool_name_mapping[tool_meta.name] = tool_meta.name

        if tool_functions:
            self._mcp_server = create_sdk_mcp_server(
                name="harness-tools", version="1.0.0", tools=tool_functions
            )

            logger.debug(f"Created MCP server with {len(tool_functions)} tools")

    def _create_client_options(self, config: HarnessConfig):
        """Create Claude client options"""
        from claude_agent_sdk import ClaudeAgentOptions

        if not self._mcp_server and self._get_registered_tools():
            self._create_mcp_server()

        options_dict = {
            "system_prompt": config.system_prompt,
            "max_turns": config.max_turns,
        }

        if config.model:
            options_dict["model"] = config.model

        if self._mcp_server:
            options_dict["mcp_servers"] = {"harness": self._mcp_server}

            # Build allowed_tools list - try to get actual tool names
            # For now, use the naming convention but this should be dynamic
            tool_names = [f"mcp__harness__{name}" for name in self._tool_name_mapping.keys()]
            if tool_names:
                options_dict["allowed_tools"] = tool_names

        # Merge provider-specific options
        options_dict.update(config.provider_options)

        return ClaudeAgentOptions(**options_dict)

    async def run(
        self, prompt: str, config_override: Optional[HarnessConfig] = None
    ) -> AgentResponse:
        """Run Claude agent and return final response"""
        from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, query

        config = self._merge_config(config_override)
        start_time = datetime.now()

        logger.info(
            "claude.run.start",
            extra={
                "request_id": config.request_id,
                "model": config.model,
                "max_turns": config.max_turns,
            },
        )

        try:
            options = self._create_client_options(config)

            messages = []
            text_blocks = []  # Accumulate all text blocks
            final_result = None

            async def _query():
                nonlocal final_result
                async for message in query(prompt=prompt, options=options):
                    messages.append(message)

                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                text_blocks.append(block.text)
                    elif isinstance(message, ResultMessage):
                        final_result = str(message.result)

            # Apply timeout
            if config.timeout_sec:
                await asyncio.wait_for(_query(), timeout=config.timeout_sec)
            else:
                await _query()

            end_time = datetime.now()
            latency_ms = (end_time - start_time).total_seconds() * 1000

            # Prefer ResultMessage, fallback to concatenated TextBlocks
            final_output = final_result if final_result else "\n".join(text_blocks)

            metadata = {
                "provider": "claude",
                "message_count": len(messages),
                "text_blocks": len(text_blocks),
                "latency_ms": latency_ms,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
            }

            logger.info(
                "claude.run.complete",
                extra={
                    "request_id": config.request_id,
                    "latency_ms": latency_ms,
                    "message_count": len(messages),
                },
            )

            return AgentResponse(
                final_output=final_output,
                provider="claude",
                request_id=config.request_id,
                messages=messages,
                metadata=metadata,
            )

        except asyncio.TimeoutError:
            logger.error("claude.run.timeout", extra={"request_id": config.request_id})
            raise TimeoutError(
                f"Claude request timed out after {config.timeout_sec}s",
                "claude",
                request_id=config.request_id,
            ) from None

        except Exception as e:
            logger.error(
                "claude.run.error",
                extra={"request_id": config.request_id, "error": str(e)},
                exc_info=True,
            )
            raise ProviderError(
                f"Claude error: {str(e)}", "claude", request_id=config.request_id
            ) from e

    async def stream(
        self, prompt: str, config_override: Optional[HarnessConfig] = None
    ) -> AsyncIterator[str]:
        """Stream Claude agent responses (incremental deltas)"""
        from claude_agent_sdk import AssistantMessage, TextBlock, query

        config = self._merge_config(config_override)
        options = self._create_client_options(config)

        logger.info("claude.stream.start", extra={"request_id": config.request_id})

        previous_text = ""

        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            # Yield delta only (difference from previous)
                            current_text = block.text
                            if current_text.startswith(previous_text):
                                delta = current_text[len(previous_text) :]
                                if delta:
                                    yield delta
                                    previous_text = current_text
                            else:
                                # New block
                                yield current_text
                                previous_text = current_text
        except Exception as e:
            logger.error(
                "claude.stream.error", extra={"request_id": config.request_id, "error": str(e)}
            )
            raise ProviderError(f"Claude streaming error: {str(e)}", "claude") from e

    async def aclose(self) -> None:
        """Clean up MCP server"""
        await super().aclose()
        if self._mcp_server:
            # MCP server cleanup if needed
            self._mcp_server = None
            logger.debug("Closed MCP server")


class AgentHarness:
    """
    High-level harness that supports hot-swapping between providers.

    Usage:
        async with AgentHarness(provider="openai", config=config) as harness:
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
        api_key: Optional[str] = None,
    ):
        self.provider_name = provider.lower()
        self.config = config or HarnessConfig()
        self.api_key = api_key
        self._lock = asyncio.Lock()

        if self.provider_name not in self.PROVIDERS:
            raise ValueError(
                f"Unknown provider: {provider}. " f"Available: {list(self.PROVIDERS.keys())}"
            )

        self._harness = self._create_harness()

    def _create_harness(self) -> BaseHarness:
        """Create provider-specific harness"""
        harness_class = self.PROVIDERS[self.provider_name]
        return harness_class(config=self.config, api_key=self.api_key)

    async def switch_provider(self, provider: str) -> None:
        """
        Hot-swap to a different provider.

        Args:
            provider: Provider name ("openai", "claude", or "anthropic")
        """
        async with self._lock:
            provider = provider.lower()
            if provider not in self.PROVIDERS:
                raise ValueError(
                    f"Unknown provider: {provider}. " f"Available: {list(self.PROVIDERS.keys())}"
                )

            # Close old harness
            await self._harness.aclose()

            self.provider_name = provider
            self._harness = self._create_harness()

            logger.info(f"Switched to {provider} provider")

    async def run(
        self, prompt: str, config_override: Optional[HarnessConfig] = None
    ) -> AgentResponse:
        """Run agent with current provider"""
        async with self._lock:
            return await self._harness.run(prompt, config_override)

    async def stream(
        self, prompt: str, config_override: Optional[HarnessConfig] = None
    ) -> AsyncIterator[str]:
        """Stream responses with current provider"""
        # Don't lock the entire stream, just the harness access
        harness = self._harness
        async for chunk in harness.stream(prompt, config_override):
            yield chunk

    async def compare_providers(
        self,
        prompt: str,
        providers: Optional[list[str]] = None,
        config_override: Optional[HarnessConfig] = None,
    ) -> dict[str, AgentResponse]:
        """
        Run the same prompt on multiple providers in parallel.

        Args:
            prompt: User prompt
            providers: List of provider names (defaults to all)
            config_override: Optional config override

        Returns:
            Dictionary mapping provider name to response
        """
        if providers is None:
            providers = ["openai", "claude"]

        config = self.config if not config_override else config_override

        async def run_provider(provider: str) -> tuple[str, AgentResponse]:
            try:
                # Create separate harness instance
                harness_class = self.PROVIDERS[provider]
                harness = harness_class(config=config, api_key=self.api_key)

                async with harness:
                    response = await harness.run(prompt)
                    return provider, response
            except Exception as e:
                logger.error(f"Error running {provider}: {e}")
                return provider, AgentResponse(
                    final_output=f"Error: {str(e)}",
                    provider=provider,
                    request_id=config.request_id,
                    metadata={"error": True, "exception": str(e)},
                )

        # Run in parallel
        results = await asyncio.gather(*[run_provider(p) for p in providers])
        return dict(results)

    async def aclose(self) -> None:
        """Clean up resources"""
        await self._harness.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()
