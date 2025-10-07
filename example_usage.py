"""
Example usage of the unified Agent Harness

Demonstrates hot-swapping, tools, streaming, error handling, and more.
"""

import asyncio
import logging
from agent_harness import (
    AgentHarness,
    HarnessConfig,
    register_tool,
    get_registry,
    ProviderError,
    TimeoutError
)

# Configure logging to see structured logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Register tools once - they work with both providers
@register_tool(description="Get the current weather for a city")
def get_weather(city: str) -> str:
    """Get weather information for a given city"""
    weather_data = {
        "San Francisco": "sunny, 72°F",
        "New York": "cloudy, 65°F",
        "London": "rainy, 58°F",
        "Tokyo": "clear, 68°F",
    }
    return f"The weather in {city} is {weather_data.get(city, 'unknown')}."


@register_tool(description="Calculate the sum of two numbers")
def add(a: float, b: float) -> float:
    """Add two numbers together"""
    return a + b


@register_tool(description="Calculate the product of two numbers")
def multiply(a: float, b: float) -> float:
    """Multiply two numbers"""
    return a * b


async def basic_example():
    """Basic example with context manager and hot-swapping"""
    print("=" * 60)
    print("BASIC EXAMPLE: Hot-swapping with async context manager")
    print("=" * 60 + "\n")
    
    config = HarnessConfig(
        system_prompt="You are a helpful assistant. Be concise.",
        max_turns=5,
        timeout_sec=30.0
    )
    
    prompt = "What is 2 + 2?"
    
    # Use async context manager for automatic cleanup
    async with AgentHarness(provider="openai", config=config) as harness:
        print(f"Prompt: {prompt}\n")
        
        result = await harness.run(prompt)
        print(f"🤖 OpenAI Response: {result.final_output}")
        print(f"   Request ID: {result.request_id}")
        print(f"   Latency: {result.latency_ms:.2f}ms\n")
        
        # Hot-swap to Claude
        await harness.switch_provider("claude")
        
        result = await harness.run(prompt)
        print(f"🤖 Claude Response: {result.final_output}")
        print(f"   Request ID: {result.request_id}")
        print(f"   Latency: {result.latency_ms:.2f}ms\n")


async def tool_example():
    """Example with registered tools"""
    print("=" * 60)
    print("TOOL EXAMPLE: Using registered tools")
    print("=" * 60 + "\n")
    
    print(f"📋 Registered tools: {list(get_registry().get_all().keys())}\n")
    
    config = HarnessConfig(
        system_prompt="You are a helpful assistant. Use tools when appropriate.",
        tool_names=["get_weather", "add", "multiply"],
        max_turns=10,
        timeout_sec=30.0
    )
    
    prompts = [
        "What's the weather in Tokyo?",
        "What is 15 multiplied by 7?",
    ]
    
    async with AgentHarness(provider="openai", config=config) as harness:
        for prompt in prompts:
            print(f"Prompt: {prompt}")
            result = await harness.run(prompt)
            print(f"Response: {result.final_output}\n")


async def streaming_example():
    """Example with streaming responses"""
    print("=" * 60)
    print("STREAMING EXAMPLE: Real-time incremental deltas")
    print("=" * 60 + "\n")
    
    config = HarnessConfig(
        system_prompt="You are a creative writer.",
        max_turns=3
    )
    
    async with AgentHarness(provider="openai", config=config) as harness:
        prompt = "Write a haiku about coding"
        print(f"Prompt: {prompt}\n")
        print("Streaming: ", end="", flush=True)
        
        async for delta in harness.stream(prompt):
            print(delta, end="", flush=True)
        
        print("\n")


async def comparison_example():
    """Compare both providers side-by-side in parallel"""
    print("=" * 60)
    print("COMPARISON EXAMPLE: Parallel provider execution")
    print("=" * 60 + "\n")
    
    config = HarnessConfig(
        system_prompt="You are a data analyst. Be concise.",
        max_turns=5,
        timeout_sec=30.0
    )
    
    prompt = "What are the benefits of async programming?"
    print(f"Prompt: {prompt}\n")
    
    async with AgentHarness(provider="openai", config=config) as harness:
        # This runs both providers in parallel
        results = await harness.compare_providers(prompt)
        
        for provider, response in results.items():
            print(f"\n{'─' * 60}")
            print(f"{provider.upper()} Response:")
            print('─' * 60)
            print(response.final_output)
            if not response.error:
                print(f"\n📊 Latency: {response.latency_ms:.2f}ms")


async def error_handling_example():
    """Example with error handling and retries"""
    print("=" * 60)
    print("ERROR HANDLING EXAMPLE")
    print("=" * 60 + "\n")
    
    config = HarnessConfig(
        system_prompt="You are a helpful assistant.",
        max_turns=2,
        timeout_sec=0.001,  # Very short timeout to trigger error
        retry_attempts=2
    )
    
    async with AgentHarness(provider="openai", config=config) as harness:
        try:
            result = await harness.run("Tell me a long story")
            print(f"Response: {result.final_output}")
        except TimeoutError as e:
            print(f"⏱️  Timeout Error: {e}")
            print(f"   Provider: {e.provider}")
            print(f"   Retryable: {e.retryable}")
        except ProviderError as e:
            print(f"❌ Provider Error: {e}")
            print(f"   Provider: {e.provider}")


async def config_override_example():
    """Example with per-call config overrides"""
    print("\n" + "=" * 60)
    print("CONFIG OVERRIDE EXAMPLE")
    print("=" * 60 + "\n")
    
    base_config = HarnessConfig(
        system_prompt="You are a helpful assistant.",
        temperature=0.7,
        max_turns=5
    )
    
    async with AgentHarness(provider="openai", config=base_config) as harness:
        # First call with base config
        result1 = await harness.run("Count to 3")
        print(f"With base config (temp=0.7): {result1.final_output}\n")
        
        # Override for a single call
        override = HarnessConfig(
            temperature=0.0,  # Deterministic
            max_turns=3
        )
        result2 = await harness.run("Count to 3", config_override=override)
        print(f"With override (temp=0.0): {result2.final_output}\n")


async def main():
    """Run all examples"""
    
    print("\n🚀 Agent Harness Examples\n")
    
    try:
        await basic_example()
        await tool_example()
        # await streaming_example()  # Uncomment to test
        # await comparison_example()  # Uncomment to test
        # await error_handling_example()  # Uncomment to test
        # await config_override_example()  # Uncomment to test
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Ensure API keys are set:
    # export OPENAI_API_KEY="sk-..."
    # export ANTHROPIC_API_KEY="sk-ant-..."
    
    asyncio.run(main())
