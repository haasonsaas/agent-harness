"""
Example usage of the unified Agent Harness

Demonstrates hot-swapping between OpenAI and Anthropic providers
with a shared tool registry.
"""

import asyncio
from agent_harness import (
    AgentHarness,
    HarnessConfig,
    register_tool,
    get_registry
)


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
    """Basic example: same prompt on both providers"""
    print("=" * 60)
    print("BASIC EXAMPLE: Hot-swapping providers")
    print("=" * 60 + "\n")
    
    config = HarnessConfig(
        system_prompt="You are a helpful assistant. Be concise.",
        max_turns=5
    )
    
    harness = AgentHarness(provider="openai", config=config)
    
    prompt = "What is 2 + 2?"
    
    print(f"Prompt: {prompt}\n")
    
    print("🤖 OpenAI Response:")
    result = await harness.run(prompt)
    print(f"   {result.final_output}")
    print(f"   Metadata: {result.metadata}\n")
    
    harness.switch_provider("claude")
    
    print("🤖 Claude Response:")
    result = await harness.run(prompt)
    print(f"   {result.final_output}")
    print(f"   Metadata: {result.metadata}\n")


async def tool_example():
    """Example with tools from the registry"""
    print("=" * 60)
    print("TOOL EXAMPLE: Using registered tools")
    print("=" * 60 + "\n")
    
    config = HarnessConfig(
        system_prompt="You are a helpful assistant. Use tools when appropriate.",
        tool_names=["get_weather", "add", "multiply"],
        max_turns=10
    )
    
    prompts = [
        "What's the weather in Tokyo?",
        "What is 15 multiplied by 7?",
    ]
    
    for provider in ["openai", "claude"]:
        print(f"\n{'─' * 60}")
        print(f"Provider: {provider.upper()}")
        print('─' * 60)
        
        harness = AgentHarness(provider=provider, config=config)
        
        for prompt in prompts:
            print(f"\nPrompt: {prompt}")
            result = await harness.run(prompt)
            print(f"Response: {result.final_output}")


async def streaming_example():
    """Example with streaming responses"""
    print("\n" + "=" * 60)
    print("STREAMING EXAMPLE: Real-time responses")
    print("=" * 60 + "\n")
    
    config = HarnessConfig(
        system_prompt="You are a creative writer.",
        max_turns=3
    )
    
    harness = AgentHarness(provider="openai", config=config)
    
    prompt = "Write a haiku about coding"
    print(f"Prompt: {prompt}\n")
    print("Streaming response: ", end="", flush=True)
    
    async for chunk in harness.stream(prompt):
        print(chunk, end="", flush=True)
    
    print("\n")


async def comparison_example():
    """Compare both providers side-by-side"""
    print("=" * 60)
    print("COMPARISON EXAMPLE: Side-by-side provider comparison")
    print("=" * 60 + "\n")
    
    config = HarnessConfig(
        system_prompt="You are a data analyst. Be concise and insightful.",
        max_turns=5
    )
    
    harness = AgentHarness(provider="openai", config=config)
    
    prompt = "What are the main benefits of asynchronous programming?"
    print(f"Prompt: {prompt}\n")
    
    results = await harness.compare_providers(prompt)
    
    for provider, response in results.items():
        print(f"\n{'─' * 60}")
        print(f"{provider.upper()} Response:")
        print('─' * 60)
        print(response.final_output)
        if not response.metadata.get("error"):
            print(f"\nMetadata: {response.metadata}")


async def dynamic_tool_registration():
    """Demonstrate dynamic tool registration"""
    print("\n" + "=" * 60)
    print("DYNAMIC REGISTRATION: Adding tools at runtime")
    print("=" * 60 + "\n")
    
    @register_tool(description="Convert Celsius to Fahrenheit")
    def celsius_to_fahrenheit(celsius: float) -> float:
        """Convert temperature from Celsius to Fahrenheit"""
        return (celsius * 9/5) + 32
    
    print(f"Registered tools: {list(get_registry().get_all().keys())}\n")
    
    config = HarnessConfig(
        system_prompt="You are a helpful assistant.",
        tool_names=["celsius_to_fahrenheit"],
        max_turns=5
    )
    
    harness = AgentHarness(provider="openai", config=config)
    
    prompt = "What is 25 degrees Celsius in Fahrenheit?"
    print(f"Prompt: {prompt}")
    
    result = await harness.run(prompt)
    print(f"Response: {result.final_output}\n")


async def provider_specific_options():
    """Example with provider-specific options"""
    print("=" * 60)
    print("PROVIDER-SPECIFIC OPTIONS")
    print("=" * 60 + "\n")
    
    config = HarnessConfig(
        system_prompt="You are a helpful coding assistant.",
        max_turns=5,
        provider_options={
            "permission_mode": "acceptEdits",
            "allowed_tools": ["Read", "Bash"]
        }
    )
    
    harness = AgentHarness(provider="claude", config=config)
    
    prompt = "What files are in the current directory?"
    print(f"Prompt: {prompt}")
    
    result = await harness.run(prompt)
    print(f"Response: {result.final_output}\n")


async def main():
    """Run all examples"""
    
    # Show registered tools
    print("\n📋 Registered Tools:")
    for name, tool in get_registry().get_all().items():
        print(f"   • {name}: {tool.description}")
    print()
    
    # Run examples
    await basic_example()
    await tool_example()
    
    # Uncomment to run additional examples:
    # await streaming_example()
    # await comparison_example()
    # await dynamic_tool_registration()
    # await provider_specific_options()


if __name__ == "__main__":
    # Ensure API keys are set:
    # export OPENAI_API_KEY="your-key-here"
    # export ANTHROPIC_API_KEY="your-key-here"
    
    asyncio.run(main())
