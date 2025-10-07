# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-01-06

### Added
- **JSON Schema mapping** for tools with full type support (str, int, float, bool, list, dict, Optional, Union)
- **Custom error taxonomy** (HarnessError, ProviderError, ToolError, SchemaError, RateLimitError, TimeoutError)
- **Structured logging** with request IDs and timing metadata
- **Enhanced HarnessConfig** with validation and new fields:
  - `timeout_sec` - Request timeout
  - `max_output_tokens` - Maximum output tokens
  - `top_p` - Top-p sampling
  - `stop_sequences` - Stop sequences
  - `retry_attempts` - Number of retry attempts
  - `retry_backoff` - Backoff multiplier for retries
  - `request_id` - Unique request identifier
- **Resource management** with `aclose()` and async context manager support
- **Registry locking** for thread-safe concurrent access
- **Comprehensive test suite** with unit tests for registry, config, and errors
- **Packaging** with pyproject.toml and optional dependencies
- **CI/CD** with GitHub Actions (linting, type checking, testing)
- **Per-call config overrides** via `run(prompt, config_override=...)`
- **Parallel provider comparison** in `compare_providers()`

### Changed
- **Fixed streaming consistency**: OpenAI yields deltas, Claude now computes deltas from full blocks
- **Fixed Claude message aggregation**: Accumulates all TextBlocks instead of overwriting
- **Improved tool name resolution**: Dynamic mapping instead of hardcoded prefixes
- **Enhanced AgentResponse** with `provider`, `request_id`, `latency_ms` metadata
- **Better error handling** with provider-specific error wrapping
- **Async-first API** with proper resource cleanup

### Fixed
- Tool registration now generates correct JSON Schema for MCP
- Claude harness properly aggregates multiple TextBlocks
- Streaming no longer duplicates content from Claude
- Thread-safe registry operations
- Temperature only set when explicitly provided

## [0.1.0] - 2025-01-05

### Added
- Initial release
- Unified tool registry with `@register_tool` decorator
- `BaseHarness` abstract class
- `OpenAIHarness` and `ClaudeHarness` implementations
- `AgentHarness` with hot-swapping capability
- Basic configuration and response types
- Example usage scripts
- Documentation
