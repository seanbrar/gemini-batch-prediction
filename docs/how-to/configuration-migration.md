# Configuration System Migration Guide

This guide covers migrating from the legacy dict-based configuration to the new resolve-once, freeze-then-flow system.

## Quick Migration

### Before (Legacy)

```python
from gemini_batch.config import get_ambient_config, config_scope

# Dict-based access
config = get_ambient_config()
model = config["model"]
api_key = config.get("api_key")
```

### After (New System)

```python
from gemini_batch.config import resolve_config

# Type-safe access
resolved = resolve_config()
frozen = resolved.to_frozen()
model = frozen.model
api_key = frozen.api_key
```

### Compatibility Shim (During Migration)

```python
from gemini_batch.config.compatibility import ConfigCompatibilityShim

# Works with both FrozenConfig and dict
shim = ConfigCompatibilityShim(config)
model = shim.model  # Unified access
api_key = shim.api_key
```

## Configuration Introspection

Debug configuration issues:

```bash
# Check if config is valid
python -m gemini_batch.config --check

# Show detailed config info
python -m gemini_batch.config

# JSON output for scripts
python -m gemini_batch.config --json
```

## Key Changes

1. **Immutable Configuration**: Config objects are frozen after resolution
2. **Explicit Sources**: Track where each config value came from
3. **Type Safety**: Strong typing with Pydantic validation
4. **Security**: API keys are redacted in string representations
5. **Precedence**: Programmatic > Environment > Project > Home > Defaults

## Migration Checklist

- [ ] Replace `get_ambient_config()` with `resolve_config()`
- [ ] Update dict access (`config["key"]`) to attribute access (`config.key`)
- [ ] Use `ConfigCompatibilityShim` for gradual migration
- [ ] Test with configuration introspection tools
- [ ] Remove legacy `config_scope` usage

## Environment Variables

The new system uses the same environment variables:

- `GEMINI_API_KEY` - Your API key
- `GEMINI_MODEL` - Model name (default: gemini-2.0-flash)
- `GEMINI_TIER` - API tier (free, tier_1, tier_2, tier_3)
- `GEMINI_ENABLE_CACHING` - Enable caching (true/false)
- `GEMINI_TTL_SECONDS` - Cache TTL in seconds

For more details, see the configuration documentation.
