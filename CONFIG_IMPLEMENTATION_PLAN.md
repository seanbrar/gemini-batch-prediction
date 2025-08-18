# Configuration System Implementation Plan

> **Based on:** ADR-0007, Configuration Concepts, Deep-Dive Spec, and Architecture Rubric
> **Goal:** Replace dict-based ambient config with resolve-once, freeze-then-flow immutable system

---

## Implementation Task Breakdown

### Phase 1: Core Data & Schema Layer (Tasks 1-3)
**Focus:** Foundational data structures and validation

- [ ] **Task 1:** Create core data structures: ResolvedConfig and FrozenConfig classes with proper type safety and validation
- [ ] **Task 2:** Implement settings schema with Pydantic integration for field validation, defaults, and type coercion
- [ ] **Task 3:** Create SourceMap system for audit-grade origin tracking with secret redaction capabilities

**Commit Point 1:** Core data structures with basic validation

### Phase 2: Resolution Engine (Tasks 4-8)
**Focus:** Multi-source configuration resolution with precedence

- [ ] **Task 4:** Implement file resolution layer: TOML parser for pyproject.toml and home config with profile support
- [ ] **Task 5:** Build environment variable resolution with GEMINI_* prefix handling and type coercion
- [ ] **Task 6:** Create precedence resolution algorithm: programmatic > env > project > home > defaults
- [ ] **Task 7:** Implement profile selection system with GEMINI_PROFILE support and error handling
- [ ] **Task 8:** Build main resolve_config() function with all source integration and validation

**Commit Point 2:** Complete resolution engine with all sources

### Phase 3: Integration & Extension (Tasks 9-12)
**Focus:** Pipeline integration and observability

- [ ] **Task 9:** Update config_scope() to work with new ResolvedConfig and entry-time-only resolution
- [ ] **Task 10:** Create provider config registry system for optional vendor-specific config views
- [ ] **Task 11:** Implement telemetry and observability: metrics, audit logging, and redacted output
- [ ] **Task 12:** Add validation rules and invariant checking (api_key requirement, ttl_seconds >= 1, etc)

**Commit Point 3:** Full system with observability and validation

### Phase 4: Pipeline Integration (Tasks 13-14)
**Focus:** Command pipeline integration with compatibility

- [ ] **Task 13:** Update InitialCommand and pipeline entry points to use FrozenConfig instead of dict
- [ ] **Task 14:** Create compatibility shim for dict-based config access during migration period

**Commit Point 4:** Pipeline integration with backward compatibility

### Phase 5: Testing & Quality (Tasks 15-17)
**Focus:** Comprehensive test coverage and security

- [ ] **Task 15:** Build comprehensive unit tests for precedence, validation, profiles, and edge cases
- [ ] **Task 16:** Add integration tests for pipeline flow with frozen config and handler consumption
- [ ] **Task 17:** Implement security tests ensuring secret redaction and no leakage in logs/audits

**Commit Point 5:** Full test coverage and security validation

### Phase 6: Migration & Documentation (Tasks 18-20)
**Focus:** Handler migration and tooling

- [ ] **Task 18:** Update all pipeline handlers to consume FrozenConfig attributes instead of dict access
- [ ] **Task 19:** Add CLI command for config introspection: print effective config and check validation
- [ ] **Task 20:** Update documentation and examples to reflect new configuration patterns and migration guide

**Final Commit:** Complete migration to new configuration system

---

## Architecture Rubric Alignment

This implementation prioritizes:

### Simplicity & Elegance (5/5)
- Single resolution moment eliminates complexity
- Clear data flow: sources → merge → freeze → pipeline
- Advanced types eliminate conceptual overhead

### Data-Centricity (5/5)
- Rich, self-validating data structures (ResolvedConfig, FrozenConfig)
- Pure transformation functions for resolution
- Flow from one valid state to another

### Clarity & Explicitness (5/5)
- No hidden state or action-at-a-distance
- Transparent precedence algorithm
- Explicit origin tracking via SourceMap

### Architectural Robustness (5/5)
- Invalid states structurally impossible via frozen dataclasses
- Type system guarantees correctness
- Early validation prevents runtime errors

### Extensibility (5/5)
- Clear provider seam via registry
- Schema-driven field additions
- Core remains untouched for new features

---

## Key Design Principles

1. **Resolve once, freeze then flow:** Configuration resolved before pipeline entry
2. **Immutable commands:** FrozenConfig attached to InitialCommand
3. **Predictable precedence:** Programmatic > Env > Project > Home > Defaults
4. **Audit-grade observability:** SourceMap with secret redaction
5. **Provider neutrality:** Core stays neutral, extensions at adapter boundary

---

## Dependencies Required

- `pydantic` or `pydantic-settings` for typed schema validation
- `tomli` or `tomllib` (stdlib Py 3.11+) for TOML parsing
- Existing: `contextvars` (stdlib) for async-safe scoping

---

## Migration Strategy

1. **Introduce new system alongside old** (compatibility shim)
2. **Migrate pipeline entry points** to use FrozenConfig
3. **Update handlers** from dict access to attribute access
4. **Remove dict compatibility** and old ambient resolution
5. **Document deprecations** and provide migration tools

---

## Success Criteria

- [ ] All configuration resolved before pipeline entry
- [ ] No handler performs configuration I/O during execution
- [ ] Secrets never appear in logs or audit output
- [ ] Clear error messages for missing required configuration
- [ ] Profiles work in both project and home files
- [ ] SourceMap explains every effective value
- [ ] Provider registry enables clean vendor-specific views
- [ ] Full test coverage with security validation
- [ ] Migration completed without breaking changes
