# Design Doc: CacheManager as a Planner

**Author:** Sean Brar
**Date:** July 12, 2025
**Status:** Proposed
**Context:** GSoC 2025 - Architectural proposal for improving cache management
**Current Architecture Baseline:** Week 6 implementation (commit 4355aa6)

> **Note**: This document proposes a refactoring of the cache management system
> developed during weeks 4-6 of the GSoC project. The "current architecture"
> refers to the wrapper-based approach implemented in the Week 6 codebase.

---

## 1. Abstract

The current architecture successfully decouples caching logic from the `GeminiClient` by using a "Wrapper" pattern, where `CacheManager.generate_with_cache` wraps the client's raw generation function. This is a significant improvement.

This document proposes an evolution of that design to a **Planner-Executor** pattern. In this model, the `CacheManager`'s role is simplified further: it transitions from an active orchestrator to a stateless **Planner**. It analyzes a request and returns a simple, descriptive "action plan." The `GeminiClient` then acts as an **Executor**, reading the plan and performing the corresponding action.

This change aims for radical simplicity, making the control flow flatter, more readable, and significantly easier to test and maintain.

## 2. The "Advise, Don't Act" Philosophy

The core principle of the Planner-Executor pattern is a strict separation of decision-making from action-taking.

* **The Planner (`CacheManager`):** Its sole responsibility is to make decisions. It inspects the content and context of a request and produces a recommendation. It answers the question, *"What should be done?"* It does not perform the action itself.

* **The Executor (`GeminiClient`):** Its responsibility is to act. It receives the plan from the Planner and executes the specified action. It answers the question, *"How is it done?"* It trusts the Planner's decision and does not need to know the reasoning behind it.

This pattern eliminates the need for passing function callbacks (`raw_generation_func`) and removes complex, nested control flows, replacing them with a simple, linear sequence: **Plan -> Execute**.

## 3. Key Components

### 3.1. The `CacheAction` Data Transfer Object

The centerpiece of this pattern is a simple, immutable data class that represents the "plan." It serves as the contract between the Planner and the Executor.

```python
# In gemini_batch/client/models.py (a new proposed file for shared data classes)
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List
from google.genai import types

class CacheStrategy(Enum):
    GENERATE_RAW = auto()
    GENERATE_WITH_OPTIMIZED_PARTS = auto() # Implicit Caching
    GENERATE_FROM_EXPLICIT_CACHE = auto()

@dataclass(frozen=True)
class CacheAction:
    """An immutable plan describing the caching action to be taken."""
    strategy: CacheStrategy
    payload: Dict[str, Any] = field(default_factory=dict)
```

* **`strategy`**: An enum that explicitly states which action to take. This is far more readable and type-safe than relying on string literals.
* **`payload`**: A dictionary containing all the necessary data for the Executor to perform the action (e.g., `cache_name`, optimized `parts`).

### 3.2. The `CacheManager` (The Planner)

The `CacheManager` is refactored to contain one primary public method: `plan_generation`.

* **Responsibility**: Analyzes content, determines the caching strategy, manages the state of the cache objects (creating/retrieving them via the API), and returns a `CacheAction`.
* **Key Characteristic**: It is **stateless** regarding the generation request itself. It does not execute generation calls, handle retries, or process responses. Its only side effects are interactions with the Gemini Caches API (`client.caches.create`, `client.caches.get`).

```python
# In gemini_batch/client/cache_manager.py
class CacheManager:
    # ... __init__ and other cache state management methods ...

    def plan_generation(self, parts: List[types.Part]) -> CacheAction:
        """Analyzes content and returns a plan for the client to execute."""
        token_count = self.token_counter.count_tokens(...)
        cache_analysis = self.config_manager.can_use_caching(...)

        if not cache_analysis.get("supported"):
            return CacheAction(strategy=CacheStrategy.GENERATE_RAW, payload={'parts': parts})

        strategy_name = cache_analysis["recommendation"]

        if strategy_name == "explicit":
            # Perform cache creation/retrieval here (the Planner's only side effect)
            cached_content_obj = self.get_or_create(...)
            _, prompt_parts = self.content_processor.separate_cacheable_content(parts)
            return CacheAction(
                strategy=CacheStrategy.GENERATE_FROM_EXPLICIT_CACHE,
                payload={'cache_name': cached_content_obj.name, 'parts': prompt_parts}
            )

        elif strategy_name == "implicit":
            optimized_parts = self.content_processor.optimize_for_implicit_cache(parts)
            return CacheAction(
                strategy=CacheStrategy.GENERATE_WITH_OPTIMIZED_PARTS,
                payload={'parts': optimized_parts}
            )

        # Default fallback
        return CacheAction(strategy=CacheStrategy.GENERATE_RAW, payload={'parts': parts})
```

### 3.3. The `GeminiClient` (The Executor)

The `GeminiClient._execute_generation` method becomes radically simplified. Its logic is reduced to a simple dispatch table based on the `CacheAction.strategy`.

```python
# In gemini_batch/gemini_client.py
class GeminiClient:
    def _execute_generation(self, ...):
        parts = self.content_processor.process(content)

        # 1. Get the plan from the Planner
        action = self.cache_manager.plan_generation(parts) if self.cache_manager else \
                 CacheAction(strategy=CacheStrategy.GENERATE_RAW, payload={'parts': parts})

        # 2. Execute the plan
        raw_response = None
        if action.strategy == CacheStrategy.GENERATE_RAW:
            raw_response = self._generate_with_parts(action.payload['parts'], ...)
        
        elif action.strategy == CacheStrategy.GENERATE_WITH_OPTIMIZED_PARTS:
            raw_response = self._generate_with_parts(action.payload['parts'], ...)

        elif action.strategy == CacheStrategy.GENERATE_FROM_EXPLICIT_CACHE:
            raw_response = self._generate_with_cache_reference(
                cache_name=action.payload['cache_name'],
                prompt_parts=action.payload['parts'],
                ...
            )
        
        # 3. Process the response (responsibility remains with the client)
        return self._process_response(raw_response, ...)
```

## 4. Architectural Benefits

This pattern yields significant improvements in key areas:

1. **Radical Simplicity & Readability**: The control flow in `GeminiClient` is no longer nested or callback-driven. It is a flat, descriptive `if/elif` block. The logic is immediately understandable: get a plan, execute the plan.

2. **Superior Testability**:
    * **Testing the Planner (`CacheManager`)**: Becomes trivial. It's a nearly pure function. You provide input (`parts`) and assert that the correct `CacheAction` is returned. No mocking of API calls or generation functions is required.
    * **Testing the Executor (`GeminiClient`)**: Becomes declarative. You can mock `cache_manager.plan_generation` to return a specific `CacheAction` and assert that the correct internal generation method (`_generate_with_parts`, `_generate_with_cache_reference`) is called with the correct payload.

3. **Maximum Decoupling**: The components are now more decoupled than ever.
    * `CacheManager` knows nothing about retries, response parsing, or error handling.
    * `GeminiClient` knows nothing about the *logic* of TTL calculation, content hashing, or token thresholds for caching. It only needs to know how to act on the final decision.

4. **Maintainability & Extensibility**: Adding a new caching strategy is a clean, localized process:
    1. Add a new value to the `CacheStrategy` enum.
    2. Add the logic to `CacheManager.plan_generation` to return the new `CacheAction`.
    3. Add an `elif` block to `GeminiClient._execute_generation` to handle the new strategy.
    The existing components and logic paths remain untouched, minimizing the risk of regression.

## 5. Conclusion

The Planner-Executor pattern represents a mature architectural choice that prioritizes clarity, testability, and long-term maintainability. By transforming the `CacheManager` from an active wrapper into a passive advisor, we create a system where each component has a single, well-defined responsibility. This radically simple approach results in code that is easier to reason about, safer to modify, and more robust in its execution. It is the recommended architectural direction for the `CacheManager` and `GeminiClient` interaction.

## Revision History

**Revision 1** (July 12, 2025)

* Initial proposal of Planner-Executor pattern
* Basic `CacheAction` structure with generic payload
