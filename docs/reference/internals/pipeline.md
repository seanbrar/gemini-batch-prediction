# Pipeline Internals

Core pipeline handlers and supporting utilities. These are stable but considered lower-level than the public helpers.

## Handler Protocol

::: gemini_batch.pipeline.base.BaseAsyncHandler

## Handlers

::: gemini_batch.pipeline.source_handler.SourceHandler

::: gemini_batch.pipeline.planner.ExecutionPlanner

::: gemini_batch.pipeline.remote_materialization.RemoteMaterializationStage

::: gemini_batch.pipeline.cache_stage.CacheStage

::: gemini_batch.pipeline.api_handler.APIHandler

::: gemini_batch.pipeline.rate_limit_handler.RateLimitHandler

::: gemini_batch.pipeline.result_builder.ResultBuilder

## Execution State and Identity

::: gemini_batch.pipeline.execution_state.ExecutionHints

::: gemini_batch.pipeline.cache_identity.det_shared_key

## Type Erasure (Internal)

::: gemini_batch.pipeline._erasure.ErasedAsyncHandler
