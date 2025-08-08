import inspect

import pytest

from gemini_batch.core import exceptions, models, types


class TestSimplicityCompliance:
    """Tests that verify the core module maintains architectural simplicity."""

    @pytest.mark.unit
    @pytest.mark.contract
    def test_core_module_has_minimal_public_interface(self):
        """The core module should have a minimal, focused public interface."""
        # Check that core module doesn't expose too many public items
        core_public_items = [
            name
            for name in dir(types)
            if not name.startswith("_") and not inspect.ismodule(getattr(types, name))
        ]

        # Should have a reasonable number of public items (not too many)
        # Allow for 25 items to account for type imports and core functionality
        assert len(core_public_items) <= 25, (
            f"Too many public items: {core_public_items}"
        )

    @pytest.mark.unit
    @pytest.mark.contract
    def test_dataclasses_have_simple_constructors(self):
        """Dataclasses should have simple, obvious constructors."""
        dataclasses = [
            types.Success,
            types.Failure,
            types.ConversationTurn,
            types.Source,
            types.InitialCommand,
            types.ResolvedCommand,
            models.RateLimits,
            models.CachingCapabilities,
            models.ModelCapabilities,
        ]

        for cls in dataclasses:
            sig = inspect.signature(cls.__init__)

            # Should have reasonable number of parameters (not too complex)
            param_count = len(sig.parameters) - 1  # Exclude self
            assert param_count <= 6, (
                f"{cls.__name__} has too many parameters: {param_count}"
            )

    @pytest.mark.unit
    @pytest.mark.contract
    def test_functions_have_simple_signatures(self):
        """Functions should have simple, clear signatures."""
        functions = [
            models.get_model_capabilities,
            models.get_rate_limits,
            models.can_use_caching,
        ]

        for func in functions:
            sig = inspect.signature(func)

            # Should have reasonable number of parameters
            param_count = len(sig.parameters)
            assert param_count <= 3, (
                f"{func.__name__} has too many parameters: {param_count}"
            )

    @pytest.mark.unit
    @pytest.mark.contract
    def test_no_complex_inheritance_hierarchies(self):
        """The core module should avoid complex inheritance hierarchies."""
        # Check that exceptions have simple inheritance
        exception_classes = [
            exceptions.GeminiBatchError,
            exceptions.APIError,
            exceptions.PipelineError,
            exceptions.ConfigurationError,
            exceptions.SourceError,
            exceptions.MissingKeyError,
            exceptions.NetworkError,
            exceptions.FileError,
            exceptions.ValidationError,
            exceptions.UnsupportedContentError,
        ]

        for cls in exception_classes:
            # Should have simple inheritance (max 4 levels including object)
            # This accounts for: Class -> GeminiBatchError -> Exception -> BaseException -> object
            mro = cls.__mro__
            assert len(mro) <= 5, f"{cls.__name__} has complex inheritance: {mro}"
