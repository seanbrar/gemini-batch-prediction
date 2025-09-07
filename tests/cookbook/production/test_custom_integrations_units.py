from __future__ import annotations

from tests.cookbook.support import load_recipe_module


def test_print_reporter_methods_emit_output(capsys):
    mod = load_recipe_module("cookbook/production/custom-integrations.py")
    rep = mod.PrintReporter()
    rep.record_timing("scope.name", 0.123, depth=2)
    rep.record_metric("metric.name", 42, parent_scope="root")
    out = capsys.readouterr().out
    assert "TIMING scope.name" in out and "0.1230s" in out
    assert "METRIC metric.name" in out and "root" in out
