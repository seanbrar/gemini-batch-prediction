from gemini_batch.core.types import APICall, ExecutionPlan, RateConstraint


def test_execution_plan_can_carry_rate_constraint():
    plan = ExecutionPlan(
        calls=(APICall(model_name="gemini-2.0-flash", api_parts=(), api_config={}),),
        rate_constraint=RateConstraint(requests_per_minute=60, tokens_per_minute=None),
    )
    assert isinstance(plan.rate_constraint, RateConstraint | type(None))
