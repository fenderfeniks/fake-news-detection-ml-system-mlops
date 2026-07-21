import importlib


def _get_schedule(dag):
    return getattr(dag, "schedule", None) or getattr(dag, "schedule_interval", None)


def test_batch_analytics_uses_real_config():
    import dags.batch_analytics as mod

    task = mod.dag.get_task("run_batch_inference")
    limits = task.container_resources.limits
    assert limits["cpu"] == "1", (
        f"Применился DEFAULT_CONFIG вместо variables.json. cpu limits = {limits['cpu']!r}"
    )
    assert limits["memory"] == "2Gi", f"Неверный memory limit: {limits['memory']!r}"


def test_retrain_uses_real_config():
    import dags.retrain_model_dag as mod

    task = mod.dag.get_task("run_lora_finetuning")
    limits = task.container_resources.limits
    assert limits["cpu"] == "1", f"DEFAULT_CONFIG перебил variables.json. cpu = {limits['cpu']!r}"


def test_maintenance_schedule_from_variables():
    import dags.system_maintenance as mod

    actual = str(_get_schedule(mod.dag))
    assert actual == "0 3 * * 0", (
        f"Расписание взято из DEFAULT_CONFIG (@daily). Реальное: {actual!r}"
    )


def test_quality_control_drift_threshold_from_variables():
    import dags.quality_control as mod

    task = mod.dag.get_task("evaluate_model")
    args = task.arguments
    assert any("0.9" in str(a) for a in args), (
        f"drift_threshold из variables.json (0.9) не попал в аргументы. Аргументы: {args}"
    )
