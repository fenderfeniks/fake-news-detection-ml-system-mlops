import importlib

import pytest


def _get_schedule(dag):
    """Совместимо с Airflow 2.4+ где schedule_interval убран."""
    return getattr(dag, "schedule", None) or getattr(dag, "schedule_interval", None)


@pytest.mark.parametrize(
    "module,expected",
    [
        (
            "dags.retrain_model_dag",
            {
                "dag_id": "weekly_classifier_finetuning",
                "schedule": "@weekly",
                "task_order": [
                    ("run_lora_finetuning", "evaluate_staging_model"),
                    ("evaluate_staging_model", "request_manual_approval"),
                ],
                "all_tasks": {
                    "run_lora_finetuning",
                    "evaluate_staging_model",
                    "request_manual_approval",
                },
            },
        ),
        (
            "dags.promote_to_prod",
            {
                "dag_id": "promote_to_prod",
                "schedule": None,
                "task_order": [
                    ("promote_staging_to_prod", "restart_api_deployment"),
                ],
                "all_tasks": {
                    "promote_staging_to_prod",
                    "restart_api_deployment",
                },
            },
        ),
        (
            "dags.quality_control",
            {
                "dag_id": "model_drift_detection",
                "schedule": "@weekly",
                "task_order": [
                    ("evaluate_model", "alert_if_drift"),
                ],
                "all_tasks": {"evaluate_model", "alert_if_drift"},
            },
        ),
        (
            "dags.batch_analytics",
            {
                "dag_id": "batch_analytics_reporting",
                "schedule": "0 8 * * *",
                "task_order": [],
                "all_tasks": {"run_batch_inference"},
            },
        ),
        (
            "dags.system_maintenance",
            {
                "dag_id": "system_maintenance",
                "schedule": "0 3 * * 0",
                "task_order": [],
                "all_tasks": {"cleanup_mlruns"},
            },
        ),
    ],
)
def test_dag_structure(module, expected):
    mod = importlib.import_module(module)
    dag = mod.dag

    assert dag.dag_id == expected["dag_id"], f"Неверный dag_id: {dag.dag_id!r}"

    actual_schedule = str(_get_schedule(dag))
    expected_schedule = str(expected["schedule"])
    assert actual_schedule == expected_schedule, (
        f"Неверное расписание: {actual_schedule!r}, ожидалось {expected_schedule!r}"
    )

    actual_tasks = {t.task_id for t in dag.tasks}
    assert actual_tasks == expected["all_tasks"], (
        f"Таски не совпадают. Есть: {actual_tasks}, ожидалось: {expected['all_tasks']}"
    )

    for upstream_id, downstream_id in expected["task_order"]:
        upstream = dag.get_task(upstream_id)
        assert downstream_id in upstream.downstream_task_ids, (
            f"Ожидалось {upstream_id} >> {downstream_id}, но зависимости нет"
        )


def test_promote_is_manual_trigger_only():
    import dags.promote_to_prod as mod

    assert _get_schedule(mod.dag) is None, (
        "promote_to_prod должен быть schedule=None (ручной запуск)"
    )


def test_retrain_does_not_auto_promote():
    import dags.retrain_model_dag as mod

    task_ids = {t.task_id for t in mod.dag.tasks}
    assert "promote_to_prod" not in task_ids, (
        "retrain не должен содержать таск promote_to_prod — промоут только ручной"
    )
