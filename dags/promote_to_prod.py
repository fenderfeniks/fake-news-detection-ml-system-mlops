# dags/promote_to_prod.py
"""
DAG: Model Promotion (Manual Approval Gate).
Переводит модель из Staging в Production и перезапускает API.
"""

import pendulum
from airflow import DAG
from airflow.models import Variable
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s


# ИСПРАВЛЕНИЕ: Меняем дефолтный тег на api-latest
IMAGE = Variable.get("PROJECT_IMAGE", default_var="my-company/fake_news_detector:api-latest")
NAMESPACE = Variable.get("K8S_NAMESPACE", default_var="ml-pipelines")

# ИСПРАВЛЕНИЕ: Защитный словарь-заглушка от краша парсера Airflow
DEFAULT_CONFIG = {
    "default_args": {
        "owner": "mlops",
    },
    "mount_path": "/app/models",
    "pvc_name": "model-weights-pvc",
}

# ИСПРАВЛЕНИЕ: Передаем default_var
CONFIG = Variable.get("promotion_config", default_var=DEFAULT_CONFIG, deserialize_json=True)

default_args = {
    "owner": CONFIG["default_args"]["owner"],
    "depends_on_past": False,
    "start_date": pendulum.datetime(2026, 1, 1, tz="UTC"),
}

with DAG(
    "promote_to_prod",
    default_args=default_args,
    schedule=None,  # СТРОГО РУЧНОЙ ЗАПУСК
    catchup=False,
    tags=["nlp", "mlops", "production"],
) as dag:
    promote_model = KubernetesPodOperator(
        task_id="promote_staging_to_prod",
        name="promote-model-pod",
        namespace=NAMESPACE,
        image=IMAGE,
        cmds=["python", "-m", "src.jobs.promote"],
        service_account_name="airflow-worker-sa",
        env_from=[
            k8s.V1EnvFromSource(
                config_map_ref=k8s.V1ConfigMapEnvSource(name="fake-news-api-config")
            ),
        ],
        get_logs=True,
        is_delete_operator_pod=True,
    )

    restart_api = KubernetesPodOperator(
        task_id="restart_api_deployment",
        name="restart-api-pod",
        namespace=NAMESPACE,
        image="bitnami/kubectl:1.29",
        cmds=["kubectl", "rollout", "restart", "deployment/fake-news-api", "-n", NAMESPACE],
        get_logs=True,
        is_delete_operator_pod=True,
        service_account_name="airflow-worker-sa",
    )
    promote_model >> restart_api
