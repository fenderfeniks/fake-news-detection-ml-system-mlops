# src/api/metrics.py
from prometheus_client import Counter, Histogram


# Счетчик запросов на классификацию
CLASSIFICATION_REQUESTS_TOTAL = Counter(
    "nlp_classification_requests_total",
    "Total number of classification requests",
    ["source"],  # label: "tg", "rest"
)

# Время инференса классификатора
CLASSIFICATION_INFERENCE_TIME = Histogram(
    "nlp_classification_inference_seconds", "Time spent classifying text in model", ["source"]
)
