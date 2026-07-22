# Spam Detector — Demo

A lightweight Streamlit app that connects to the classification API and lets you paste any email text to instantly check whether it's spam or legitimate.

No ML setup required — just point it at a running API.

---

## What it does

You paste a piece of email text. The app sends it to the `/api/v1/classify` endpoint and displays:

- **Verdict** — Spam 🚨 or Ham ✅
- **Confidence score** — how certain the model is
- **Raw probabilities** — expandable JSON for both classes

---

## Requirements

- Python 3.10+
- A running instance of the classification API (local or remote)

---

## Run locally

```bash
cd demo
pip install -r requirements.txt

export API_URL=http://localhost:8000/api/v1/classify
export API_KEY=your_api_key_here   # leave empty if auth is disabled

streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Run with Docker

```bash
docker build -t spam-detector-demo ./demo

docker run -p 8501:8501 \
  -e API_URL=http://host.docker.internal:8000/api/v1/classify \
  -e API_KEY=your_api_key_here \
  spam-detector-demo
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `API_URL` | `http://localhost:8000/api/v1/classify` | Classification endpoint |
| `API_KEY` | *(empty)* | API key sent in `X-API-Key` header |

---

## Starting the API (if you don't have one running)

If you want to run everything locally from scratch:

```bash
# From the project root
uv pip install -e ".[api]"
cp .env.example .env   # fill in MLFLOW_TRACKING_URI and API_KEY

python -m src.run_api
# API is now at http://localhost:8000
```

Then start the demo in a second terminal as shown above.

---

## Quickstart notebook

`demo/quickstart.ipynb` walks through the same classification flow programmatically — useful for exploring model outputs or integrating the API into your own code.

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/classify",
    json={"text": "Congratulations! You've won a free iPhone. Click here to claim."},
    headers={"X-API-Key": "your_key"},
)
print(response.json())
# {'label_id': 1, 'confidence': 0.9871, 'all_probabilities': [0.0129, 0.9871]}
```

---

## Interpreting results

| `label_id` | Meaning | Confidence |
|---|---|---|
| `0` | Ham — looks legitimate | Probability the model assigns to class 0 |
| `1` | Spam — likely unsolicited | Probability the model assigns to class 1 |

The classification threshold is set at **0.5422** by default (tuned for optimal Expected Recall on the test set). It can be adjusted in `configs/main.yaml` → `inference.threshold` without retraining.
