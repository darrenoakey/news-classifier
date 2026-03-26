![](banner.jpg)

# news-classifier

A Go HTTP service that classifies news article headlines into categories (`great`, `good`, `other`) using exported sklearn models — a Decision Tree with CountVectorizer and a Linear SVC with TF-IDF. Both models run in pure Go with zero Python runtime dependency.

Models are trained in Python (sklearn), exported to JSON via `scripts/export_model.py`, and loaded by the Go service at startup. Hot-reload is supported — push new models to disk and hit `/reload`.

## Running

```bash
# Build the binary
./run build

# Start the server (default :8810, models from local/models/)
./run start

# With custom options
./run start -addr :9000 -models /path/to/models

# Run tests
./run test

# Run all checks (gofmt, go vet, tests)
./run check
```

## API

All endpoints return JSON with `Content-Type: application/json`.

### POST /classify

Classify a single headline directly.

**Request:**
```json
{"title": "OpenAI releases GPT-5"}
```

**Response:**
```json
{
  "tree_label": "great",
  "tree_score": 2.0,
  "svm_label": "great",
  "svm_score": 2.0
}
```

Labels are `great` (score 2.0), `good` (1.0), or `other` (0.0). An empty title returns `unclassified` with score 0.

### POST /execute

Engine HTTP-mode envelope format. Accepts an input map with a `title` field and passes through standard news fields (`link`, `summary`, `published`, `feed_name`, `feed_url`) alongside classification results.

**Request:**
```json
{
  "input": {
    "title": "City council votes on parking meters",
    "link": "https://example.com/article",
    "feed_name": "Local News",
    "published": "2026-03-25"
  },
  "config": {},
  "function": "compute"
}
```

**Response:**
```json
{
  "title": "City council votes on parking meters",
  "tree_label": "other",
  "tree_score": 0.0,
  "svm_label": "other",
  "svm_score": 0.0,
  "link": "https://example.com/article",
  "feed_name": "Local News",
  "published": "2026-03-25"
}
```

### POST /execute-batch

Batch version of `/execute`. Accepts a JSON array of envelopes, returns results in the same order.

**Request:**
```json
[
  {"input": {"title": "AI breakthrough announced"}, "config": {}, "function": "compute"},
  {"input": {"title": "Local sports recap"}, "config": {}, "function": "compute"}
]
```

**Response:**
```json
[
  {"title": "AI breakthrough announced", "tree_label": "great", "tree_score": 2.0, "svm_label": "great", "svm_score": 2.0},
  {"title": "Local sports recap", "tree_label": "other", "tree_score": 0.0, "svm_label": "other", "svm_score": 0.0}
]
```

### GET /reload

Re-reads models from disk without restarting. Call this after training produces new model files. Returns `{"ok":true,"reloaded":true}` on success.

### GET /health

Returns `{"ok":true}`. Use for liveness checks.

## Model Export

The Python script `scripts/export_model.py` converts sklearn joblib models to JSON:

```bash
python3 scripts/export_model.py
```

It reads `tree_model.joblib` (CountVectorizer + DecisionTreeClassifier) and `svm_model.joblib` (TfidfVectorizer + LinearSVC) and writes `tree_model.json` and `svm_model.json` for the Go service to load.

## Project Structure

```
src/
  cmd/classifier/main.go     # Entry point, flag parsing
  pkg/classifier/
    classifier.go             # Model loading, thread-safe classify
    server.go                 # HTTP handlers
    tree.go                   # Decision tree inference
    svm.go                    # Linear SVC inference
    tokenize.go               # N-gram tokenization
scripts/
  export_model.py             # sklearn -> JSON exporter
run                           # Build/test/start wrapper
```

## License

This project is licensed under [CC BY-NC 4.0](https://darren-static.waft.dev) - free to use and modify, but no commercial use without permission.
