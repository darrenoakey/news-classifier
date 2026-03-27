package classifier

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func newTestServer(t *testing.T) *Server {
	t.Helper()
	c, err := New(testModelDir)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return NewServer(c, ":0")
}

func TestHealthEndpoint(t *testing.T) {
	srv := newTestServer(t)
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("/health status = %d, want 200", w.Code)
	}
	var body map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode /health response: %v", err)
	}
	if ok, _ := body["ok"].(bool); !ok {
		t.Errorf("/health response ok = %v, want true", body["ok"])
	}
}

func TestClassifyEndpoint(t *testing.T) {
	srv := newTestServer(t)

	cases := []struct {
		name  string
		title string
	}{
		{"ai news", "OpenAI releases new GPT model with improved capabilities"},
		{"local news", "City council votes on new parking meter rates downtown"},
		{"empty", ""},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			body, _ := json.Marshal(map[string]string{"title": tc.title})
			req := httptest.NewRequest(http.MethodPost, "/classify", bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()
			srv.Handler().ServeHTTP(w, req)

			if w.Code != http.StatusOK {
				t.Fatalf("/classify status = %d, want 200", w.Code)
			}
			var result Result
			if err := json.Unmarshal(w.Body.Bytes(), &result); err != nil {
				t.Fatalf("decode /classify response: %v", err)
			}
			if tc.title == "" {
				if result.TreeLabel != "unclassified" {
					t.Errorf("empty title TreeLabel = %q, want unclassified", result.TreeLabel)
				}
			} else {
				validLabels := map[string]bool{"great": true, "good": true, "other": true}
				if !validLabels[result.TreeLabel] {
					t.Errorf("TreeLabel = %q, want great/good/other", result.TreeLabel)
				}
				if !validLabels[result.SVMLabel] {
					t.Errorf("SVMLabel = %q, want great/good/other", result.SVMLabel)
				}
			}
		})
	}
}

func TestClassifyEndpointMethodNotAllowed(t *testing.T) {
	srv := newTestServer(t)
	req := httptest.NewRequest(http.MethodGet, "/classify", nil)
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)
	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("/classify GET status = %d, want 405", w.Code)
	}
}

func TestClassifyEndpointBadJSON(t *testing.T) {
	srv := newTestServer(t)
	req := httptest.NewRequest(http.MethodPost, "/classify", bytes.NewBufferString("not json"))
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)
	if w.Code != http.StatusBadRequest {
		t.Errorf("/classify bad JSON status = %d, want 400", w.Code)
	}
}

func TestExecuteEndpoint(t *testing.T) {
	srv := newTestServer(t)

	envelope := map[string]any{
		"input": map[string]any{
			"title":     "OpenAI releases GPT-5",
			"link":      "https://example.com/article",
			"feed_name": "Tech News",
			"published": "2026-01-01",
		},
		"config":   map[string]any{},
		"function": "compute",
	}
	body, _ := json.Marshal(envelope)
	req := httptest.NewRequest(http.MethodPost, "/execute", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("/execute status = %d, want 200", w.Code)
	}
	var out map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode /execute response: %v", err)
	}
	requiredFields := []string{"title", "tree_label", "tree_score", "svm_label", "svm_score", "link", "feed_name", "published", "classified_at"}
	for _, f := range requiredFields {
		if _, ok := out[f]; !ok {
			t.Errorf("/execute response missing field %q", f)
		}
	}
}

func TestExecuteEndpointPassthrough(t *testing.T) {
	srv := newTestServer(t)

	envelope := map[string]any{
		"input": map[string]any{
			"title":    "Some news article",
			"link":     "https://example.com/news",
			"summary":  "Article summary here",
			"feed_url": "https://example.com/feed.xml",
		},
		"config":   map[string]any{},
		"function": "compute",
	}
	body, _ := json.Marshal(envelope)
	req := httptest.NewRequest(http.MethodPost, "/execute", bytes.NewReader(body))
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)

	var out map[string]any
	json.Unmarshal(w.Body.Bytes(), &out) //nolint:errcheck

	for _, field := range []string{"link", "summary", "feed_url"} {
		if _, ok := out[field]; !ok {
			t.Errorf("/execute response missing passthrough field %q", field)
		}
	}
}

func TestExecuteEndpointMethodNotAllowed(t *testing.T) {
	srv := newTestServer(t)
	req := httptest.NewRequest(http.MethodGet, "/execute", nil)
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)
	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("/execute GET status = %d, want 405", w.Code)
	}
}

func TestExecuteBatchEndpoint(t *testing.T) {
	srv := newTestServer(t)

	batch := []map[string]any{
		{
			"input": map[string]any{
				"title":     "OpenAI releases GPT-5",
				"link":      "https://example.com/article1",
				"feed_name": "Tech News",
				"published": "2026-01-01",
			},
			"config":   map[string]any{},
			"function": "compute",
		},
		{
			"input": map[string]any{
				"title":    "City council votes on parking meters",
				"link":     "https://example.com/article2",
				"summary":  "Local news summary",
				"feed_url": "https://example.com/feed.xml",
			},
			"config":   map[string]any{},
			"function": "compute",
		},
	}
	body, _ := json.Marshal(batch)
	req := httptest.NewRequest(http.MethodPost, "/execute-batch", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("/execute-batch status = %d, want 200", w.Code)
	}
	var out []map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode /execute-batch response: %v", err)
	}
	if len(out) != 2 {
		t.Fatalf("/execute-batch response len = %d, want 2", len(out))
	}

	// First item: classification fields + passthrough link/feed_name/published
	for _, f := range []string{"title", "tree_label", "tree_score", "svm_label", "svm_score", "link", "feed_name", "published", "classified_at"} {
		if _, ok := out[0][f]; !ok {
			t.Errorf("/execute-batch item 0 missing field %q", f)
		}
	}

	// Second item: classification fields + passthrough link/summary/feed_url
	for _, f := range []string{"title", "tree_label", "tree_score", "svm_label", "svm_score", "link", "summary", "feed_url", "classified_at"} {
		if _, ok := out[1][f]; !ok {
			t.Errorf("/execute-batch item 1 missing field %q", f)
		}
	}

	// Verify ordering is preserved
	if title, _ := out[0]["title"].(string); title != "OpenAI releases GPT-5" {
		t.Errorf("/execute-batch item 0 title = %q, want OpenAI releases GPT-5", title)
	}
	if title, _ := out[1]["title"].(string); title != "City council votes on parking meters" {
		t.Errorf("/execute-batch item 1 title = %q, want City council votes on parking meters", title)
	}
}

func TestExecuteBatchEndpointEmpty(t *testing.T) {
	srv := newTestServer(t)

	body, _ := json.Marshal([]map[string]any{})
	req := httptest.NewRequest(http.MethodPost, "/execute-batch", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("/execute-batch empty status = %d, want 200", w.Code)
	}
	var out []map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode /execute-batch empty response: %v", err)
	}
	if len(out) != 0 {
		t.Errorf("/execute-batch empty response len = %d, want 0", len(out))
	}
}

func TestExecuteBatchEndpointMethodNotAllowed(t *testing.T) {
	srv := newTestServer(t)
	req := httptest.NewRequest(http.MethodGet, "/execute-batch", nil)
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)
	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("/execute-batch GET status = %d, want 405", w.Code)
	}
}

func TestExecuteBatchEndpointBadJSON(t *testing.T) {
	srv := newTestServer(t)
	req := httptest.NewRequest(http.MethodPost, "/execute-batch", bytes.NewBufferString("not json"))
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)
	if w.Code != http.StatusBadRequest {
		t.Errorf("/execute-batch bad JSON status = %d, want 400", w.Code)
	}
}
