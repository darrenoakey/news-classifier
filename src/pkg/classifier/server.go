package classifier

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"
)

// Server wraps a Classifier in an HTTP handler.
type Server struct {
	c    *Classifier
	addr string
	mux  *http.ServeMux
}

// classifyRequest is the JSON body for POST /classify.
type classifyRequest struct {
	Title string `json:"title"`
}

// executeEnvelope is the engine HTTP-mode request body for POST /execute.
type executeEnvelope struct {
	Input    map[string]any `json:"input"`
	Config   map[string]any `json:"config"`
	Function string         `json:"function"`
}

// passthroughFields are input fields copied verbatim into /execute responses.
var passthroughFields = []string{"link", "summary", "published", "feed_name", "feed_url"}

// NewServer creates a Server with the given classifier and listen address.
func NewServer(c *Classifier, addr string) *Server {
	s := &Server{c: c, addr: addr, mux: http.NewServeMux()}
	s.mux.HandleFunc("/classify", s.handleClassify)
	s.mux.HandleFunc("/execute", s.handleExecute)
	s.mux.HandleFunc("/execute-batch", s.handleExecuteBatch)
	s.mux.HandleFunc("/reload", s.handleReload)
	s.mux.HandleFunc("/health", handleHealth)
	return s
}

// ListenAndServe starts the HTTP server and blocks until it returns an error.
func (s *Server) ListenAndServe() error {
	log.Printf("news-classifier listening on %s", s.addr)
	return http.ListenAndServe(s.addr, s.mux)
}

// Handler returns the http.Handler for use in tests without a live server.
func (s *Server) Handler() http.Handler {
	return s.mux
}

// handleClassify handles POST /classify.
// Body: {"title":"..."}, Response: Result JSON.
func (s *Server) handleClassify(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req classifyRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request: "+err.Error(), http.StatusBadRequest)
		return
	}
	result := s.c.Classify(req.Title)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result) //nolint:errcheck
}

// handleExecute handles POST /execute (engine HTTP-mode envelope).
// Passes through known news fields from input alongside classification results.
func (s *Server) handleExecute(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var env executeEnvelope
	if err := json.NewDecoder(r.Body).Decode(&env); err != nil {
		http.Error(w, "bad request: "+err.Error(), http.StatusBadRequest)
		return
	}

	title := ""
	if v, ok := env.Input["title"]; ok {
		title, _ = v.(string)
	}

	result := s.c.Classify(title)

	out := map[string]any{
		"title":         title,
		"tree_label":    result.TreeLabel,
		"tree_score":    result.TreeScore,
		"svm_label":     result.SVMLabel,
		"svm_score":     result.SVMScore,
		"classified_at": time.Now().Format("2006-01-02"),
	}
	for _, field := range passthroughFields {
		if v, ok := env.Input[field]; ok {
			out[field] = v
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(out) //nolint:errcheck
}

// handleExecuteBatch handles POST /execute-batch.
// Body: JSON array of executeEnvelope objects. Returns JSON array of results in the same order.
func (s *Server) handleExecuteBatch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var envs []executeEnvelope
	if err := json.NewDecoder(r.Body).Decode(&envs); err != nil {
		http.Error(w, "bad request: "+err.Error(), http.StatusBadRequest)
		return
	}

	results := make([]map[string]any, len(envs))
	for i, env := range envs {
		title := ""
		if v, ok := env.Input["title"]; ok {
			title, _ = v.(string)
		}

		result := s.c.Classify(title)

		out := map[string]any{
			"title":         title,
			"tree_label":    result.TreeLabel,
			"tree_score":    result.TreeScore,
			"svm_label":     result.SVMLabel,
			"svm_score":     result.SVMScore,
			"classified_at": time.Now().Format("2006-01-02"),
		}
		for _, field := range passthroughFields {
			if v, ok := env.Input[field]; ok {
				out[field] = v
			}
		}
		results[i] = out
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(results) //nolint:errcheck
}

// handleReload re-reads models from disk. Called by the training cell after
// producing new models. GET or POST both accepted.
func (s *Server) handleReload(w http.ResponseWriter, _ *http.Request) {
	if err := s.c.Reload(); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	fmt.Fprintln(w, `{"ok":true,"reloaded":true}`)
}

// handleHealth handles GET /health.
func handleHealth(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	fmt.Fprintln(w, `{"ok":true}`)
}
