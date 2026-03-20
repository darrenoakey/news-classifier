package classifier

import (
	"path/filepath"
	"runtime"
	"testing"
)

// testModelDir resolves the absolute path to local/models/ relative to this
// source file so that tests work regardless of the working directory.
var testModelDir = func() string {
	_, file, _, _ := runtime.Caller(0)
	// file is .../src/pkg/classifier/tree_test.go
	// project root is four levels up from pkg/classifier/
	return filepath.Join(filepath.Dir(file), "..", "..", "..", "local", "models")
}()

func TestLoadTree(t *testing.T) {
	m, err := LoadTree(testModelDir + "/tree_model.json")
	if err != nil {
		t.Fatalf("LoadTree: %v", err)
	}
	if len(m.Vocabulary) == 0 {
		t.Error("vocabulary is empty")
	}
	if len(m.Tree.Classes) == 0 {
		t.Error("classes is empty")
	}
	if len(m.Tree.Feature) == 0 {
		t.Error("tree has no nodes")
	}
}

func TestTreePredict(t *testing.T) {
	m, err := LoadTree(testModelDir + "/tree_model.json")
	if err != nil {
		t.Fatalf("LoadTree: %v", err)
	}

	cases := []struct {
		title string
	}{
		{"OpenAI releases GPT-5 with major reasoning improvements"},
		{"Local council approves new parking regulations downtown"},
		{"Scientists discover new species in Amazon rainforest"},
	}

	for _, tc := range cases {
		t.Run(tc.title[:20], func(t *testing.T) {
			label := m.Predict(tc.title)
			validLabels := map[string]bool{"great": true, "good": true, "other": true}
			if !validLabels[label] {
				t.Errorf("Predict(%q) = %q, want one of great/good/other", tc.title, label)
			}
		})
	}
}

func TestTreePredictEmptyTitle(t *testing.T) {
	m, err := LoadTree(testModelDir + "/tree_model.json")
	if err != nil {
		t.Fatalf("LoadTree: %v", err)
	}
	label := m.Predict("")
	if label == "" {
		t.Error("Predict('') returned empty string, want a class label")
	}
}
