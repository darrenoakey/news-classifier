package classifier

import (
	"testing"
)

func TestNew(t *testing.T) {
	c, err := New(testModelDir)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if c.tree == nil {
		t.Error("tree model is nil")
	}
	if c.svm == nil {
		t.Error("SVM model is nil")
	}
}

func TestNewMissingDir(t *testing.T) {
	_, err := New("/nonexistent/path")
	if err == nil {
		t.Error("expected error for missing model directory, got nil")
	}
}

func TestClassifyEmptyTitle(t *testing.T) {
	c, err := New(testModelDir)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	r := c.Classify("")
	if r.TreeLabel != "unclassified" {
		t.Errorf("empty title TreeLabel = %q, want unclassified", r.TreeLabel)
	}
	if r.SVMLabel != "unclassified" {
		t.Errorf("empty title SVMLabel = %q, want unclassified", r.SVMLabel)
	}
	if r.TreeScore != 0 {
		t.Errorf("empty title TreeScore = %v, want 0", r.TreeScore)
	}
	if r.SVMScore != 0 {
		t.Errorf("empty title SVMScore = %v, want 0", r.SVMScore)
	}
}

func TestClassifyKnownTitles(t *testing.T) {
	c, err := New(testModelDir)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	validLabels := map[string]bool{"great": true, "good": true, "other": true}

	titles := []string{
		"OpenAI releases GPT-5 with major reasoning improvements",
		"Scientists discover breakthrough cancer treatment",
		"Local council approves new parking regulations downtown",
		"Claude AI announces new context window improvements",
		"Stock market reaches all-time high amid tech rally",
		"New study shows coffee consumption linked to longevity",
	}

	for _, title := range titles {
		t.Run(title[:min(len(title), 30)], func(t *testing.T) {
			r := c.Classify(title)
			if !validLabels[r.TreeLabel] {
				t.Errorf("Classify(%q).TreeLabel = %q, want great/good/other", title, r.TreeLabel)
			}
			if !validLabels[r.SVMLabel] {
				t.Errorf("Classify(%q).SVMLabel = %q, want great/good/other", title, r.SVMLabel)
			}
			if r.TreeScore != LabelScores[r.TreeLabel] {
				t.Errorf("Classify(%q).TreeScore = %v, want %v", title, r.TreeScore, LabelScores[r.TreeLabel])
			}
			if r.SVMScore != LabelScores[r.SVMLabel] {
				t.Errorf("Classify(%q).SVMScore = %v, want %v", title, r.SVMScore, LabelScores[r.SVMLabel])
			}
		})
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
