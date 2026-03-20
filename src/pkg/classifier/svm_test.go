package classifier

import (
	"testing"
)

func TestLoadSVM(t *testing.T) {
	m, err := LoadSVM(testModelDir + "/svm_model.json")
	if err != nil {
		t.Fatalf("LoadSVM: %v", err)
	}
	if len(m.Vocabulary) == 0 {
		t.Error("vocabulary is empty")
	}
	if len(m.Classes) == 0 {
		t.Error("classes is empty")
	}
	if len(m.Coef) == 0 {
		t.Error("coef is empty")
	}
	if len(m.IDF) == 0 {
		t.Error("idf is empty")
	}
}

func TestSVMPredict(t *testing.T) {
	m, err := LoadSVM(testModelDir + "/svm_model.json")
	if err != nil {
		t.Fatalf("LoadSVM: %v", err)
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

func TestSVMPredictEmptyTitle(t *testing.T) {
	m, err := LoadSVM(testModelDir + "/svm_model.json")
	if err != nil {
		t.Fatalf("LoadSVM: %v", err)
	}
	label := m.Predict("")
	if label == "" {
		t.Error("Predict('') returned empty string, want a class label")
	}
}
