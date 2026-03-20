package classifier

import (
	"encoding/json"
	"fmt"
	"os"
)

// treeData holds the raw decision tree structure exported from sklearn.
type treeData struct {
	Feature       []int       `json:"feature"`
	Threshold     []float64   `json:"threshold"`
	ChildrenLeft  []int       `json:"children_left"`
	ChildrenRight []int       `json:"children_right"`
	Value         [][]float64 `json:"value"` // [n_nodes][n_classes]
	Classes       []string    `json:"classes"`
}

// TreeModel is an exported sklearn DecisionTreeClassifier with CountVectorizer.
type TreeModel struct {
	Vocabulary map[string]int `json:"vocabulary"`
	NGramMin   int            `json:"ngram_min"`
	NGramMax   int            `json:"ngram_max"`
	Binary     bool           `json:"binary"`
	Tree       treeData       `json:"tree"`
}

// LoadTree reads a TreeModel from the JSON file at path.
func LoadTree(path string) (*TreeModel, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read tree model %q: %w", path, err)
	}
	var m TreeModel
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("parse tree model %q: %w", path, err)
	}
	return &m, nil
}

// Predict returns the predicted class label for the given title.
func (m *TreeModel) Predict(title string) string {
	features := buildCountFeatures(title, m.Vocabulary, m.NGramMin, m.NGramMax, m.Binary)
	node := 0
	t := m.Tree
	for t.ChildrenLeft[node] != -1 {
		feat := t.Feature[node]
		val := 0.0
		if feat >= 0 && feat < len(features) {
			val = features[feat]
		}
		if val <= t.Threshold[node] {
			node = t.ChildrenLeft[node]
		} else {
			node = t.ChildrenRight[node]
		}
	}
	counts := t.Value[node]
	bestIdx := 0
	for i := 1; i < len(counts); i++ {
		if counts[i] > counts[bestIdx] {
			bestIdx = i
		}
	}
	return t.Classes[bestIdx]
}

// buildCountFeatures constructs a CountVectorizer feature vector for title.
// When binary is true, term presence is 1.0 regardless of frequency.
func buildCountFeatures(title string, vocab map[string]int, ngramMin, ngramMax int, binary bool) []float64 {
	tokens := Tokenize(title)
	grams := NGrams(tokens, ngramMin, ngramMax)
	features := make([]float64, len(vocab))
	for _, g := range grams {
		if idx, ok := vocab[g]; ok {
			if binary {
				features[idx] = 1.0
			} else {
				features[idx]++
			}
		}
	}
	return features
}
