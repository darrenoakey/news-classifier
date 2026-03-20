// Package classifier provides text classification using exported sklearn models.
// It supports DecisionTree and LinearSVC models with Count/Tfidf vectorization.
//
// Typical usage:
//
//	c, err := classifier.New("local/models")
//	if err != nil { ... }
//	result := c.Classify("OpenAI releases new model")
package classifier

import (
	"fmt"
	"path/filepath"
)

// LabelScores maps class labels to numeric scores for downstream ranking.
var LabelScores = map[string]float64{
	"great": 2.0,
	"good":  1.0,
	"other": 0.0,
}

// Result holds classification predictions from both models.
type Result struct {
	TreeLabel string  `json:"tree_label"`
	TreeScore float64 `json:"tree_score"`
	SVMLabel  string  `json:"svm_label"`
	SVMScore  float64 `json:"svm_score"`
}

// Classifier holds loaded tree and SVM models for text classification.
type Classifier struct {
	tree *TreeModel
	svm  *SVMModel
}

// New loads tree_model.json and svm_model.json from modelDir and returns
// a ready Classifier. Returns an error if either model cannot be loaded.
func New(modelDir string) (*Classifier, error) {
	tree, err := LoadTree(filepath.Join(modelDir, "tree_model.json"))
	if err != nil {
		return nil, fmt.Errorf("load tree model: %w", err)
	}
	svm, err := LoadSVM(filepath.Join(modelDir, "svm_model.json"))
	if err != nil {
		return nil, fmt.Errorf("load SVM model: %w", err)
	}
	return &Classifier{tree: tree, svm: svm}, nil
}

// Classify returns predictions for the given title from both models.
// An empty title returns "unclassified" from both models with score 0.
func (c *Classifier) Classify(title string) Result {
	treeLabel := "unclassified"
	svmLabel := "unclassified"
	if title != "" {
		treeLabel = c.tree.Predict(title)
		svmLabel = c.svm.Predict(title)
	}
	return Result{
		TreeLabel: treeLabel,
		TreeScore: LabelScores[treeLabel],
		SVMLabel:  svmLabel,
		SVMScore:  LabelScores[svmLabel],
	}
}
