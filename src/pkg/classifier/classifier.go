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
	"log"
	"path/filepath"
	"sync"
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
// Thread-safe: Reload swaps models atomically while Classify uses a read lock.
type Classifier struct {
	mu       sync.RWMutex
	tree     *TreeModel
	svm      *SVMModel
	modelDir string
}

// New loads tree_model.json and svm_model.json from modelDir and returns
// a ready Classifier. Returns an error if either model cannot be loaded.
func New(modelDir string) (*Classifier, error) {
	c := &Classifier{modelDir: modelDir}
	if err := c.loadModels(); err != nil {
		return nil, err
	}
	return c, nil
}

// loadModels reads models from disk. Caller must hold write lock or be in constructor.
func (c *Classifier) loadModels() error {
	tree, err := LoadTree(filepath.Join(c.modelDir, "tree_model.json"))
	if err != nil {
		return fmt.Errorf("load tree model: %w", err)
	}
	svm, err := LoadSVM(filepath.Join(c.modelDir, "svm_model.json"))
	if err != nil {
		return fmt.Errorf("load SVM model: %w", err)
	}
	c.tree = tree
	c.svm = svm
	return nil
}

// Reload re-reads models from disk. Called after training produces new models.
// Returns an error if the new models fail to load (old models remain active).
func (c *Classifier) Reload() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	oldTree, oldSVM := c.tree, c.svm
	if err := c.loadModels(); err != nil {
		c.tree, c.svm = oldTree, oldSVM
		return fmt.Errorf("reload failed, keeping old models: %w", err)
	}
	log.Printf("models reloaded: tree vocab=%d, svm vocab=%d",
		len(c.tree.Vocabulary), len(c.svm.Vocabulary))
	return nil
}

// Classify returns predictions for the given title from both models.
// An empty title returns "unclassified" from both models with score 0.
func (c *Classifier) Classify(title string) Result {
	c.mu.RLock()
	tree, svm := c.tree, c.svm
	c.mu.RUnlock()

	treeLabel := "unclassified"
	svmLabel := "unclassified"
	if title != "" {
		treeLabel = tree.Predict(title)
		svmLabel = svm.Predict(title)
	}
	return Result{
		TreeLabel: treeLabel,
		TreeScore: LabelScores[treeLabel],
		SVMLabel:  svmLabel,
		SVMScore:  LabelScores[svmLabel],
	}
}
