package classifier

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// SVMModel is an exported sklearn LinearSVC with TfidfVectorizer.
type SVMModel struct {
	Vocabulary  map[string]int `json:"vocabulary"`
	NGramMin    int            `json:"ngram_min"`
	NGramMax    int            `json:"ngram_max"`
	SublinearTF bool           `json:"sublinear_tf"`
	IDF         []float64      `json:"idf"`
	Coef        [][]float64    `json:"coef"`      // [n_classes][n_features]
	Intercept   []float64      `json:"intercept"` // [n_classes]
	Classes     []string       `json:"classes"`
}

// LoadSVM reads an SVMModel from the JSON file at path.
func LoadSVM(path string) (*SVMModel, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read SVM model %q: %w", path, err)
	}
	var m SVMModel
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("parse SVM model %q: %w", path, err)
	}
	return &m, nil
}

// Predict returns the predicted class label for the given title.
// Uses TF-IDF with optional sublinear_tf and L2 normalisation, then
// applies the LinearSVC decision function (coef dot features + intercept).
func (m *SVMModel) Predict(title string) string {
	features := buildTfidfFeatures(title, m.Vocabulary, m.NGramMin, m.NGramMax, m.SublinearTF, m.IDF)
	bestIdx := 0
	bestScore := math.Inf(-1)
	for i, classCoef := range m.Coef {
		score := m.Intercept[i]
		for j, v := range features {
			if v != 0 && j < len(classCoef) {
				score += classCoef[j] * v
			}
		}
		if score > bestScore {
			bestScore = score
			bestIdx = i
		}
	}
	return m.Classes[bestIdx]
}

// buildTfidfFeatures constructs a TF-IDF feature vector for title.
// Applies sublinear_tf (log(1+tf)) when enabled, multiplies by IDF, then L2 normalises.
func buildTfidfFeatures(title string, vocab map[string]int, ngramMin, ngramMax int, sublinearTF bool, idf []float64) []float64 {
	tokens := Tokenize(title)
	grams := NGrams(tokens, ngramMin, ngramMax)

	counts := make(map[int]float64)
	for _, g := range grams {
		if idx, ok := vocab[g]; ok {
			counts[idx]++
		}
	}

	features := make([]float64, len(vocab))
	for idx, tf := range counts {
		if sublinearTF {
			tf = 1.0 + math.Log(tf)
		}
		if idx < len(idf) {
			features[idx] = tf * idf[idx]
		}
	}

	norm := 0.0
	for _, v := range features {
		norm += v * v
	}
	if norm > 0 {
		norm = math.Sqrt(norm)
		for i := range features {
			features[i] /= norm
		}
	}

	return features
}
