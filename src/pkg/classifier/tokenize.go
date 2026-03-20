package classifier

import (
	"regexp"
	"strings"
	"unicode"
)

// tokenRe matches sklearn's default token pattern: two or more word characters.
var tokenRe = regexp.MustCompile(`(?i)\b\w\w+\b`)

// Tokenize splits text into lowercase tokens matching sklearn's default pattern.
// It filters to only tokens that contain at least one alphanumeric character,
// matching CountVectorizer/TfidfVectorizer default behaviour.
func Tokenize(text string) []string {
	raw := tokenRe.FindAllString(strings.ToLower(text), -1)
	out := make([]string, 0, len(raw))
	for _, tok := range raw {
		if containsAlnum(tok) {
			out = append(out, tok)
		}
	}
	return out
}

// NGrams generates n-grams from tokens for the given min/max n-gram sizes.
// Tokens are joined with a single space, matching sklearn's behaviour.
func NGrams(tokens []string, minN, maxN int) []string {
	var grams []string
	n := len(tokens)
	for size := minN; size <= maxN; size++ {
		for i := 0; i <= n-size; i++ {
			grams = append(grams, strings.Join(tokens[i:i+size], " "))
		}
	}
	return grams
}

// containsAlnum returns true if s contains at least one letter or digit.
func containsAlnum(s string) bool {
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			return true
		}
	}
	return false
}
