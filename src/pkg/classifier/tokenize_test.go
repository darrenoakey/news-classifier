package classifier

import (
	"reflect"
	"testing"
)

func TestTokenize(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  []string
	}{
		{
			name:  "simple words",
			input: "Hello World",
			want:  []string{"hello", "world"},
		},
		{
			name:  "single char words are excluded",
			input: "I am a go developer",
			want:  []string{"am", "go", "developer"},
		},
		{
			name:  "punctuation stripped",
			input: "AI/ML is great!",
			want:  []string{"ai", "ml", "is", "great"},
		},
		{
			name:  "numbers included",
			input: "GPT-4 beats humans",
			want:  []string{"gpt", "beats", "humans"},
		},
		{
			name:  "empty string",
			input: "",
			want:  []string{},
		},
		{
			name:  "news headline",
			input: "OpenAI releases new model with improved reasoning",
			want:  []string{"openai", "releases", "new", "model", "with", "improved", "reasoning"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := Tokenize(tc.input)
			if len(got) == 0 && len(tc.want) == 0 {
				return
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("Tokenize(%q)\n  got  %v\n  want %v", tc.input, got, tc.want)
			}
		})
	}
}

func TestNGrams(t *testing.T) {
	cases := []struct {
		name   string
		tokens []string
		minN   int
		maxN   int
		want   []string
	}{
		{
			name:   "unigrams only",
			tokens: []string{"hello", "world"},
			minN:   1,
			maxN:   1,
			want:   []string{"hello", "world"},
		},
		{
			name:   "bigrams only",
			tokens: []string{"hello", "world"},
			minN:   2,
			maxN:   2,
			want:   []string{"hello world"},
		},
		{
			name:   "unigrams and bigrams",
			tokens: []string{"hello", "world"},
			minN:   1,
			maxN:   2,
			want:   []string{"hello", "world", "hello world"},
		},
		{
			name:   "three tokens bigrams",
			tokens: []string{"a", "b", "c"},
			minN:   2,
			maxN:   2,
			want:   []string{"a b", "b c"},
		},
		{
			name:   "empty tokens",
			tokens: []string{},
			minN:   1,
			maxN:   2,
			want:   nil,
		},
		{
			name:   "single token bigram yields nothing",
			tokens: []string{"only"},
			minN:   2,
			maxN:   2,
			want:   nil,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := NGrams(tc.tokens, tc.minN, tc.maxN)
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("NGrams(%v, %d, %d)\n  got  %v\n  want %v", tc.tokens, tc.minN, tc.maxN, got, tc.want)
			}
		})
	}
}
