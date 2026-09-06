package main

import (
	"fmt"
	"strings"
	"testing"
)

// go test -run TestTokenCounts -v   (needs LLMASH_TOKENIZER_JSON)
// Prints one count per sample so the same samples can be counted with the
// reference Python `tokenizers` library and diffed.
func TestTokenCounts(t *testing.T) {
	path := env("LLMASH_TOKENIZER_JSON")
	if path == "" {
		t.Skip("LLMASH_TOKENIZER_JSON not set")
	}
	tk, err := loadTokenizer(path)
	if err != nil {
		t.Fatal(err)
	}
	samples := tokenSamples()
	for i, s := range samples {
		fmt.Printf("SAMPLE %d %d\n", i, tk.Count(s))
	}
}

func tokenSamples() []string {
	return []string{
		"Hello, world!",
		"The quick brown fox jumps over the lazy dog.",
		"def sieve(n):\n    primes = [True] * (n + 1)\n    for i in range(2, int(n ** 0.5) + 1):\n        if primes[i]:\n            for j in range(i * i, n + 1, i):\n                primes[j] = False\n    return [i for i in range(2, n + 1) if primes[i]]\n",
		"I'm sure they'll've done it by 2026-09-06, won't they?   Yes.\n\n\nNo.",
		"Numbers: 1234567890 3.14159 1e10 0x1F, tabs\there\tand   spaces.",
		"日本語のテキストと絵文字 🎉🚀 mixed with English words and 漢字.",
		"<think>\nOkay, let me actually look at this instead of reaching for a template.\n</think>\n\nThe answer is 42.",
		"<tool_call>\n<function=write_file>\n<parameter=path>\nprimes.py\n</parameter>\n</function>\n</tool_call>",
		strings.Repeat("lorem ipsum dolor sit amet, consectetur adipiscing elit. ", 40),
		"   leading spaces\nand trailing spaces   \n  ",
	}
}
