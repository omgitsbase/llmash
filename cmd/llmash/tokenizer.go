package main

import (
	"encoding/json"
	"os"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"
)

// A byte-level BPE tokenizer read from a HuggingFace tokenizer.json, used
// only to COUNT tokens for the live rate on the fast-backend path (that
// backend reports usage once, at the end). The pre-tokenizer implements the
// GPT-2 / Qwen2 split pattern by hand, since Go's regexp has no lookahead:
//
//   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}
//   | ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+

type bpeTokenizer struct {
	vocab   map[string]int
	ranks   map[[2]string]int
	added   []string
	byteEnc [256]string
	cache   sync.Map
}

var (
	tokOnce sync.Once
	tokInst *bpeTokenizer
)

func genTokenizer() *bpeTokenizer {
	tokOnce.Do(func() {
		if tokenizerJSON == "" {
			return
		}
		t, err := loadTokenizer(tokenizerJSON)
		if err != nil {
			logf("no tokenizer (%v); live counts fall back to the client's estimate", err)
			return
		}
		tokInst = t
		logf("live token counts from %s", tokenizerJSON)
	})
	return tokInst
}

func loadTokenizer(path string) (*bpeTokenizer, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var doc struct {
		AddedTokens []struct {
			Content string `json:"content"`
		} `json:"added_tokens"`
		Model struct {
			Vocab  map[string]int `json:"vocab"`
			Merges []any          `json:"merges"`
		} `json:"model"`
	}
	if err := json.Unmarshal(b, &doc); err != nil {
		return nil, err
	}
	t := &bpeTokenizer{vocab: doc.Model.Vocab, ranks: map[[2]string]int{}}
	for i, m := range doc.Model.Merges {
		switch v := m.(type) {
		case string:
			parts := strings.SplitN(v, " ", 2)
			if len(parts) == 2 {
				t.ranks[[2]string{parts[0], parts[1]}] = i
			}
		case []any:
			if len(v) == 2 {
				t.ranks[[2]string{toStr(v[0]), toStr(v[1])}] = i
			}
		}
	}
	for _, a := range doc.AddedTokens {
		if a.Content != "" {
			t.added = append(t.added, a.Content)
		}
	}
	// GPT-2's byte -> unicode map: printable bytes map to themselves, the rest
	// to code points 256 upward.
	n := 0
	for b := 0; b < 256; b++ {
		if (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF) {
			t.byteEnc[b] = string(rune(b))
		} else {
			t.byteEnc[b] = string(rune(256 + n))
			n++
		}
	}
	return t, nil
}

func toStr(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}

func (t *bpeTokenizer) Count(text string) int {
	if t == nil || text == "" {
		return 0
	}
	n := 0
	// added (special) tokens are matched whole before anything else
	for _, piece := range t.splitAdded(text) {
		if piece.special {
			n++
			continue
		}
		for _, chunk := range pretokenize(piece.text) {
			n += t.bpeCount(chunk)
		}
	}
	return n
}

type piece struct {
	text    string
	special bool
}

func (t *bpeTokenizer) splitAdded(text string) []piece {
	if len(t.added) == 0 {
		return []piece{{text, false}}
	}
	var out []piece
	for len(text) > 0 {
		best, at := "", -1
		for _, a := range t.added {
			if i := strings.Index(text, a); i >= 0 && (at < 0 || i < at || (i == at && len(a) > len(best))) {
				best, at = a, i
			}
		}
		if at < 0 {
			out = append(out, piece{text, false})
			break
		}
		if at > 0 {
			out = append(out, piece{text[:at], false})
		}
		out = append(out, piece{best, true})
		text = text[at+len(best):]
	}
	return out
}

func isLetter(r rune) bool { return unicode.IsLetter(r) }
func isNumber(r rune) bool { return unicode.IsNumber(r) }
func isSpace(r rune) bool  { return unicode.IsSpace(r) }
func isNL(r rune) bool     { return r == '\r' || r == '\n' }

// pretokenize splits the way the Qwen2 pattern does.
func pretokenize(s string) []string {
	var out []string
	rs := []rune(s)
	n := len(rs)
	i := 0
	for i < n {
		start := i
		r := rs[i]
		// contractions: 's 't 're 've 'm 'll 'd (case-insensitive)
		if r == '\'' && i+1 < n {
			rest := strings.ToLower(string(rs[i+1 : min(i+3, n)]))
			for _, c := range []string{"re", "ve", "ll", "s", "t", "m", "d"} {
				if strings.HasPrefix(rest, c) {
					i += 1 + len(c)
					goto emit
				}
			}
		}
		// [^\r\n\p{L}\p{N}]?\p{L}+
		if isLetter(r) || (!isNL(r) && !isNumber(r) && i+1 < n && isLetter(rs[i+1])) {
			if !isLetter(r) {
				i++
			}
			for i < n && isLetter(rs[i]) {
				i++
			}
			goto emit
		}
		// \p{N}  (one digit)
		if isNumber(r) {
			i++
			goto emit
		}
		// ` ?[^\s\p{L}\p{N}]+[\r\n]*`
		if !isSpace(r) || (r == ' ' && i+1 < n && !isSpace(rs[i+1]) && !isLetter(rs[i+1]) && !isNumber(rs[i+1])) {
			if r == ' ' {
				i++
			}
			for i < n && !isSpace(rs[i]) && !isLetter(rs[i]) && !isNumber(rs[i]) {
				i++
			}
			for i < n && isNL(rs[i]) {
				i++
			}
			goto emit
		}
		// whitespace: \s*[\r\n]+ | \s+(?!\S) | \s+
		{
			j := i
			for j < n && isSpace(rs[j]) {
				j++
			}
			// find the last newline in the run
			lastNL := -1
			for k := i; k < j; k++ {
				if isNL(rs[k]) {
					lastNL = k
				}
			}
			if lastNL >= 0 {
				i = lastNL + 1 // \s*[\r\n]+
				goto emit
			}
			if j < n {
				// \s+(?!\S): leave the last space to join the next token
				if j-i > 1 {
					i = j - 1
				} else {
					i = j
				}
			} else {
				i = j
			}
		}
	emit:
		if i == start {
			i++
		}
		out = append(out, string(rs[start:i]))
	}
	return out
}

func (t *bpeTokenizer) bpeCount(chunk string) int {
	if v, ok := t.cache.Load(chunk); ok {
		return v.(int)
	}
	// bytes -> the unicode alphabet the vocabulary is written in
	syms := make([]string, 0, len(chunk))
	for i := 0; i < len(chunk); i++ {
		syms = append(syms, t.byteEnc[chunk[i]])
	}
	for len(syms) > 1 {
		bestRank, bestAt := -1, -1
		for i := 0; i+1 < len(syms); i++ {
			if r, ok := t.ranks[[2]string{syms[i], syms[i+1]}]; ok && (bestRank < 0 || r < bestRank) {
				bestRank, bestAt = r, i
			}
		}
		if bestAt < 0 {
			break
		}
		merged := syms[bestAt] + syms[bestAt+1]
		syms = append(syms[:bestAt], append([]string{merged}, syms[bestAt+2:]...)...)
	}
	n := 0
	for _, s := range syms {
		if _, ok := t.vocab[s]; ok {
			n++
		} else {
			// unknown symbol: count its bytes, the way a byte-fallback would
			n += utf8.RuneCountInString(s)
		}
	}
	t.cache.Store(chunk, n)
	return n
}

func countTokens(text string) int {
	return genTokenizer().Count(text)
}
