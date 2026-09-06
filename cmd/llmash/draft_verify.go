package main

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

// llama.cpp accepts a drafter only if the vocabularies match and its encoder is
// shaped for the target's hidden size. Both are in the GGUF header, so the
// answer costs a few kilobytes instead of a load.

type ggufSpec struct {
	Arch    string
	Embed   int64   // <arch>.embedding_length
	Blocks  int64   // <arch>.block_count
	Vocab   int64   // number of tokens in the vocabulary
	Layers  []int64 // <arch>.target_layers, the layers a drafter reads from
	Enc     int64   // fc.weight's input width = len(Layers) * target Embed
	Tensors int64
	partial bool // the read ended early, so a zero field means "not reached"
}

func (s ggufSpec) hidden() bool {
	return s.Arch == "eagle3" || s.Arch == "dflash" || s.Arch == "dspark"
}

// Tensor shapes sit after the vocabulary, so wantTensors costs a few megabytes.
func scanGGUF(r io.Reader, wantTensors bool) (ggufSpec, error) {
	var s ggufSpec
	g := &ggufReader{r: bufio.NewReaderSize(r, 1<<20)}
	magic := make([]byte, 4)
	if _, err := io.ReadFull(g.r, magic); err != nil || string(magic) != "GGUF" {
		return s, fmt.Errorf("not a GGUF file")
	}
	ver, err := g.u32()
	if err != nil {
		return s, err
	}
	if ver < 2 || ver > 3 {
		return s, fmt.Errorf("GGUF version %d, which this llama.cpp does not read", ver)
	}
	nTensors, err := g.u64()
	if err != nil {
		return s, err
	}
	nKV, err := g.u64()
	if err != nil {
		return s, err
	}
	s.Tensors = int64(nTensors)
	if s.Tensors == 0 {
		return s, fmt.Errorf("no tensors in the file")
	}

	s.partial = true
	for i := uint64(0); i < nKV; i++ {
		k, err := g.str()
		if err != nil {
			return s, nil
		}
		t, err := g.u32()
		if err != nil {
			return s, nil
		}
		if t != 9 {
			v, err := g.value(t)
			if err != nil {
				return s, nil
			}
			switch {
			case k == "general.architecture":
				s.Arch, _ = v.(string)
			case strings.HasSuffix(k, ".embedding_length"):
				s.Embed, _ = v.(int64)
			case strings.HasSuffix(k, ".block_count"):
				s.Blocks, _ = v.(int64)
			case strings.HasSuffix(k, ".vocab_size"):
				if n, ok := v.(int64); ok && s.Vocab == 0 {
					s.Vocab = n
				}
			}
			continue
		}
		// An array's length comes before its contents, which is all the
		// vocabulary count is.
		et, err := g.u32()
		if err != nil {
			return s, nil
		}
		n, err := g.u64()
		if err != nil {
			return s, nil
		}
		if k == "tokenizer.ggml.tokens" {
			s.Vocab = int64(n)
		}
		if strings.HasSuffix(k, ".target_layers") && n <= 64 && et >= 4 && et <= 5 {
			for j := uint64(0); j < n; j++ {
				v, err := g.value(et)
				if err != nil {
					return s, nil
				}
				id, _ := v.(int64)
				s.Layers = append(s.Layers, id)
			}
			continue
		}
		if err := g.skipArray(et, n); err != nil {
			return s, nil
		}
	}
	if !wantTensors {
		s.partial = false
		return s, nil
	}

	for i := uint64(0); i < nTensors; i++ {
		name, err := g.str()
		if err != nil {
			return s, nil
		}
		nd, err := g.u32()
		if err != nil {
			return s, nil
		}
		var dims []int64
		for d := uint32(0); d < nd; d++ {
			v, err := g.u64()
			if err != nil {
				return s, nil
			}
			dims = append(dims, int64(v))
		}
		if _, err := g.u32(); err != nil { // ggml type
			return s, nil
		}
		if _, err := g.u64(); err != nil { // offset
			return s, nil
		}
		if (name == "fc.weight" || strings.HasSuffix(name, ".fc.weight")) && len(dims) > 0 {
			s.Enc = dims[0]
		}
	}
	s.partial = false
	return s, nil
}

func specOfFile(path string) (ggufSpec, error) {
	f, err := os.Open(path)
	if err != nil {
		return ggufSpec{}, err
	}
	defer f.Close()
	return scanGGUF(f, true)
}

// A problem reaching the hub, as opposed to a verdict on the file.
var errUnreadable = errors.New("could not read the header")

func unreadable(format string, a ...any) error {
	return fmt.Errorf("%w: %s", errUnreadable, fmt.Sprintf(format, a...))
}

func specOfURL(url string) (ggufSpec, error) {
	var last error
	for _, window := range []int64{1 << 20, 24 << 20} {
		req, err := http.NewRequest("GET", url, nil)
		if err != nil {
			return ggufSpec{}, err
		}
		req.Header.Set("User-Agent", "llmash")
		req.Header.Set("Range", fmt.Sprintf("bytes=0-%d", window-1))
		resp, err := (&http.Client{Timeout: 90 * time.Second}).Do(req)
		if err != nil {
			return ggufSpec{}, unreadable("%v", err)
		}
		if resp.StatusCode != 200 && resp.StatusCode != 206 {
			resp.Body.Close()
			switch resp.StatusCode {
			case 401, 403:
				return ggufSpec{}, unreadable("the hub wants an account for this file")
			case 429:
				return ggufSpec{}, unreadable("the hub is rate limiting this address")
			default:
				return ggufSpec{}, unreadable("the hub answered %d", resp.StatusCode)
			}
		}
		s, err := scanGGUF(io.LimitReader(resp.Body, window), true)
		resp.Body.Close()
		if err != nil {
			return s, err
		}
		if !s.partial {
			return s, nil
		}
		last = unreadable("its header is longer than %s", humanBytes(window))
	}
	return ggufSpec{partial: true}, last
}

// Why a drafter cannot serve a target, or nil if it can.
func pairs(target, draft ggufSpec) error {
	if draft.Arch == "" {
		return fmt.Errorf("no architecture in its header")
	}
	if target.Vocab > 0 && draft.Vocab > 0 {
		if diff := target.Vocab - draft.Vocab; diff > 128 || diff < -128 {
			return fmt.Errorf("its vocabulary has %d tokens against this model's %d",
				draft.Vocab, target.Vocab)
		}
	}
	if !draft.hidden() {
		return nil
	}
	if len(draft.Layers) == 0 {
		return fmt.Errorf("it is a %s drafter but names no layers to read", draft.Arch)
	}
	if draft.Arch == "eagle3" && len(draft.Layers) != 3 {
		return fmt.Errorf("EAGLE-3 reads exactly 3 layers, this one names %d", len(draft.Layers))
	}
	if target.Blocks > 0 {
		for _, id := range draft.Layers {
			if id < 0 || id >= target.Blocks {
				return fmt.Errorf("it reads layer %d, and this model has %d", id, target.Blocks)
			}
		}
	}
	if draft.Enc > 0 && target.Embed > 0 {
		want := int64(len(draft.Layers)) * target.Embed
		if draft.Enc != want {
			return fmt.Errorf("its encoder takes %d values where this model gives %d",
				draft.Enc, want)
		}
	}
	return nil
}

func fitsTarget(m *Model, c draftCand) error {
	target, err := specOfFile(m.GGUF)
	if err != nil {
		return nil
	}
	draft, err := specOfURL(hubDownloadURL(c.Repo, c.File))
	if errors.Is(err, errUnreadable) {
		return nil
	}
	if err != nil {
		return err
	}
	return pairs(target, draft)
}
