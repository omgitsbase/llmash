package main

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
)

// The key-value block of a GGUF header. Only scalars are kept; arrays are
// stepped over exactly (the tokenizer's vocabulary alone is ~300k strings).

var quantNames = map[int]string{
	0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
	10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S",
	15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS",
	20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S",
	25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M",
	30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0",
}

type ggufReader struct {
	r   *bufio.Reader
	buf [8]byte
}

func (g *ggufReader) u32() (uint32, error) {
	if _, err := io.ReadFull(g.r, g.buf[:4]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(g.buf[:4]), nil
}

func (g *ggufReader) u64() (uint64, error) {
	if _, err := io.ReadFull(g.r, g.buf[:8]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(g.buf[:8]), nil
}

func (g *ggufReader) str() (string, error) {
	n, err := g.u64()
	if err != nil {
		return "", err
	}
	if n > 64<<20 {
		return "", errors.New("gguf string too long")
	}
	b := make([]byte, n)
	if _, err := io.ReadFull(g.r, b); err != nil {
		return "", err
	}
	return string(b), nil
}

func (g *ggufReader) skip(n int64) error {
	_, err := g.r.Discard(int(n))
	return err
}

// value reads one typed value; scalars come back as string/int64/float64/bool,
// arrays as nil after being skipped exactly.
func (g *ggufReader) value(t uint32) (any, error) {
	switch t {
	case 0:
		if _, err := io.ReadFull(g.r, g.buf[:1]); err != nil {
			return nil, err
		}
		return int64(g.buf[0]), nil
	case 1:
		if _, err := io.ReadFull(g.r, g.buf[:1]); err != nil {
			return nil, err
		}
		return int64(int8(g.buf[0])), nil
	case 2:
		if _, err := io.ReadFull(g.r, g.buf[:2]); err != nil {
			return nil, err
		}
		return int64(binary.LittleEndian.Uint16(g.buf[:2])), nil
	case 3:
		if _, err := io.ReadFull(g.r, g.buf[:2]); err != nil {
			return nil, err
		}
		return int64(int16(binary.LittleEndian.Uint16(g.buf[:2]))), nil
	case 4:
		v, err := g.u32()
		return int64(v), err
	case 5:
		v, err := g.u32()
		return int64(int32(v)), err
	case 6:
		v, err := g.u32()
		return float64(math.Float32frombits(v)), err
	case 7:
		if _, err := io.ReadFull(g.r, g.buf[:1]); err != nil {
			return nil, err
		}
		return g.buf[0] != 0, nil
	case 8:
		return g.str()
	case 9:
		et, err := g.u32()
		if err != nil {
			return nil, err
		}
		n, err := g.u64()
		if err != nil {
			return nil, err
		}
		return nil, g.skipArray(et, n)
	case 10:
		v, err := g.u64()
		return int64(v), err
	case 11:
		v, err := g.u64()
		return int64(v), err
	case 12:
		v, err := g.u64()
		return math.Float64frombits(v), err
	}
	return nil, fmt.Errorf("unknown gguf type %d", t)
}

// The skip has to be exact: land one byte off and every key after the array
// is read from the middle of it.
func (g *ggufReader) skipArray(et uint32, n uint64) error {
	fixed := map[uint32]int64{0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
	if w, ok := fixed[et]; ok {
		return g.skip(w * int64(n))
	}
	if et == 8 {
		for i := uint64(0); i < n; i++ {
			ln, err := g.u64()
			if err != nil {
				return err
			}
			if err := g.skip(int64(ln)); err != nil {
				return err
			}
		}
		return nil
	}
	for i := uint64(0); i < n; i++ {
		if _, err := g.value(et); err != nil {
			return err
		}
	}
	return nil
}

// readGGUFMeta pulls the header's scalar key-values; never the tensor data.
func readGGUFMeta(path string) map[string]any {
	f, err := os.Open(path)
	if err != nil {
		return map[string]any{}
	}
	defer f.Close()
	return ggufMetaFrom(bufio.NewReaderSize(f, 1<<20))
}

// ggufMetaFrom reads the key-values it can reach and stops at the first one
// it cannot, so a truncated header (the first few MB of a remote file) still
// yields everything that precedes the tokenizer.
func ggufMetaFrom(r *bufio.Reader) map[string]any {
	out := map[string]any{}
	g := &ggufReader{r: r}
	magic := make([]byte, 4)
	if _, err := io.ReadFull(g.r, magic); err != nil || string(magic) != "GGUF" {
		return out
	}
	ver, err := g.u32()
	if err != nil {
		return out
	}
	nTensors, err := g.u64()
	if err != nil {
		return out
	}
	nKV, err := g.u64()
	if err != nil {
		return out
	}
	for i := uint64(0); i < nKV; i++ {
		k, err := g.str()
		if err != nil {
			break
		}
		t, err := g.u32()
		if err != nil {
			break
		}
		v, err := g.value(t)
		if err != nil {
			break
		}
		if v != nil {
			out[k] = v
		}
	}
	out["_n_tensors"] = int64(nTensors)
	out["_version"] = int64(ver)
	return out
}

func metaStr(m map[string]any, k string) string {
	if v, ok := m[k]; ok && v != nil {
		switch t := v.(type) {
		case string:
			return t
		case bool:
			if t {
				return "true"
			}
			return "false"
		default:
			return fmt.Sprint(t)
		}
	}
	return ""
}

func metaInt(m map[string]any, k string) (int64, bool) {
	switch t := m[k].(type) {
	case int64:
		return t, true
	case float64:
		return int64(t), true
	}
	return 0, false
}

// hasMTP: true when the GGUF carries multi-token-prediction heads, detected by
// tensor name (the metadata flag is not written consistently by converters).
var mtpCache = map[string]bool{}

func hasMTP(path string) bool {
	if v, ok := mtpCache[path]; ok {
		return v
	}
	found := false
	func() {
		f, err := os.Open(path)
		if err != nil {
			return
		}
		defer f.Close()
		g := &ggufReader{r: bufio.NewReaderSize(f, 1<<20)}
		magic := make([]byte, 4)
		if _, err := io.ReadFull(g.r, magic); err != nil || string(magic) != "GGUF" {
			return
		}
		if _, err := g.u32(); err != nil {
			return
		}
		nTensors, err := g.u64()
		if err != nil {
			return
		}
		nKV, err := g.u64()
		if err != nil {
			return
		}
		for i := uint64(0); i < nKV; i++ {
			if _, err := g.str(); err != nil {
				return
			}
			t, err := g.u32()
			if err != nil {
				return
			}
			if _, err := g.value(t); err != nil {
				return
			}
		}
		for i := uint64(0); i < nTensors; i++ {
			name, err := g.str()
			if err != nil {
				return
			}
			if strings.Contains(name, "nextn") || strings.Contains(strings.ToLower(name), "mtp") {
				found = true
				return
			}
			nd, err := g.u32()
			if err != nil {
				return
			}
			if err := g.skip(8*int64(nd) + 4 + 8); err != nil {
				return
			}
		}
	}()
	mtpCache[path] = found
	if found {
		logf("%s: MTP heads present, speculative decoding on", baseName(path))
	}
	return found
}

func prettyParams(n int64) string {
	switch {
	case n >= 1e12:
		return fmt.Sprintf("%.1fT", float64(n)/1e12)
	case n >= 1e9:
		return fmt.Sprintf("%.1fB", float64(n)/1e9)
	case n >= 1e6:
		return fmt.Sprintf("%.2fM", float64(n)/1e6)
	}
	return fmt.Sprint(n)
}
