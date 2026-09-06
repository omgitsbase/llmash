package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

var host = func() string {
	h := os.Getenv("OLLAMA_HOST")
	if h == "" {
		h = "http://127.0.0.1:11434"
	}
	if !strings.HasPrefix(h, "http") {
		h = "http://" + h
	}
	return strings.TrimRight(h, "/")
}()

var client = &http.Client{Transport: &http.Transport{DisableKeepAlives: true}}

func request(ctx context.Context, method, path string, body any) (*http.Response, error) {
	var rd io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		rd = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, host+path, rd)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	return client.Do(req)
}

func call(method, path string, body any, timeout time.Duration) (*http.Response, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	r, err := request(ctx, method, path, body)
	if err != nil {
		cancel()
		return nil, err
	}
	// the body must be drained before the timeout matters again
	data, rerr := io.ReadAll(r.Body)
	r.Body.Close()
	cancel()
	if rerr != nil {
		return nil, rerr
	}
	r.Body = io.NopCloser(bytes.NewReader(data))
	return r, nil
}

func readAll(r *http.Response) ([]byte, error) {
	defer r.Body.Close()
	return io.ReadAll(r.Body)
}

func callJSON(method, path string, body any, timeout time.Duration) (map[string]any, *http.Response, error) {
	r, err := call(method, path, body, timeout)
	if err != nil {
		return nil, nil, err
	}
	data, _ := io.ReadAll(r.Body)
	var d map[string]any
	if len(bytes.TrimSpace(data)) > 0 {
		if err := json.Unmarshal(data, &d); err != nil {
			return nil, r, fmt.Errorf("not JSON: %s", strings.TrimSpace(string(data)))
		}
	}
	if d == nil {
		d = map[string]any{}
	}
	return d, r, nil
}

// One ndjson line at a time, as the server sends them.
func stream(ctx context.Context, path string, body any, fn func(map[string]any) bool) error {
	r, err := request(ctx, "POST", path, body)
	if err != nil {
		return err
	}
	defer r.Body.Close()
	sc := bufio.NewScanner(r.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var ev map[string]any
		if err := json.Unmarshal(line, &ev); err != nil {
			return errNotNDJSON
		}
		if !fn(ev) {
			return nil
		}
	}
	return sc.Err()
}

var errNotNDJSON = fmt.Errorf("the server sent a response that isn't NDJSON")

// Two tries: the server can be briefly busy loading a model, and a single
// short timeout makes a working server look dead.
func up() bool {
	for _, wait := range []time.Duration{4 * time.Second, 8 * time.Second} {
		if _, err := call("GET", "/api/version", nil, wait); err == nil {
			return true
		}
		time.Sleep(300 * time.Millisecond)
	}
	return false
}

func needServer() {
	if up() {
		return
	}
	fmt.Fprintf(os.Stderr, "llmash isn't running at %s.\nStart it with:  %s serve\n", host, prog)
	exit(1)
}

func str(m map[string]any, k string) string {
	if v, ok := m[k]; ok && v != nil {
		switch t := v.(type) {
		case string:
			return t
		default:
			return fmt.Sprint(t)
		}
	}
	return ""
}

func num(m map[string]any, k string) float64 {
	switch v := m[k].(type) {
	case float64:
		return v
	case int:
		return float64(v)
	case int64:
		return float64(v)
	case float32:
		return float64(v)
	}
	return 0
}

func sub(m map[string]any, k string) map[string]any {
	if v, ok := m[k].(map[string]any); ok {
		return v
	}
	return map[string]any{}
}

func list(m map[string]any, k string) []any {
	if v, ok := m[k].([]any); ok {
		return v
	}
	return nil
}
