package main

import (
	"io"
	"os"
	"strings"
	"syscall"
	"time"
	"unsafe"
)

// Shared plumbing for updates: a byte-rate cap so an update never takes the
// whole line, and a machine-wide lock so two updates cannot run at once.

// updateRate is the cap in bytes per second (20 Mbit/s by default).
func updateRate() float64 {
	if mbit := envFloat("LLMASH_UPDATE_MBPS", 20); mbit > 0 {
		return mbit * 1000 * 1000 / 8
	}
	return 0
}

// A pacer holds the average at the cap by measuring against a schedule
// rather than sleeping a fixed amount per chunk: Windows' timer granularity is
// around 15 ms, so a per-chunk sleep overshoots badly, while a schedule
// self-corrects on the next chunk.
type limiter struct {
	rate  float64
	start time.Time
	sent  int64
}

func newLimiter(rate float64) *limiter {
	return &limiter{rate: rate, start: time.Now()}
}

func (l *limiter) take(n int) {
	if l.rate <= 0 {
		return
	}
	l.sent += int64(n)
	ahead := float64(l.sent)/l.rate - time.Since(l.start).Seconds()
	if ahead > 0.002 {
		time.Sleep(time.Duration(ahead * float64(time.Second)))
	}
}

type throttledReader struct {
	r io.Reader
	l *limiter
}

func (t *throttledReader) Read(p []byte) (int, error) {
	if len(p) > 64<<10 {
		p = p[:64<<10]
	}
	n, err := t.r.Read(p)
	if n > 0 {
		t.l.take(n)
	}
	return n, err
}

// throttle caps how fast a body is read.
func throttle(r io.Reader) io.Reader {
	rate := updateRate()
	if rate <= 0 {
		return r
	}
	return &throttledReader{r: r, l: newLimiter(rate)}
}

// ---------------------------------------------------------------- the lock

var (
	procCreateMutexW        = kernel32.NewProc("CreateMutexW")
	procReleaseMutex        = kernel32.NewProc("ReleaseMutex")
	procWaitForSingleObject = kernel32.NewProc("WaitForSingleObject")
)

const errAlreadyExists = 183

// takeUpdateLock holds a named mutex for as long as this update runs, so a
// second `llmash update` on the same machine stops instead of racing it.
func takeUpdateLock() (func(), bool) {
	h, _, err := procCreateMutexW.Call(0, 1, uintptr(unsafe.Pointer(utf16(`Local\llmash-update`))))
	if h == 0 {
		return func() {}, true // no mutex available: do not block the update
	}
	if errno, ok := err.(syscall.Errno); ok && uintptr(errno) == errAlreadyExists {
		// Someone owns it. Give it a moment in case they are on the way out.
		if r, _, _ := procWaitForSingleObject.Call(h, 2000); r != 0 {
			procCloseHandle.Call(h)
			return func() {}, false
		}
	}
	return func() {
		procReleaseMutex.Call(h)
		procCloseHandle.Call(h)
	}, true
}

// ------------------------------------------------------------- json on disk

// readJSONFile reads a file PowerShell may have written, which means it may
// start with a byte-order mark that encoding/json refuses.
func readJSONFile(path string) ([]byte, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return []byte(strings.TrimPrefix(string(b), "\ufeff")), nil
}
