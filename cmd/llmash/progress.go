package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Ollama's progress display: a set of lines (spinners and bars) on stderr,
// redrawn in place ten times a second.

const (
	defaultTermWidth  = 80
	defaultTermHeight = 24
)

type progState interface{ String() string }

type progress struct {
	mu       sync.Mutex
	w        *bufio.Writer
	pos      int
	states   []progState
	stopOnce sync.Once
	done     chan struct{}
}

func newProgress(w io.Writer) *progress {
	p := &progress{w: bufio.NewWriter(w), done: make(chan struct{})}
	go p.run()
	return p
}

func (p *progress) add(state progState) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.states = append(p.states, state)
}

func (p *progress) halt() (bool, int) {
	var stopped bool
	p.stopOnce.Do(func() {
		close(p.done)
		stopped = true
	})
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, s := range p.states {
		if sp, ok := s.(*progSpinner); ok {
			sp.stop()
		}
	}
	if stopped {
		p.renderLocked()
	}
	return stopped, p.pos
}

func (p *progress) stop() bool {
	stopped, _ := p.halt()
	if stopped {
		fmt.Fprint(p.w, "\n")
		p.w.Flush()
	}
	return stopped
}

func (p *progress) stopAndClear() bool {
	defer p.w.Flush()
	fmt.Fprint(p.w, "\033[?25l")
	defer fmt.Fprint(p.w, "\033[?25h")
	stopped, pos := p.halt()
	if stopped {
		for i := 0; i < pos; i++ {
			if i > 0 {
				fmt.Fprint(p.w, "\033[A")
			}
			fmt.Fprint(p.w, "\033[2K\033[1G")
		}
	}
	return stopped
}

func (p *progress) render() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.renderLocked()
}

func (p *progress) renderLocked() {
	defer p.w.Flush()
	fmt.Fprint(p.w, "\033[?2026h")
	defer fmt.Fprint(p.w, "\033[?2026l")
	fmt.Fprint(p.w, "\033[?25l")
	defer fmt.Fprint(p.w, "\033[?25h")
	for i := 0; i < p.pos-1; i++ {
		fmt.Fprint(p.w, "\033[A")
	}
	fmt.Fprint(p.w, "\033[1G")
	height := defaultTermHeight
	max := len(p.states)
	if max > height {
		max = height
	}
	for i := len(p.states) - max; i < len(p.states); i++ {
		fmt.Fprint(p.w, p.states[i].String(), "\033[K")
		if i < len(p.states)-1 {
			fmt.Fprint(p.w, "\n")
		}
	}
	p.pos = len(p.states)
}

func (p *progress) run() {
	t := time.NewTicker(100 * time.Millisecond)
	defer t.Stop()
	for {
		select {
		case <-p.done:
			return
		case <-t.C:
			p.render()
		}
	}
}

// ------------------------------------------------------------- spinner

type progSpinner struct {
	message      atomic.Value
	mu           sync.Mutex
	messageWidth int
	parts        []string
	value        int
	stopped      time.Time
	stopOnce     sync.Once
	done         chan struct{}
}

func newSpinner(message string) *progSpinner {
	s := &progSpinner{
		parts: []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"},
		done:  make(chan struct{}),
	}
	s.message.Store(message)
	go s.run()
	return s
}

func (s *progSpinner) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	var sb strings.Builder
	if m, ok := s.message.Load().(string); ok && len(m) > 0 {
		m = strings.TrimSpace(m)
		if s.messageWidth > 0 && len(m) > s.messageWidth {
			m = m[:s.messageWidth]
		}
		sb.WriteString(m)
		if pad := s.messageWidth - sb.Len(); pad > 0 {
			sb.WriteString(strings.Repeat(" ", pad))
		}
		sb.WriteString(" ")
	}
	if s.stopped.IsZero() {
		sb.WriteString(s.parts[s.value])
		sb.WriteString(" ")
	}
	return sb.String()
}

func (s *progSpinner) run() {
	t := time.NewTicker(100 * time.Millisecond)
	defer t.Stop()
	for {
		select {
		case <-s.done:
			return
		case <-t.C:
			s.mu.Lock()
			s.value = (s.value + 1) % len(s.parts)
			s.mu.Unlock()
		}
	}
}

func (s *progSpinner) stop() {
	s.mu.Lock()
	if s.stopped.IsZero() {
		s.stopped = time.Now()
	}
	s.mu.Unlock()
	s.stopOnce.Do(func() { close(s.done) })
}

// ----------------------------------------------------------------- bar

type progBar struct {
	mu           sync.Mutex
	message      string
	messageWidth int
	maxValue     int64
	initialValue int64
	currentValue int64
	started      time.Time
	stopped      time.Time
	maxBuckets   int
	buckets      []progBucket
}

type progBucket struct {
	updated time.Time
	value   int64
}

func newBar(message string, maxValue, initialValue int64) *progBar {
	b := &progBar{message: message, messageWidth: -1, maxValue: maxValue,
		initialValue: initialValue, currentValue: initialValue, started: time.Now(), maxBuckets: 10}
	if initialValue >= maxValue {
		b.stopped = time.Now()
	}
	return b
}

// Two units at most: "1h20m", "45s".
func formatDuration(d time.Duration) string {
	switch {
	case d >= 100*time.Hour:
		return "99h+"
	case d >= time.Hour:
		return fmt.Sprintf("%dh%dm", int(d.Hours()), int(d.Minutes())%60)
	default:
		return d.Round(time.Second).String()
	}
}

func (b *progBar) String() string {
	w := termWidthOf(os.Stderr)
	if w <= 0 {
		w = defaultTermWidth
	}
	b.mu.Lock()
	defer b.mu.Unlock()

	var pre strings.Builder
	if len(b.message) > 0 {
		m := strings.TrimSpace(b.message)
		if b.messageWidth > 0 && len(m) > b.messageWidth {
			m = m[:b.messageWidth]
		}
		pre.WriteString(m)
		if pad := b.messageWidth - pre.Len(); pad > 0 {
			pre.WriteString(strings.Repeat(" ", pad))
		}
		pre.WriteString(" ")
	}
	fmt.Fprintf(&pre, "%3.0f%%", b.percent())

	var suf strings.Builder
	pad := func(s string, n int) {
		if k := n - len(s); k > 0 {
			suf.WriteString(strings.Repeat(" ", k))
		}
		suf.WriteString(s)
	}
	if b.stopped.IsZero() {
		pad(humanBytes(b.currentValue), 6)
		suf.WriteString("/")
		pad(humanBytes(b.maxValue), 6)
	} else {
		pad(humanBytes(b.maxValue), 6)
		suf.WriteString(strings.Repeat(" ", 7))
	}
	rate := b.rate()
	if b.stopped.IsZero() && rate > 0 {
		suf.WriteString("  ")
		pad(humanBytes(int64(rate)), 6)
		suf.WriteString("/s")
	} else {
		suf.WriteString(strings.Repeat(" ", 10))
	}
	if b.stopped.IsZero() && rate > 0 {
		suf.WriteString("  ")
		remaining := time.Duration(int64(float64(b.maxValue-b.currentValue)/rate)) * time.Second
		pad(formatDuration(remaining), 6)
	} else {
		suf.WriteString(strings.Repeat(" ", 8))
	}

	var mid strings.Builder
	f := w - pre.Len() - suf.Len() - 5
	if f < 0 {
		f = 0
	}
	n := int(float64(f) * b.percent() / 100)
	mid.WriteString(" ▕")
	if n > 0 {
		mid.WriteString(strings.Repeat("█", n))
	}
	if f-n > 0 {
		mid.WriteString(strings.Repeat(" ", f-n))
	}
	mid.WriteString("▏ ")
	return pre.String() + mid.String() + suf.String()
}

func (b *progBar) set(value int64) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if value >= b.maxValue {
		value = b.maxValue
	}
	b.currentValue = value
	if b.currentValue >= b.maxValue {
		b.stopped = time.Now()
	}
	if len(b.buckets) == 0 || time.Since(b.buckets[len(b.buckets)-1].updated) > time.Second {
		b.buckets = append(b.buckets, progBucket{updated: time.Now(), value: value})
		if len(b.buckets) > b.maxBuckets {
			b.buckets = b.buckets[1:]
		}
	}
}

func (b *progBar) percent() float64 {
	if b.maxValue > 0 {
		return float64(b.currentValue) / float64(b.maxValue) * 100
	}
	return 0
}

func (b *progBar) rate() float64 {
	var num, den float64
	if !b.stopped.IsZero() {
		num = float64(b.currentValue - b.initialValue)
		den = b.stopped.Sub(b.started).Round(time.Second).Seconds()
	} else if len(b.buckets) > 1 {
		first, last := b.buckets[0], b.buckets[len(b.buckets)-1]
		num = float64(last.value - first.value)
		den = last.updated.Sub(first.updated).Round(time.Second).Seconds()
	}
	if den == 0 {
		return 0
	}
	return num / den
}
