package main

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"
	"unicode"
)

// Ollama prints its tables with olekukonko/tablewriter configured for no
// borders, no header line, left alignment, no leading whitespace and a
// four-space column gap. This reproduces that layout exactly, including the
// wrap at 30 columns and the trailing padding on every row.

const twMaxWidth = 30
const twPad = "    "

type table struct {
	header []string
	rows   [][]string
	wrapAt int
}

func newTable(header ...string) *table {
	return &table{header: header, wrapAt: twMaxWidth}
}

func (t *table) add(cells ...string) { t.rows = append(t.rows, cells) }

// One cell becomes one or more display lines; the widest of them sets the
// column width.
func (t *table) split(s string) []string {
	raw := strings.Split(s, "\n")
	max := 0
	for _, ln := range raw {
		if w := dispWidth(ln); w > max {
			max = w
		}
	}
	if max <= t.wrapAt {
		return raw
	}
	var out []string
	for i, para := range raw {
		if i > 0 {
			out = append(out, " ")
		}
		out = append(out, wrapString(para, t.wrapAt)...)
	}
	return out
}

func (t *table) String() string {
	cols := len(t.header)
	for _, r := range t.rows {
		if len(r) > cols {
			cols = len(r)
		}
	}
	widths := make([]int, cols)
	head := make([][]string, cols)
	body := make([][][]string, len(t.rows))
	measure := func(lines []string, c int) {
		for _, ln := range lines {
			if w := dispWidth(ln); w > widths[c] {
				widths[c] = w
			}
		}
	}
	for c := 0; c < cols; c++ {
		cell := ""
		if c < len(t.header) {
			cell = twTitle(t.header[c])
		}
		head[c] = t.split(cell)
		measure(head[c], c)
	}
	for i, r := range t.rows {
		body[i] = make([][]string, cols)
		for c := 0; c < cols; c++ {
			cell := ""
			if c < len(r) {
				cell = r[c]
			}
			body[i][c] = t.split(cell)
			measure(body[i][c], c)
		}
	}

	var sb strings.Builder
	if len(t.header) > 0 {
		lines := 1
		for c := range head {
			if n := len(head[c]); n > lines {
				lines = n
			}
		}
		for x := 0; x < lines; x++ {
			for c := 0; c < cols; c++ {
				sb.WriteString(padRight(lineAt(head[c], x), widths[c]))
				if c == cols-1 {
					sb.WriteString(" ")
				} else {
					sb.WriteString(twPad)
				}
			}
			sb.WriteString("\n")
		}
	}
	for i := range t.rows {
		lines := 1
		for c := range body[i] {
			if n := len(body[i][c]); n > lines {
				lines = n
			}
		}
		for x := 0; x < lines; x++ {
			for c := 0; c < cols; c++ {
				sb.WriteString(padRight(lineAt(body[i][c], x), widths[c]))
				sb.WriteString(twPad)
			}
			sb.WriteString("\n")
		}
	}
	return sb.String()
}

func lineAt(lines []string, i int) string {
	if i < len(lines) {
		return lines[i]
	}
	return "  "
}

func padRight(s string, width int) string {
	if n := width - dispWidth(s); n > 0 {
		return s + strings.Repeat(" ", n)
	}
	return s
}

// Greedy fill, and a word longer than the limit widens the column instead of
// being broken.
func wrapString(s string, lim int) []string {
	words := strings.Fields(strings.ReplaceAll(s, "\n", " "))
	for _, w := range words {
		if n := dispWidth(w); n > lim {
			lim = n
		}
	}
	var lines []string
	var cur []string
	n := 0
	for _, w := range words {
		width := dispWidth(w)
		if n > 0 && n+1+width > lim {
			lines = append(lines, strings.Join(cur, " "))
			cur, n = nil, 0
		}
		if n > 0 {
			n++
		}
		cur = append(cur, w)
		n += width
	}
	if len(cur) > 0 {
		lines = append(lines, strings.Join(cur, " "))
	}
	if len(lines) == 0 {
		return []string{""}
	}
	return lines
}

// Header cells: underscores and separating dots become spaces, then upper case.
func twTitle(name string) string {
	rs := []rune(name)
	for i, r := range rs {
		switch r {
		case '_':
			rs[i] = ' '
		case '.':
			if (i != 0 && !unicode.IsDigit(rs[i-1])) || (i != len(rs)-1 && !unicode.IsDigit(rs[i+1])) {
				rs[i] = ' '
			}
		}
	}
	out := strings.ToUpper(strings.TrimSpace(string(rs)))
	if out == "" && len(name) > 0 {
		return " "
	}
	return out
}

// Terminal cell width: two columns for the wide CJK and emoji blocks, none
// for combining marks.
func dispWidth(s string) int {
	n := 0
	for _, r := range s {
		switch {
		case r == 0:
		case unicode.Is(unicode.Mn, r) || unicode.Is(unicode.Me, r):
		case r >= 0x1100 && (r <= 0x115f ||
			r == 0x2329 || r == 0x232a ||
			(r >= 0x2e80 && r <= 0xa4cf && r != 0x303f) ||
			(r >= 0xac00 && r <= 0xd7a3) ||
			(r >= 0xf900 && r <= 0xfaff) ||
			(r >= 0xfe30 && r <= 0xfe6f) ||
			(r >= 0xff00 && r <= 0xff60) ||
			(r >= 0xffe0 && r <= 0xffe6) ||
			(r >= 0x1f300 && r <= 0x1f64f) ||
			(r >= 0x1f900 && r <= 0x1f9ff) ||
			(r >= 0x20000 && r <= 0x3fffd)):
			n += 2
		default:
			n++
		}
	}
	return n
}

// ------------------------------------------------ ollama's format package

func humanBytes(b int64) string {
	const kb, mb, gb, tb = 1000, 1000 * 1000, 1000 * 1000 * 1000, 1000 * 1000 * 1000 * 1000
	var value float64
	var unit string
	switch {
	case b >= tb:
		value, unit = float64(b)/tb, "TB"
	case b >= gb:
		value, unit = float64(b)/gb, "GB"
	case b >= mb:
		value, unit = float64(b)/mb, "MB"
	case b >= kb:
		value, unit = float64(b)/kb, "KB"
	default:
		return fmt.Sprintf("%d B", b)
	}
	switch {
	case value >= 10:
		return fmt.Sprintf("%d %s", int(value), unit)
	case value != math.Trunc(value):
		return fmt.Sprintf("%.1f %s", value, unit)
	default:
		return fmt.Sprintf("%d %s", int(value), unit)
	}
}

func humanNumber(b uint64) string {
	switch {
	case b >= 1e9:
		n := float64(b) / 1e9
		if n == math.Floor(n) {
			return fmt.Sprintf("%.0fB", n)
		}
		return fmt.Sprintf("%.1fB", n)
	case b >= 1e6:
		n := float64(b) / 1e6
		if n == math.Floor(n) {
			return fmt.Sprintf("%.0fM", n)
		}
		return fmt.Sprintf("%.2fM", n)
	case b >= 1000:
		return fmt.Sprintf("%.0fK", float64(b)/1000)
	default:
		return strconv.FormatUint(b, 10)
	}
}

func humanDuration(d time.Duration) string {
	seconds := int(d.Seconds())
	switch {
	case seconds < 1:
		return "Less than a second"
	case seconds == 1:
		return "1 second"
	case seconds < 60:
		return fmt.Sprintf("%d seconds", seconds)
	}
	minutes := int(d.Minutes())
	switch {
	case minutes == 1:
		return "About a minute"
	case minutes < 60:
		return fmt.Sprintf("%d minutes", minutes)
	}
	hours := int(math.Round(d.Hours()))
	switch {
	case hours == 1:
		return "About an hour"
	case hours < 48:
		return fmt.Sprintf("%d hours", hours)
	case hours < 24*7*2:
		return fmt.Sprintf("%d days", hours/24)
	case hours < 24*30*2:
		return fmt.Sprintf("%d weeks", hours/24/7)
	case hours < 24*365*2:
		return fmt.Sprintf("%d months", hours/24/30)
	}
	return fmt.Sprintf("%d years", int(d.Hours())/24/365)
}

func humanTime(t time.Time, zero string) string {
	if t.IsZero() {
		return zero
	}
	delta := time.Since(t)
	if int(delta.Hours())/24/365 < -20 {
		return "Forever"
	} else if delta < 0 {
		return humanDuration(-delta) + " from now"
	}
	return humanDuration(delta) + " ago"
}

func humanTimeISO(s, zero string) string {
	t, err := time.Parse(time.RFC3339Nano, s)
	if err != nil {
		return zero
	}
	return humanTime(t, zero)
}
