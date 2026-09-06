package main

import (
	"bufio"
	"os"
	"strings"
	"testing"
)

// Drives the line editor from a byte stream (what the console sends in VT
// input mode) and checks the line it hands back.
func typeLine(t *testing.T, keys string) string {
	t.Helper()
	e := newEditor()
	e.hist = &history{enabled: false, limit: 100}
	e.in = bufio.NewReader(strings.NewReader(keys))

	old := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stdout = w
	done := make(chan struct{})
	go func() {
		io := make([]byte, 1<<16)
		for {
			if _, err := r.Read(io); err != nil {
				break
			}
		}
		close(done)
	}()
	line, _ := e.readline()
	w.Close()
	os.Stdout = old
	<-done
	return line
}

func TestEditorKeys(t *testing.T) {
	const (
		up    = "\x1b[A"
		down  = "\x1b[B"
		right = "\x1b[C"
		left  = "\x1b[D"
		home  = "\x1b[H"
		end   = "\x1b[F"
		del   = "\x1b[3~"
		cr    = "\r"
	)
	cases := []struct{ name, keys, want string }{
		{"plain", "hello" + cr, "hello"},
		{"backspace", "helllo\x7f" + cr, "helll"},
		{"ctrl-h backspace", "abc\b" + cr, "ab"},
		{"left insert", "helo" + left + "l" + cr, "hello"},
		{"home", "world" + home + "hello " + cr, "hello world"},
		{"end", "hello" + home + "say " + end + "!" + cr, "say hello!"},
		{"delete key", "hello" + home + del + cr, "ello"},
		{"ctrl-a ctrl-e", "world\x01hello \x05!" + cr, "hello world!"},
		{"ctrl-k", "hello world" + home + right + right + right + right + right + "\x0b" + cr, "hello"},
		{"ctrl-u", "hello world" + "\x15" + "bye" + cr, "bye"},
		{"ctrl-w", "hello big world\x17" + cr, "hello big "},
		{"alt-b alt-f", "one two\x1bb\x1bbX" + cr, "Xone two"},
		{"alt-backspace", "one two\x1b\x7f" + cr, "one "},
		{"right stops at end", "hi" + right + right + "!" + cr, "hi!"},
		{"tab is spaces", "a\tb" + cr, "a        b"},
		{"unicode", "héllo 世界" + cr, "héllo 世界"},
		{"ctrl-j multiline", "one\ntwo" + cr, "one\ntwo"},
	}
	for _, c := range cases {
		if got := typeLine(t, c.keys); got != c.want {
			t.Errorf("%s: got %q, want %q", c.name, got, c.want)
		}
	}
}

func TestEditorHistory(t *testing.T) {
	e := newEditor()
	e.hist = &history{enabled: false, limit: 100, lines: []string{"first", "second"}, pos: 2}
	old := os.Stdout
	devnull, _ := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
	os.Stdout = devnull
	defer func() { os.Stdout = old; devnull.Close() }()

	e.in = bufio.NewReader(strings.NewReader("\x1b[A\r"))
	if got, _ := e.readline(); got != "second" {
		t.Errorf("one up: got %q", got)
	}
	e.hist.pos = 2
	e.in = bufio.NewReader(strings.NewReader("\x1b[A\x1b[A\r"))
	if got, _ := e.readline(); got != "first" {
		t.Errorf("two up: got %q", got)
	}
	e.hist.pos = 2
	e.in = bufio.NewReader(strings.NewReader("\x1b[A\x1b[A\x1b[B\r"))
	if got, _ := e.readline(); got != "second" {
		t.Errorf("up up down: got %q", got)
	}
	e.hist.pos = 2
	e.in = bufio.NewReader(strings.NewReader("draft\x1b[A\x1b[B\r"))
	if got, _ := e.readline(); got != "draft" {
		t.Errorf("down restores the draft: got %q", got)
	}
}

func TestTableMatchesTablewriter(t *testing.T) {
	tb := newTable("NAME", "ID", "SIZE", "MODIFIED")
	tb.add("llama3.2:latest", "a80c4f17acd5", "2.0 GB", "2 days ago")
	tb.add("qwen3:8b", "500a1f067a9f", "5.2 GB", "3 weeks ago")
	want := "" +
		"NAME               ID              SIZE      MODIFIED    \n" +
		"llama3.2:latest    a80c4f17acd5    2.0 GB    2 days ago     \n" +
		"qwen3:8b           500a1f067a9f    5.2 GB    3 weeks ago    \n"
	if got := tb.String(); got != want {
		t.Errorf("table mismatch:\ngot:\n%q\nwant:\n%q", got, want)
	}
}

func TestHumanBytes(t *testing.T) {
	cases := []struct {
		in   int64
		want string
	}{
		{999, "999 B"},
		{1000, "1 KB"},
		{1500, "1.5 KB"},
		{2019393189, "2.0 GB"},
		{17559196320, "17 GB"},
	}
	for _, c := range cases {
		if got := humanBytes(c.in); got != c.want {
			t.Errorf("humanBytes(%d) = %q, want %q", c.in, got, c.want)
		}
	}
}
