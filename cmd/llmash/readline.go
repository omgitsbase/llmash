package main

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"unsafe"
)

// The prompt line for `run`: raw console input with VT escape sequences, the
// same keys and history file ollama uses, and its grey placeholder.

const (
	charNull      = 0
	charLineStart = 1
	charBackward  = 2
	charInterrupt = 3
	charDelete    = 4
	charLineEnd   = 5
	charForward   = 6
	charBell      = 7
	charCtrlH     = 8
	charTab       = 9
	charCtrlJ     = 10
	charKill      = 11
	charCtrlL     = 12
	charEnter     = 13
	charNext      = 14
	charPrev      = 16
	charCtrlU     = 21
	charCtrlW     = 23
	charEsc       = 27
	charSpace     = 32
	charEscapeEx  = 91
	charBackspace = 127

	keyDel    = 51
	keyUp     = 65
	keyDown   = 66
	keyRight  = 67
	keyLeft   = 68
	metaEnd   = 70
	metaStart = 72

	pasteMarker = 50

	colorGrey    = "\x1b[38;5;245m"
	colorDefault = "\x1b[0m"

	startBracketedPaste = "\x1b[?2004h"
	endBracketedPaste   = "\x1b[?2004l"
)

var (
	errInterrupt  = errors.New("interrupt")
	errEditPrompt = errors.New("edit prompt")
)

const (
	enableProcessedInput      = 0x0001
	enableLineInput           = 0x0002
	enableEchoInput           = 0x0004
	enableVirtualTerminalIn   = 0x0200
	enableProcessedOutputFlag = 0x0001
)

var (
	procGetConsoleCP = kernel32.NewProc("GetConsoleCP")
	procSetConsoleCP = kernel32.NewProc("SetConsoleCP")
)

type rawState struct {
	mode uint32
	cp   uintptr
	on   bool
}

func setRawMode() *rawState {
	fd := os.Stdin.Fd()
	var mode uint32
	if r, _, _ := procGetConsoleMode.Call(fd, uintptr(unsafe.Pointer(&mode))); r == 0 {
		return &rawState{}
	}
	raw := mode &^ (enableEchoInput | enableProcessedInput | enableLineInput | enableProcessedOutputFlag)
	raw |= enableVirtualTerminalIn
	procSetConsoleMode.Call(fd, uintptr(raw))
	cp, _, _ := procGetConsoleCP.Call()
	procSetConsoleCP.Call(65001) // typed text arrives as UTF-8
	return &rawState{mode: mode, cp: cp, on: true}
}

func (s *rawState) restore() {
	if s == nil || !s.on {
		return
	}
	procSetConsoleMode.Call(os.Stdin.Fd(), uintptr(s.mode))
	if s.cp != 0 {
		procSetConsoleCP.Call(s.cp)
	}
}

// ------------------------------------------------------------- history

type history struct {
	lines   []string
	pos     int
	enabled bool
	limit   int
}

func newHistory() *history {
	h := &history{enabled: true, limit: 100}
	if b, err := os.ReadFile(historyPath()); err == nil {
		for _, ln := range strings.Split(strings.ReplaceAll(string(b), "\r\n", "\n"), "\n") {
			if ln != "" {
				h.lines = append(h.lines, ln)
			}
		}
	}
	h.pos = len(h.lines)
	return h
}

func historyPath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".ollama", "history")
}

func (h *history) add(s string) {
	h.pos = len(h.lines)
	if !h.enabled {
		return
	}
	if h.pos > 0 && h.lines[h.pos-1] == s {
		h.pos = len(h.lines)
		return
	}
	h.lines = append(h.lines, s)
	if len(h.lines) > h.limit {
		h.lines = h.lines[len(h.lines)-h.limit:]
	}
	h.pos = len(h.lines)
	h.save()
}

func (h *history) save() {
	p := historyPath()
	if p == "" {
		return
	}
	os.MkdirAll(filepath.Dir(p), 0o755)
	os.WriteFile(p, []byte(strings.Join(h.lines, "\n")+"\n"), 0o600)
}

// ---------------------------------------------------------- the editor

type editor struct {
	prompt, altPrompt           string
	placeholder, altPlaceholder string
	useAlt                      bool
	pasting                     bool
	prefill                     string
	hist                        *history

	in          *bufio.Reader
	buf         []rune
	pos         int
	rows        int      // rows below the first that the last redraw used
	pastedLines []string // lines already committed inside this one prompt
}

func newEditor() *editor {
	return &editor{
		prompt: ">>> ", altPrompt: "... ",
		placeholder: "Send a message (/? for help)", altPlaceholder: "Press Enter to send",
		hist: newHistory(), in: bufio.NewReader(os.Stdin),
	}
}

func (e *editor) curPrompt() string {
	if e.useAlt || e.pasting {
		return e.altPrompt
	}
	return e.prompt
}

func (e *editor) curPlaceholder() string {
	if e.useAlt {
		return e.altPlaceholder
	}
	return e.placeholder
}

func width() int {
	w := termWidth()
	if w < 10 {
		w = 80
	}
	return w
}

func (e *editor) draw() {
	w := width()
	prompt := e.curPrompt()
	pw := dispWidth(prompt)
	if e.rows > 0 {
		fmt.Printf("\x1b[%dA", e.rows)
	}
	fmt.Print("\r" + prompt + string(e.buf) + "\x1b[0J")
	total := pw + dispWidth(string(e.buf))
	endRow, endCol := total/w, total%w
	if endCol == 0 && total > 0 {
		fmt.Print(" \r") // force the deferred wrap so the cursor is on the new row
	}
	if len(e.buf) == 0 {
		if ph := e.curPlaceholder(); ph != "" && (!e.pasting || e.useAlt) {
			fmt.Print(colorGrey + ph + colorDefault)
			fmt.Print("\r")
			if pw > 0 {
				fmt.Printf("\x1b[%dC", pw)
			}
			e.rows = 0
			return
		}
	}
	target := pw + dispWidth(string(e.buf[:e.pos]))
	tRow, tCol := target/w, target%w
	if endRow > tRow {
		fmt.Printf("\x1b[%dA", endRow-tRow)
	}
	fmt.Print("\r")
	if tCol > 0 {
		fmt.Printf("\x1b[%dC", tCol)
	}
	e.rows = tRow
}

func (e *editor) insert(r rune) {
	e.buf = append(e.buf, 0)
	copy(e.buf[e.pos+1:], e.buf[e.pos:])
	e.buf[e.pos] = r
	e.pos++
}

func (e *editor) replace(s []rune) {
	e.buf = append([]rune{}, s...)
	e.pos = len(e.buf)
}

func isWordRune(r rune) bool { return r != ' ' && r != '\t' }

func (e *editor) leftWord() int {
	i := e.pos
	for i > 0 && !isWordRune(e.buf[i-1]) {
		i--
	}
	for i > 0 && isWordRune(e.buf[i-1]) {
		i--
	}
	return i
}

func (e *editor) rightWord() int {
	i := e.pos
	for i < len(e.buf) && !isWordRune(e.buf[i]) {
		i++
	}
	for i < len(e.buf) && isWordRune(e.buf[i]) {
		i++
	}
	return i
}

func (e *editor) readline() (string, error) {
	st := setRawMode()
	defer st.restore()
	e.rows = 0
	e.buf = e.buf[:0]
	e.pos = 0
	fmt.Print(e.curPrompt())

	if e.prefill != "" {
		lines := strings.Split(e.prefill, "\n")
		e.prefill = ""
		for i, ln := range lines {
			for _, r := range ln {
				e.insert(r)
			}
			if i < len(lines)-1 {
				e.pastedLine()
			}
		}
	}
	e.draw()

	var esc, escex, metaDel bool
	var saved []rune
	for {
		r, _, err := e.in.ReadRune()
		if err != nil {
			fmt.Println()
			return "", io.EOF
		}
		if escex {
			escex = false
			switch r {
			case keyUp:
				if e.hist.pos > 0 {
					if e.hist.pos == len(e.hist.lines) {
						saved = append([]rune{}, e.buf...)
					}
					e.hist.pos--
					e.replace([]rune(e.hist.lines[e.hist.pos]))
				}
			case keyDown:
				if e.hist.pos < len(e.hist.lines) {
					e.hist.pos++
					if e.hist.pos == len(e.hist.lines) {
						e.replace(saved)
					} else {
						e.replace([]rune(e.hist.lines[e.hist.pos]))
					}
				}
			case keyLeft:
				if e.pos > 0 {
					e.pos--
				}
			case keyRight:
				if e.pos < len(e.buf) {
					e.pos++
				}
			case pasteMarker:
				code := ""
				for i := 0; i < 3; i++ {
					c, _, err := e.in.ReadRune()
					if err != nil {
						return "", io.EOF
					}
					code += string(c)
				}
				if code == "00~" {
					e.pasting = true
				} else if code == "01~" {
					e.pasting = false
				}
			case keyDel:
				if e.pos < len(e.buf) {
					e.buf = append(e.buf[:e.pos], e.buf[e.pos+1:]...)
				}
				metaDel = true
			case metaStart:
				e.pos = 0
			case metaEnd:
				e.pos = len(e.buf)
			default:
				continue
			}
			e.draw()
			continue
		}
		if esc {
			esc = false
			switch r {
			case 'b':
				e.pos = e.leftWord()
			case 'f':
				e.pos = e.rightWord()
			case charBackspace:
				i := e.leftWord()
				e.buf = append(e.buf[:i], e.buf[e.pos:]...)
				e.pos = i
			case charEscapeEx:
				escex = true
				continue
			}
			e.draw()
			continue
		}

		switch r {
		case charNull:
			continue
		case charEsc:
			esc = true
			continue
		case charInterrupt:
			e.pastedLines = nil
			e.useAlt = false
			return "", errInterrupt
		case charPrev:
			if e.hist.pos > 0 {
				if e.hist.pos == len(e.hist.lines) {
					saved = append([]rune{}, e.buf...)
				}
				e.hist.pos--
				e.replace([]rune(e.hist.lines[e.hist.pos]))
			}
		case charNext:
			if e.hist.pos < len(e.hist.lines) {
				e.hist.pos++
				if e.hist.pos == len(e.hist.lines) {
					e.replace(saved)
				} else {
					e.replace([]rune(e.hist.lines[e.hist.pos]))
				}
			}
		case charLineStart:
			e.pos = 0
		case charLineEnd:
			e.pos = len(e.buf)
		case charBackward:
			if e.pos > 0 {
				e.pos--
			}
		case charForward:
			if e.pos < len(e.buf) {
				e.pos++
			}
		case charBackspace, charCtrlH:
			if len(e.buf) == 0 && len(e.pastedLines) > 0 {
				last := e.pastedLines[len(e.pastedLines)-1]
				e.pastedLines = e.pastedLines[:len(e.pastedLines)-1]
				fmt.Print("\r\x1b[K\x1b[A\r\x1b[K")
				if len(e.pastedLines) == 0 {
					e.useAlt = false
				}
				fmt.Print(e.curPrompt())
				e.rows = 0
				e.replace([]rune(last))
			} else if e.pos > 0 {
				e.buf = append(e.buf[:e.pos-1], e.buf[e.pos:]...)
				e.pos--
			}
		case charTab:
			for i := 0; i < 8; i++ {
				e.insert(' ')
			}
		case charDelete:
			if len(e.buf) > 0 {
				if e.pos < len(e.buf) {
					e.buf = append(e.buf[:e.pos], e.buf[e.pos+1:]...)
				}
			} else {
				fmt.Println()
				return "", io.EOF
			}
		case charKill:
			e.buf = e.buf[:e.pos]
		case charCtrlU:
			e.buf = append([]rune{}, e.buf[e.pos:]...)
			e.pos = 0
		case charCtrlL:
			fmt.Print("\x1b[2J\x1b[0;0f")
			e.rows = 0
			fmt.Print(e.curPrompt())
		case charCtrlW:
			i := e.leftWord()
			e.buf = append(e.buf[:i], e.buf[e.pos:]...)
			e.pos = i
		case charBell:
			out := string(e.buf)
			if len(e.pastedLines) > 0 {
				out = strings.Join(e.pastedLines, "\n") + "\n" + out
				e.pastedLines = nil
			}
			fmt.Print("\r\x1b[0J")
			e.useAlt = false
			return out, errEditPrompt
		case charCtrlJ:
			e.pastedLine()
		case charEnter:
			out := string(e.buf)
			if len(e.pastedLines) > 0 {
				out = strings.Join(e.pastedLines, "\n") + "\n" + out
				e.pastedLines = nil
			}
			e.pos = len(e.buf)
			e.draw()
			fmt.Println()
			e.useAlt = false
			if out != "" {
				e.hist.add(out)
			}
			return out, nil
		default:
			if metaDel {
				metaDel = false
				continue
			}
			if r >= charSpace {
				e.insert(r)
			}
		}
		e.draw()
	}
}

// A newline typed inside one prompt (Ctrl+J, or a pasted line break).
func (e *editor) pastedLine() {
	e.pastedLines = append(e.pastedLines, string(e.buf))
	e.buf = e.buf[:0]
	e.pos = 0
	e.rows = 0
	fmt.Println()
	e.useAlt = true
	fmt.Print(e.altPrompt)
}

func editInExternalEditor(content string) (string, error) {
	ed := first(os.Getenv("OLLAMA_EDITOR"), os.Getenv("VISUAL"), os.Getenv("EDITOR"), "notepad")
	args := strings.Fields(ed)
	if len(args) == 0 {
		return "", fmt.Errorf("no editor configured, set OLLAMA_EDITOR to the path of your preferred editor")
	}
	if _, err := exec.LookPath(args[0]); err != nil {
		return "", fmt.Errorf("editor %q not found, set OLLAMA_EDITOR to the path of your preferred editor", args[0])
	}
	f, err := os.CreateTemp("", "llmash-prompt-*.txt")
	if err != nil {
		return "", fmt.Errorf("creating temp file: %w", err)
	}
	defer os.Remove(f.Name())
	if content != "" {
		f.WriteString(content)
	}
	f.Close()
	cmd := exec.Command(args[0], append(args[1:], f.Name())...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	cmd.SysProcAttr = &syscall.SysProcAttr{}
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("editor exited with error: %w", err)
	}
	b, err := os.ReadFile(f.Name())
	if err != nil {
		return "", err
	}
	return strings.TrimRight(strings.ReplaceAll(string(b), "\r\n", "\n"), "\n"), nil
}
