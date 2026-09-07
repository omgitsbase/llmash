package main

import (
	"fmt"
	"os"
	"os/exec"
	"strings"
	"syscall"
	"unsafe"
)

var (
	kernel32                       = syscall.NewLazyDLL("kernel32.dll")
	msvcrt                         = syscall.NewLazyDLL("msvcrt.dll")
	procGetConsoleMode             = kernel32.NewProc("GetConsoleMode")
	procSetConsoleMode             = kernel32.NewProc("SetConsoleMode")
	procGetConsoleScreenBufferInfo = kernel32.NewProc("GetConsoleScreenBufferInfo")
	procGetConsoleOutputCP         = kernel32.NewProc("GetConsoleOutputCP")
	procSetConsoleOutputCP         = kernel32.NewProc("SetConsoleOutputCP")
	procGetch                      = msvcrt.NewProc("_getch")
)

const (
	dim    = "\x1b[2m"
	bold   = "\x1b[1m"
	reset  = "\x1b[0m"
	orange = "\x1b[38;5;208m"
	green  = "\x1b[38;5;42m"
	cyan   = "\x1b[38;5;44m"
)

func isConsole(f *os.File) bool {
	var mode uint32
	r, _, _ := procGetConsoleMode.Call(f.Fd(), uintptr(unsafe.Pointer(&mode)))
	return r != 0
}

// ANSI colours and cursor moves on legacy consoles.
func enableVT() {
	for _, f := range []*os.File{os.Stdout, os.Stderr} {
		var mode uint32
		if r, _, _ := procGetConsoleMode.Call(f.Fd(), uintptr(unsafe.Pointer(&mode))); r != 0 {
			procSetConsoleMode.Call(f.Fd(), uintptr(mode|0x0004))
		}
	}
}

var savedCP uintptr

// Everything printed is UTF-8; make the console read it that way for the
// life of this process and put it back on the way out.
func useUTF8() {
	if isConsole(os.Stdout) {
		savedCP, _, _ = procGetConsoleOutputCP.Call()
		procSetConsoleOutputCP.Call(65001)
	}
}

func restoreCP() {
	if savedCP != 0 {
		procSetConsoleOutputCP.Call(savedCP)
	}
}

func exit(code int) {
	restoreCP()
	os.Exit(code)
}

func termWidth() int { return termWidthOf(os.Stdout) }

func termWidthOf(f *os.File) int {
	type coord struct{ X, Y int16 }
	type rect struct{ L, T, R, B int16 }
	var info struct {
		Size coord
		Cur  coord
		Attr uint16
		Win  rect
		Max  coord
	}
	r, _, _ := procGetConsoleScreenBufferInfo.Call(f.Fd(), uintptr(unsafe.Pointer(&info)))
	if r == 0 {
		return 100
	}
	w := int(info.Win.R-info.Win.L) + 1
	if w <= 0 {
		return 100
	}
	return w
}

// One raw key, no echo (the console's own _getch).
func getch() byte {
	r, _, _ := procGetch.Call()
	return byte(r)
}

func clip(text string) bool {
	cmd := exec.Command("clip")
	cmd.Stdin = strings.NewReader(text)
	return cmd.Run() == nil
}

// A key from the console: arrows arrive as a 0xE0 or 0x00 prefix and a scan
// code, everything else as the character.
const (
	pickUp    = 0x101
	pickDown  = 0x102
	pickEnter = 0x103
	pickEsc   = 0x104
)

func readPick() int {
	ch := getch()
	switch ch {
	case 0xE0, 0x00:
		switch getch() {
		case 0x48:
			return pickUp
		case 0x50:
			return pickDown
		}
		return 0
	case '\r', '\n':
		return pickEnter
	case 0x1b, 0x03:
		return pickEsc
	}
	return int(ch)
}

// pickMenu draws a list with a cursor and moves it with the arrows. Enter returns
// the row, Escape -1, and any key in extra is returned as its negative code
// so the caller can act on it (a for the advanced view).
func pickMenu(title string, rows []string, cursor int, hint string, extra string) int {
	if cursor < 0 || cursor >= len(rows) {
		cursor = 0
	}
	fmt.Printf("%s\n", title)
	drawn := 0
	draw := func() {
		for i := 0; i < drawn; i++ {
			fmt.Print("\x1b[A")
		}
		for i, r := range rows {
			if i == cursor {
				fmt.Printf("\x1b[1G\x1b[K  %s❯ %s%s\n", bold, r, reset)
			} else {
				fmt.Printf("\x1b[1G\x1b[K    %s\n", r)
			}
		}
		fmt.Printf("\x1b[1G\x1b[K  %s%s%s\n", dim, hint, reset)
		drawn = len(rows) + 1
	}
	draw()
	for {
		switch k := readPick(); {
		case k == pickUp && cursor > 0:
			cursor--
			draw()
		case k == pickDown && cursor < len(rows)-1:
			cursor++
			draw()
		case k == pickEnter:
			return cursor
		case k == pickEsc:
			return -1
		case k > 0 && k < 0x100 && strings.ContainsRune(extra, rune(k)):
			return -k
		}
	}
}
