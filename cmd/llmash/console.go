package main

import (
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
