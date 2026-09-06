package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"unsafe"
)

var (
	advapi32             = syscall.NewLazyDLL("advapi32.dll")
	procRegOpenKeyExW    = advapi32.NewProc("RegOpenKeyExW")
	procRegQueryValueExW = advapi32.NewProc("RegQueryValueExW")
	procRegSetValueExW   = advapi32.NewProc("RegSetValueExW")
	procRegDeleteKeyW    = advapi32.NewProc("RegDeleteKeyW")
	procRegCloseKey      = advapi32.NewProc("RegCloseKey")
)

const (
	hkeyCurrentUser = 0x80000001
	keyRead         = 0x20019
	keyWrite        = 0x20006
)

func utf16(s string) *uint16 {
	p, _ := syscall.UTF16PtrFromString(s)
	return p
}

func regDeleteKey(path string) bool {
	r, _, _ := procRegDeleteKeyW.Call(hkeyCurrentUser, uintptr(unsafe.Pointer(utf16(path))))
	return r == 0
}

// The user PATH with our bin entry removed. Type (REG_SZ / REG_EXPAND_SZ) is
// preserved; nothing else about the value is touched.
func removeFromUserPath(binDir string) bool {
	var h uintptr
	r, _, _ := procRegOpenKeyExW.Call(hkeyCurrentUser, uintptr(unsafe.Pointer(utf16("Environment"))), 0,
		keyRead|keyWrite, uintptr(unsafe.Pointer(&h)))
	if r != 0 {
		return false
	}
	defer procRegCloseKey.Call(h)
	var typ, size uint32
	name := utf16("Path")
	if r, _, _ := procRegQueryValueExW.Call(h, uintptr(unsafe.Pointer(name)), 0, uintptr(unsafe.Pointer(&typ)), 0,
		uintptr(unsafe.Pointer(&size))); r != 0 || size == 0 {
		return false
	}
	buf := make([]uint16, size/2+1)
	if r, _, _ := procRegQueryValueExW.Call(h, uintptr(unsafe.Pointer(name)), 0, uintptr(unsafe.Pointer(&typ)),
		uintptr(unsafe.Pointer(&buf[0])), uintptr(unsafe.Pointer(&size))); r != 0 {
		return false
	}
	cur := syscall.UTF16ToString(buf)
	want := strings.ToLower(filepath.Clean(binDir))
	var kept []string
	changed := false
	for _, p := range strings.Split(cur, ";") {
		if p == "" {
			continue
		}
		if strings.ToLower(filepath.Clean(p)) == want {
			changed = true
			continue
		}
		kept = append(kept, p)
	}
	if !changed {
		return false
	}
	val, _ := syscall.UTF16FromString(strings.Join(kept, ";"))
	r, _, _ = procRegSetValueExW.Call(h, uintptr(unsafe.Pointer(name)), 0, uintptr(typ),
		uintptr(unsafe.Pointer(&val[0])), uintptr(len(val)*2))
	return r == 0
}

// A hidden PowerShell with its own invisible console: DETACHED_PROCESS makes
// powershell.exe exit without running its -Command (measured), CREATE_NO_WINDOW
// does not.
func hiddenPowerShell(script string, wait bool) (string, int) {
	build := func(flags uint32) *exec.Cmd {
		c := exec.Command("powershell", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-Command", script)
		c.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: flags}
		return c
	}
	cmd := build(0x08000000 | 0x00000200)
	if !wait {
		// breakaway from any job object first, so a parent's teardown cannot take it
		if err := build(0x08000000 | 0x00000200 | 0x01000000).Start(); err != nil {
			build(0x08000000 | 0x00000200).Start()
		}
		return "", 0
	}
	out, err := cmd.Output()
	code := 0
	if e, ok := err.(*exec.ExitError); ok {
		code = e.ExitCode()
	}
	return string(out), code
}

func psQuote(s string) string { return strings.ReplaceAll(s, "'", "''") }

func fileExists(p string) bool {
	st, err := os.Stat(p)
	return err == nil && !st.IsDir()
}

func dirExists(p string) bool {
	st, err := os.Stat(p)
	return err == nil && st.IsDir()
}

func which(exe string) string {
	p, err := exec.LookPath(exe)
	if err != nil {
		return ""
	}
	return p
}
