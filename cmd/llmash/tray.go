package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"
	"unsafe"
)

// The notification-area icon, in plain Win32: a hidden window, a
// Shell_NotifyIcon entry, and a popup menu built fresh each time it opens so
// the loaded models and their timers are live. The server is meant to be up
// whenever the tray is: it is started at launch and watched every 5 s.

var (
	user32                  = syscall.NewLazyDLL("user32.dll")
	shell32                 = syscall.NewLazyDLL("shell32.dll")
	procRegisterClassExW    = user32.NewProc("RegisterClassExW")
	procCreateWindowExW     = user32.NewProc("CreateWindowExW")
	procDefWindowProcW      = user32.NewProc("DefWindowProcW")
	procGetMessageW         = user32.NewProc("GetMessageW")
	procTranslateMessage    = user32.NewProc("TranslateMessage")
	procDispatchMessageW    = user32.NewProc("DispatchMessageW")
	procPostQuitMessage     = user32.NewProc("PostQuitMessage")
	procCreatePopupMenu     = user32.NewProc("CreatePopupMenu")
	procAppendMenuW         = user32.NewProc("AppendMenuW")
	procTrackPopupMenu      = user32.NewProc("TrackPopupMenu")
	procDestroyMenu         = user32.NewProc("DestroyMenu")
	procSetForegroundWindow = user32.NewProc("SetForegroundWindow")
	procGetCursorPos        = user32.NewProc("GetCursorPos")
	procPostMessageW        = user32.NewProc("PostMessageW")
	procLoadImageW          = user32.NewProc("LoadImageW")
	procShellNotifyIconW    = shell32.NewProc("Shell_NotifyIconW")
	procGetModuleHandleW    = kernel32.NewProc("GetModuleHandleW")
)

const (
	wmDestroy      = 0x0002
	wmCommand      = 0x0111
	wmUser         = 0x0400
	wmTrayCallback = wmUser + 1
	wmReopen       = wmUser + 2
	wmLButtonUp    = 0x0202
	wmRButtonUp    = 0x0205
	wmNull         = 0x0000
	nimAdd         = 0
	nimModify      = 1
	nimDelete      = 2
	nifMessage     = 0x1
	nifIcon        = 0x2
	nifTip         = 0x4
	nifInfo        = 0x10
	niifWarning    = 0x2
	mfString       = 0x0
	mfSeparator    = 0x800
	mfGrayed       = 0x1
	mfChecked      = 0x8
	mfPopup        = 0x10
	tpmReturnCmd   = 0x100
	tpmRightButton = 0x2
	imageIcon      = 1
	lrLoadFromFile = 0x10
	lrDefaultSize  = 0x40
)

type notifyIconData struct {
	CbSize           uint32
	HWnd             uintptr
	UID              uint32
	UFlags           uint32
	UCallbackMessage uint32
	HIcon            uintptr
	SzTip            [128]uint16
	DwState          uint32
	DwStateMask      uint32
	SzInfo           [256]uint16
	UVersion         uint32
	SzInfoTitle      [64]uint16
	DwInfoFlags      uint32
	GuidItem         [16]byte
	HBalloonIcon     uintptr
}

type wndClassEx struct {
	CbSize        uint32
	Style         uint32
	LpfnWndProc   uintptr
	CbClsExtra    int32
	CbWndExtra    int32
	HInstance     uintptr
	HIcon         uintptr
	HCursor       uintptr
	HbrBackground uintptr
	LpszMenuName  *uint16
	LpszClassName *uint16
	HIconSm       uintptr
}

type point struct{ X, Y int32 }

type msg struct {
	HWnd    uintptr
	Message uint32
	WParam  uintptr
	LParam  uintptr
	Time    uint32
	Pt      point
}

// What a menu item does, and what happens to the menu afterwards. Windows
// closes a popup menu the moment something is picked; everything here except
// "Close llmash" puts it straight back so a second choice costs one click.
type trayAction struct {
	fn   func()
	wait bool // finish the work first, so the reopened menu shows the result
	stop bool // no reopen
}

type trayApp struct {
	hwnd     uintptr
	nid      notifyIconData
	mu       sync.Mutex
	lock     sync.Mutex // one server start/stop at a time
	closing  bool
	actions  map[uintptr]*trayAction
	nextID   uintptr
	anchor   point
	menuOpen bool
}

var tray *trayApp

func trayStatePath() string { return filepath.Join(root, "tray.json") }

func trayState() map[string]any {
	d := map[string]any{}
	if b, err := os.ReadFile(trayStatePath()); err == nil {
		json.Unmarshal(b, &d)
	}
	return d
}

func traySaveState(k string, v any) {
	d := trayState()
	d[k] = v
	if b, err := json.Marshal(d); err == nil {
		os.WriteFile(trayStatePath(), b, 0o644)
	}
}

func startupLnk() string {
	return filepath.Join(os.Getenv("APPDATA"), "Microsoft", "Windows", "Start Menu", "Programs", "Startup", "llmash.lnk")
}

func startupEnabled() bool { return fileExists(startupLnk()) }

func enableStartup() {
	exe := filepath.Join(root, "llmashw.exe")
	icon := filepath.Join(root, "llmash.ico")
	script := fmt.Sprintf("$w = New-Object -ComObject WScript.Shell; $s = $w.CreateShortcut('%s'); "+
		"$s.TargetPath = '%s'; $s.Arguments = 'tray'; $s.WorkingDirectory = '%s'; $s.WindowStyle = 7; "+
		"$s.Description = 'llmash'; ", psQuote(startupLnk()), psQuote(exe), psQuote(root))
	if fileExists(icon) {
		script += fmt.Sprintf("$s.IconLocation = '%s'; ", psQuote(icon))
	}
	script += "$s.Save()"
	hiddenPowerShell(script, true)
	traySaveState("startup", true)
}

func toggleStartup() {
	if startupEnabled() {
		os.Remove(startupLnk())
		traySaveState("startup", false)
	} else {
		enableStartup()
	}
}

func defaultStartup() {
	if startupEnabled() {
		return
	}
	if v, ok := trayState()["startup"].(bool); ok && !v {
		return
	}
	enableStartup()
}

// ------------------------------------------------------- server control

func trayServerUp() bool {
	_, err := call("GET", "/", nil, 1500*time.Millisecond)
	return err == nil
}

func serverProcessExists() bool {
	r := psQuote(root)
	script := "@(Get-CimInstance Win32_Process -Filter \"Name='llmashw.exe' OR Name='llmash.exe'\" " +
		"| Where-Object { ($_.CommandLine -like '*" + r + "\\llmashw.exe*' -or $_.CommandLine -like '*" + r + "\\llmash.exe*') " +
		"-and $_.CommandLine -like '* serve*' }).Count"
	out, _ := hiddenPowerShell(script, true)
	var n int
	fmt.Sscan(strings.TrimSpace(out), &n)
	return n > 0
}

func startServerProcess() bool {
	if trayServerUp() {
		return true
	}
	if serverProcessExists() {
		t0 := time.Now()
		for time.Since(t0) < 20*time.Second {
			if trayServerUp() {
				return true
			}
			time.Sleep(500 * time.Millisecond)
		}
	}
	cmd := exec.Command(filepath.Join(root, "llmashw.exe"), "serve")
	cmd.Dir = root
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: createNoWindow}
	if err := cmd.Start(); err != nil {
		return false
	}
	cmd.Process.Release()
	t0 := time.Now()
	for time.Since(t0) < 30*time.Second {
		if trayServerUp() {
			return true
		}
		time.Sleep(400 * time.Millisecond)
	}
	return false
}

func stopServerProcess() {
	r := psQuote(root)
	script := "Get-CimInstance Win32_Process -Filter \"Name='llmashw.exe' OR Name='llmash.exe'\" " +
		"| Where-Object { ($_.CommandLine -like '*" + r + "\\llmashw.exe*' -or $_.CommandLine -like '*" + r + "\\llmash.exe*') " +
		"-and $_.CommandLine -like '* serve*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
	hiddenPowerShell(script, true)
}

func loadedModels() []map[string]any {
	d, _, err := callJSON("GET", "/api/ps", nil, 2*time.Second)
	if err != nil {
		return nil
	}
	var out []map[string]any
	for _, m := range list(d, "models") {
		if mm, ok := m.(map[string]any); ok {
			out = append(out, mm)
		}
	}
	return out
}

// Returns the server's complaint, or "" when the model really was retimed.
func setKeepAlive(model string, seconds int) string {
	r, err := call("POST", "/api/keep_alive", map[string]any{"model": model, "keep_alive": seconds}, 20*time.Second)
	if err != nil {
		return err.Error()
	}
	b, _ := readAll(r)
	if r.StatusCode >= 400 {
		var d map[string]any
		if json.Unmarshal(b, &d) == nil && str(d, "error") != "" {
			return str(d, "error")
		}
		return strings.TrimSpace(string(b))
	}
	return ""
}

func expiresText(m map[string]any) string {
	raw := str(m, "expires_at")
	if raw == "" {
		return ""
	}
	if strings.HasPrefix(raw, "9999") {
		return "pinned"
	}
	t, err := time.Parse(time.RFC3339Nano, raw)
	if err != nil {
		return ""
	}
	left := time.Until(t).Seconds()
	switch {
	case left <= 0:
		return "unloading"
	case left < 90:
		return fmt.Sprintf("%ds left", int(left))
	case left < 5400:
		return fmt.Sprintf("%dm left", int(left/60))
	}
	return fmt.Sprintf("%.1fh left", left/3600)
}

func sizeText(m map[string]any) string {
	b := num(m, "size_vram")
	if b == 0 {
		b = num(m, "size")
	}
	if b == 0 {
		return ""
	}
	return fmt.Sprintf("%.1f GB", b/(1<<30))
}

// ---------------------------------------------------------- the window

// A notification-area balloon, for the cases where a menu click can't do
// what it says.
func (t *trayApp) balloon(title, text string) {
	nid := t.nid
	nid.UFlags = nifInfo
	nid.DwInfoFlags = niifWarning
	if s, err := syscall.UTF16FromString(text); err == nil {
		copy(nid.SzInfo[:len(nid.SzInfo)-1], s)
	}
	if s, err := syscall.UTF16FromString(title); err == nil {
		copy(nid.SzInfoTitle[:len(nid.SzInfoTitle)-1], s)
	}
	procShellNotifyIconW.Call(nimModify, uintptr(unsafe.Pointer(&nid)))
}

func (t *trayApp) kick() {
	go func() {
		if !t.lock.TryLock() {
			return
		}
		defer t.lock.Unlock()
		if !trayServerUp() && !serverProcessExists() {
			startServerProcess()
		}
	}()
}

func (t *trayApp) restart() {
	go func() {
		t.lock.Lock()
		defer t.lock.Unlock()
		stopServerProcess()
		time.Sleep(time.Second)
		startServerProcess()
	}()
}

func (t *trayApp) quit() {
	t.mu.Lock()
	t.closing = true
	t.mu.Unlock()
	t.lock.Lock()
	stopServerProcess()
	t.lock.Unlock()
	procPostMessageW.Call(t.hwnd, wmDestroy, 0, 0)
}

func (t *trayApp) watch() {
	for {
		time.Sleep(5 * time.Second)
		t.mu.Lock()
		closing := t.closing
		t.mu.Unlock()
		if closing {
			return
		}
		if t.lock.TryLock() {
			if !trayServerUp() && !serverProcessExists() {
				startServerProcess()
			}
			t.lock.Unlock()
		}
	}
}

func (t *trayApp) item(menu uintptr, flags uintptr, text string, act *trayAction) {
	id := uintptr(0)
	if act != nil {
		t.nextID++
		id = t.nextID
		t.actions[id] = act
	} else if flags&mfPopup == 0 {
		flags |= mfGrayed
	}
	procAppendMenuW.Call(menu, flags, id, uintptr(unsafe.Pointer(utf16(text))))
}

// Retiming a model: report the failure rather than swallowing it, since the
// menu is about to reopen showing the unchanged timer.
func (t *trayApp) retime(name string, secs int) {
	if e := setKeepAlive(name, secs); e != "" {
		t.balloon("llmash", name+": "+e)
	}
}

func (t *trayApp) showMenu(pt point) {
	if t.menuOpen {
		return
	}
	t.menuOpen = true
	defer func() { t.menuOpen = false }()
	t.anchor = pt

	up := trayServerUp()
	var models []map[string]any
	if up {
		models = loadedModels()
	} else {
		t.kick()
	}
	t.actions = map[uintptr]*trayAction{}
	t.nextID = 0
	menu, _, _ := procCreatePopupMenu.Call()
	status := "starting…"
	if up {
		status = "running"
	}
	t.item(menu, mfString, "llmash · "+status, nil)
	procAppendMenuW.Call(menu, mfSeparator, 0, 0)
	if len(models) > 0 {
		for _, m := range models {
			name := str(m, "name")
			pinned := strings.HasPrefix(str(m, "expires_at"), "9999")
			subm, _, _ := procCreatePopupMenu.Call()
			head := strings.Join(nonEmpty(sizeText(m), expiresText(m)), " · ")
			if head == "" {
				head = "loaded"
			}
			t.item(subm, mfString, head, nil)
			procAppendMenuW.Call(subm, mfSeparator, 0, 0)
			t.item(subm, mfString, "Unload now", &trayAction{fn: func() { t.retime(name, 0) }, wait: true})
			for _, c := range []struct {
				label string
				secs  int
			}{{"Unload in 5 minutes", 300}, {"Unload in 15 minutes", 900}, {"Unload in 1 hour", 3600}, {"Keep loaded", -1}} {
				secs := c.secs
				flags := uintptr(mfString)
				if secs == -1 && pinned {
					flags |= mfChecked
				}
				t.item(subm, flags, c.label, &trayAction{fn: func() { t.retime(name, secs) }, wait: true})
			}
			// The state belongs on the row itself: a reopened menu should
			// show what the last click did without hovering again.
			label := name
			if s := expiresText(m); s != "" {
				label += " · " + s
			}
			procAppendMenuW.Call(menu, mfPopup|mfString, subm, uintptr(unsafe.Pointer(utf16(label))))
		}
		all := models
		t.item(menu, mfString, fmt.Sprintf("Unload all (%d)", len(models)), &trayAction{wait: true, fn: func() {
			for _, m := range all {
				t.retime(str(m, "name"), 0)
			}
		}})
	} else if up {
		t.item(menu, mfString, "No models loaded", nil)
	} else {
		t.item(menu, mfString, "Server is starting", nil)
	}
	procAppendMenuW.Call(menu, mfSeparator, 0, 0)
	t.item(menu, mfString, "Restart server", &trayAction{fn: t.restart})
	flags := uintptr(mfString)
	if startupEnabled() {
		flags |= mfChecked
	}
	t.item(menu, flags, "Start at login", &trayAction{fn: toggleStartup, wait: true})
	procAppendMenuW.Call(menu, mfSeparator, 0, 0)
	t.item(menu, mfString, "Close llmash", &trayAction{fn: t.quit, stop: true})

	procSetForegroundWindow.Call(t.hwnd)
	cmd, _, _ := procTrackPopupMenu.Call(menu, tpmReturnCmd|tpmRightButton, uintptr(pt.X), uintptr(pt.Y), 0, t.hwnd, 0)
	procPostMessageW.Call(t.hwnd, wmNull, 0, 0)
	procDestroyMenu.Call(menu)
	act := t.actions[cmd]
	if act == nil {
		return
	}
	hwnd := t.hwnd
	reopen := func() { procPostMessageW.Call(hwnd, wmReopen, 0, 0) }
	switch {
	case act.stop:
		go act.fn()
	case act.wait:
		go func() { act.fn(); reopen() }()
	default:
		go act.fn()
		reopen()
	}
}

func nonEmpty(xs ...string) []string {
	var out []string
	for _, x := range xs {
		if x != "" {
			out = append(out, x)
		}
	}
	return out
}

func trayWndProc(hwnd uintptr, m uint32, wp, lp uintptr) uintptr {
	switch m {
	case wmTrayCallback:
		if lp == wmLButtonUp || lp == wmRButtonUp {
			var pt point
			procGetCursorPos.Call(uintptr(unsafe.Pointer(&pt)))
			tray.showMenu(pt)
		}
		return 0
	case wmReopen:
		tray.showMenu(tray.anchor)
		return 0
	case wmDestroy:
		tray.nid.CbSize = uint32(unsafe.Sizeof(tray.nid))
		procShellNotifyIconW.Call(nimDelete, uintptr(unsafe.Pointer(&tray.nid)))
		procPostQuitMessage.Call(0)
		return 0
	}
	r, _, _ := procDefWindowProcW.Call(hwnd, uintptr(m), wp, lp)
	return r
}

func trayMain() {
	runtime.LockOSThread()
	loadLocal()
	tray = &trayApp{actions: map[uintptr]*trayAction{}}
	hinst, _, _ := procGetModuleHandleW.Call(0)
	className := utf16("llmashTray")
	wc := wndClassEx{CbSize: uint32(unsafe.Sizeof(wndClassEx{})), LpfnWndProc: syscall.NewCallback(trayWndProc),
		HInstance: hinst, LpszClassName: className}
	procRegisterClassExW.Call(uintptr(unsafe.Pointer(&wc)))
	hwnd, _, _ := procCreateWindowExW.Call(0, uintptr(unsafe.Pointer(className)), uintptr(unsafe.Pointer(utf16("llmash"))),
		0, 0, 0, 0, 0, 0, 0, hinst, 0)
	tray.hwnd = hwnd

	var hicon uintptr
	if p := filepath.Join(root, "llmash.ico"); fileExists(p) {
		hicon, _, _ = procLoadImageW.Call(0, uintptr(unsafe.Pointer(utf16(p))), imageIcon, 0, 0, lrLoadFromFile|lrDefaultSize)
	}
	nid := &tray.nid
	nid.CbSize = uint32(unsafe.Sizeof(*nid))
	nid.HWnd = hwnd
	nid.UID = 1
	nid.UFlags = nifMessage | nifIcon | nifTip
	nid.UCallbackMessage = wmTrayCallback
	nid.HIcon = hicon
	tip, _ := syscall.UTF16FromString("llmash")
	copy(nid.SzTip[:], tip)
	procShellNotifyIconW.Call(nimAdd, uintptr(unsafe.Pointer(nid)))

	go func() {
		tray.lock.Lock()
		startServerProcess()
		tray.lock.Unlock()
	}()
	go defaultStartup()
	go tray.watch()

	var m msg
	for {
		r, _, _ := procGetMessageW.Call(uintptr(unsafe.Pointer(&m)), 0, 0, 0)
		if int32(r) <= 0 {
			break
		}
		procTranslateMessage.Call(uintptr(unsafe.Pointer(&m)))
		procDispatchMessageW.Call(uintptr(unsafe.Pointer(&m)))
	}
}
