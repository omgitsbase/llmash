package main

import (
	"os/exec"
	"strings"
	"testing"
)

// The restart path kills processes, so what it generates has to be right. A
// stray quote here would either kill nothing or kill somebody else's server.
func TestStopScript(t *testing.T) {
	s := stopScript(`C:\Program Files\llmash`)

	for _, want := range []string{
		`'*C:\Program Files\llmash\llmashw.exe*'`, // this install only
		`'*C:\Program Files\llmash\llmash.exe*'`,
		"'* serve*'",              // the server, not a tray or a client
		"Name='llama-server.exe'", // the engines it started
		"ParentProcessId",         // and only its own
	} {
		if !strings.Contains(s, want) {
			t.Errorf("the script should contain %s\ngot: %s", want, s)
		}
	}
	if strings.Count(s, `"`)%2 != 0 {
		t.Errorf("unbalanced quotes:\n%s", s)
	}
	// the engines have to go before their parent, or they are orphaned
	if strings.Index(s, "llama-server.exe") > strings.LastIndex(s, "$s | ForEach-Object") {
		t.Error("the engines must be stopped before the server that owns them")
	}
}

// A single quote in the path is what would break the quoting, so check that
// PowerShell itself accepts what comes out.
func TestStopScriptParses(t *testing.T) {
	if _, err := exec.LookPath("powershell"); err != nil {
		t.Skip("no powershell here")
	}
	for _, root := range []string{
		`C:\Program Files\llmash`,
		`C:\Users\someone\It's Mine\llmash`,
		`B:\llmash`,
	} {
		script := stopScript(root)
		cmd := exec.Command("powershell", "-NoProfile", "-Command",
			"$ErrorActionPreference='Stop'; "+
				"$null = [ScriptBlock]::Create($input -join \"`n\"); 'parses'")
		cmd.Stdin = strings.NewReader(script)
		out, err := cmd.CombinedOutput()
		if err != nil || !strings.Contains(string(out), "parses") {
			t.Errorf("%s: PowerShell would not parse this\n%s\n%s", root, script, out)
		}
	}
}
