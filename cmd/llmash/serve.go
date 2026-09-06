package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"sync"
	"time"
)

// `llmash serve` / `llmashw serve`: the server itself.

var procGetConsoleWindow = kernel32.NewProc("GetConsoleWindow")

func hasConsole() bool {
	h, _, _ := procGetConsoleWindow.Call()
	return h != 0
}

func setupServerLogging() {
	log.SetFlags(0)
	if lf := env("LLMASH_LOGFILE"); lf != "" {
		if f, err := os.OpenFile(lf, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644); err == nil {
			log.SetOutput(f)
			return
		}
	}
	if !hasConsole() {
		if f, err := os.OpenFile(filepath.Join(root, "llmash.log"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644); err == nil {
			log.SetOutput(f)
			return
		}
	}
	log.SetOutput(os.Stdout)
}

func serveMain(args []string) {
	setupServerLogging()
	loadConfig()
	port := envInt("LLMASH_PORT", 11434)
	host := "127.0.0.1"
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--port":
			if i+1 < len(args) {
				port, _ = strconv.Atoi(args[i+1])
				i++
			}
		case "--host":
			if i+1 < len(args) {
				host = args[i+1]
				i++
			}
		}
	}
	if publicPort == port {
		publicPort = 0
		logf("public port %d is the main port; public API disabled", port)
	}
	if !fileExists(llamaBin) {
		logf("!! llama-server not found at %s", llamaBin)
		logf("   put a llama.cpp build in that runtime folder, set LLAMA_BIN, or re-run the installer, which downloads one for this machine")
		exit(1)
	}
	reg = newRegistry()
	linkKeyVal = serverLinkKey()
	loadRoutes()
	remoteExpectLoad()

	if env("LLMASH_NO_REAP") == "" {
		reapOrphans()
		reapExeRoutes()
	}
	ctx, cancel := context.WithCancel(context.Background())
	go mgr.reaper(ctx)
	go remoteReaper(ctx)
	go cliCacheTick(ctx)
	logf("models from %s", reg.Root)
	logf("llama-server %s", llamaBin)
	logf("kv cache %s, vram budget %.0f GB", kvType, vramBudgetGB)
	if publicPort != 0 {
		logf("public API on :%d (key required) — expose with `llmash link`", publicPort)
	}

	mux := buildMux()
	servers := []*http.Server{{Addr: net.JoinHostPort(host, strconv.Itoa(port)), Handler: guarded(false, mux),
		IdleTimeout: 300 * time.Second}}
	if publicPort != 0 && publicPort != port {
		servers = append(servers, &http.Server{Addr: fmt.Sprintf("127.0.0.1:%d", publicPort),
			Handler: guarded(true, mux), IdleTimeout: 300 * time.Second})
	}
	var wg sync.WaitGroup
	errc := make(chan error, len(servers))
	for _, s := range servers {
		wg.Add(1)
		go func(s *http.Server) {
			defer wg.Done()
			if err := s.ListenAndServe(); err != nil && err != http.ErrServerClosed {
				errc <- fmt.Errorf("%s: %w", s.Addr, err)
			}
		}(s)
	}
	sigc := make(chan os.Signal, 1)
	signal.Notify(sigc, os.Interrupt)
	select {
	case err := <-errc:
		logf("!! %v", err)
	case <-sigc:
		logf("shutting down")
	}
	cancel()
	for _, s := range servers {
		c, cc := context.WithTimeout(context.Background(), 3*time.Second)
		s.Shutdown(c)
		cc()
	}
	mgr.Shutdown()
	wg.Wait()
}
