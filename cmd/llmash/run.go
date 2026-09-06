package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

// `run`: ollama's session, command for command — one-shot generate, the
// interactive prompt with its slash commands, word wrap, thinking output and
// timings.

type message map[string]any

type runOptions struct {
	model, parentModel string
	system             string
	prompt             string
	messages           []message
	loadedMessages     []message
	options            map[string]any
	format             string
	think              any // nil, true/false, or a level
	hideThinking       bool
	keepAlive          any
	wordWrap           bool
	multiModal         bool
	verbose            bool
	showConnect        bool
	images             []string
}

func (o runOptions) copy() runOptions {
	n := o
	n.messages = append([]message{}, o.messages...)
	n.loadedMessages = append([]message{}, o.loadedMessages...)
	n.options = map[string]any{}
	for k, v := range o.options {
		n.options[k] = v
	}
	return n
}

// -------------------------------------------------------- the response

type displayState struct {
	lineLength int
	wordBuffer string
}

func displayResponse(content string, wordWrap bool, state *displayState) {
	w := termWidth()
	if w == 0 {
		w = 80
	}
	if wordWrap && w >= 10 {
		for _, ch := range content {
			if state.lineLength+1 > w-5 {
				if dispWidth(state.wordBuffer) > w-10 {
					fmt.Printf("%s%c", state.wordBuffer, ch)
					state.wordBuffer = ""
					state.lineLength = 0
					continue
				}
				if a := dispWidth(state.wordBuffer); a > 0 {
					fmt.Printf("\x1b[%dD", a)
				}
				fmt.Print("\x1b[K\n")
				fmt.Printf("%s%c", state.wordBuffer, ch)
				state.lineLength = dispWidth(state.wordBuffer) + dispWidth(string(ch))
				continue
			}
			fmt.Print(string(ch))
			state.lineLength += dispWidth(string(ch))
			if dispWidth(string(ch)) >= 2 {
				state.wordBuffer = ""
				continue
			}
			switch ch {
			case ' ', '\t':
				state.wordBuffer = ""
			case '\n', '\r':
				state.lineLength = 0
				state.wordBuffer = ""
			default:
				state.wordBuffer += string(ch)
			}
		}
		return
	}
	fmt.Printf("%s%s", state.wordBuffer, content)
	state.wordBuffer = ""
}

func thinkOpen(plain bool) string {
	if plain {
		return "Thinking...\n"
	}
	return colorGrey + "\x1b[1m" + "Thinking...\n" + colorDefault + colorGrey
}

func thinkClose(plain bool) string {
	if plain {
		return "...done thinking.\n\n"
	}
	return colorGrey + "\x1b[1m" + "...done thinking.\n\n" + colorDefault
}

func renderToolCalls(calls []any, plain bool) string {
	out, expl, vals := "", "", ""
	if !plain {
		expl, vals = colorGrey+"\x1b[1m", colorDefault
		out += expl
	}
	for i, tc := range calls {
		f, _ := tc.(map[string]any)
		fn := sub(f, "function")
		args, err := json.Marshal(fn["arguments"])
		if err != nil {
			return ""
		}
		if i > 0 {
			out += "\n"
		}
		out += fmt.Sprintf("  Model called a non-existent function '%s()' with arguments: %s",
			vals+str(fn, "name")+expl, vals+string(args)+expl)
	}
	if !plain {
		out += colorDefault
	}
	return out
}

func summary(m map[string]any) {
	dur := func(k string) time.Duration { return time.Duration(int64(num(m, k))) }
	if d := dur("total_duration"); d > 0 {
		fmt.Fprintf(os.Stderr, "total duration:       %v\n", d)
	}
	if d := dur("load_duration"); d > 0 {
		fmt.Fprintf(os.Stderr, "load duration:        %v\n", d)
	}
	pc := int(num(m, "prompt_eval_count"))
	if pc > 0 {
		fmt.Fprintf(os.Stderr, "prompt eval count:    %d token(s)\n", pc)
	}
	if d := dur("prompt_eval_duration"); d > 0 {
		fmt.Fprintf(os.Stderr, "prompt eval duration: %s\n", d)
		fmt.Fprintf(os.Stderr, "prompt eval rate:     %.2f tokens/s\n", float64(pc)/d.Seconds())
	}
	ec := int(num(m, "eval_count"))
	if ec > 0 {
		fmt.Fprintf(os.Stderr, "eval count:           %d token(s)\n", ec)
	}
	if d := dur("eval_duration"); d > 0 {
		fmt.Fprintf(os.Stderr, "eval duration:        %s\n", d)
		fmt.Fprintf(os.Stderr, "eval rate:            %.2f tokens/s\n", float64(ec)/d.Seconds())
	}
}

// ------------------------------------------------------ server helpers

// The format flag goes on the wire as JSON: "json", or a schema object.
func formatValue(f string) any {
	if f == "" {
		return nil
	}
	if f == "json" {
		return "json"
	}
	var v any
	if json.Unmarshal([]byte(f), &v) == nil {
		return v
	}
	return f
}

func (o runOptions) body(extra map[string]any) map[string]any {
	b := map[string]any{"model": o.model}
	if len(o.options) > 0 {
		b["options"] = o.options
	}
	if f := formatValue(o.format); f != nil {
		b["format"] = f
	}
	if o.think != nil {
		b["think"] = o.think
	}
	if o.keepAlive != nil {
		b["keep_alive"] = o.keepAlive
	}
	for k, v := range extra {
		b[k] = v
	}
	return b
}

func showModel(name string) (map[string]any, int) {
	d, r, err := callJSON("POST", "/api/show", map[string]any{"model": name}, 60*time.Second)
	if err != nil {
		die("Error: %v", err)
	}
	return d, r.StatusCode
}

// ollama pulls a model the first time you run one you don't have.
func showOrPull(name string) map[string]any {
	d, code := showModel(name)
	if code == 200 {
		return d
	}
	if code != 404 {
		die("Error: %s", first(str(d, "error"), "could not read that model"))
	}
	cmdPull(name)
	d, code = showModel(name)
	if code != 200 {
		die("Error: %s", first(str(d, "error"), "model not found"))
	}
	return d
}

func caps(info map[string]any) []string {
	var out []string
	for _, c := range list(info, "capabilities") {
		out = append(out, fmt.Sprint(c))
	}
	return out
}

func hasCap(info map[string]any, want string) bool {
	for _, c := range caps(info) {
		if c == want {
			return true
		}
	}
	return false
}

// Thinking is on by default for a model that supports it, unless the flag said
// otherwise.
func inferThinking(info map[string]any, o *runOptions, explicit bool) {
	if explicit {
		return
	}
	if hasCap(info, "thinking") {
		o.think = true
		return
	}
	o.think = nil
}

// A blank generate request loads (or with keep_alive 0, unloads) the model.
func loadOrUnloadModel(o *runOptions) error {
	p := newProgress(os.Stderr)
	defer p.stopAndClear()
	p.add(newSpinner(""))
	body := o.body(map[string]any{"prompt": "", "stream": false})
	d, r, err := callJSON("POST", "/api/generate", body, 10*time.Minute)
	if err != nil {
		return err
	}
	if r.StatusCode >= 400 {
		return errors.New(first(str(d, "error"), r.Status))
	}
	return nil
}

// --------------------------------------------------------------- chat

func interruptible() (context.Context, func()) {
	ctx, cancel := context.WithCancel(context.Background())
	sigc := make(chan os.Signal, 1)
	signal.Notify(sigc, os.Interrupt)
	go func() {
		select {
		case <-sigc:
			cancel()
		case <-ctx.Done():
		}
	}()
	return ctx, func() { signal.Stop(sigc); cancel() }
}

func chatOnce(o runOptions) (message, error) {
	p := newProgress(os.Stderr)
	defer p.stopAndClear()
	p.add(newSpinner(""))
	ctx, cancel := interruptible()
	defer cancel()

	state := &displayState{}
	var thinking, full strings.Builder
	var latest map[string]any
	role := "assistant"
	opened, closed := false, false
	stopped := false
	var apiErr string

	msgs := make([]any, 0, len(o.messages))
	for _, m := range o.messages {
		msgs = append(msgs, map[string]any(m))
	}
	body := o.body(map[string]any{"messages": msgs, "stream": true})

	err := stream(ctx, "/api/chat", body, func(ev map[string]any) bool {
		if e := str(ev, "error"); e != "" {
			apiErr = e
			return false
		}
		m := sub(ev, "message")
		if str(m, "content") != "" || !o.hideThinking {
			if !stopped {
				p.stopAndClear()
				stopped = true
			}
		}
		latest = ev
		if r := str(m, "role"); r != "" {
			role = r
		}
		if t := str(m, "thinking"); t != "" && !o.hideThinking {
			if !opened {
				fmt.Print(thinkOpen(false))
				opened, closed = true, false
			}
			thinking.WriteString(t)
			displayResponse(t, o.wordWrap, state)
		}
		content := str(m, "content")
		calls := list(m, "tool_calls")
		if opened && !closed && (content != "" || len(calls) > 0) {
			if !strings.HasSuffix(thinking.String(), "\n") {
				fmt.Println()
			}
			fmt.Print(thinkClose(false))
			opened, closed = false, true
			state = &displayState{}
		}
		full.WriteString(content)
		if len(calls) > 0 {
			fmt.Print(renderToolCalls(calls, false))
		}
		displayResponse(content, o.wordWrap, state)
		return true
	})
	p.stopAndClear()
	if apiErr != "" {
		return nil, errors.New(apiErr)
	}
	if errors.Is(err, context.Canceled) || ctx.Err() != nil {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	if len(o.messages) > 0 {
		fmt.Println()
		fmt.Println()
	}
	if o.verbose && latest != nil {
		summary(latest)
	}
	return message{"role": role, "thinking": thinking.String(), "content": full.String()}, nil
}

func generateOnce(o runOptions) error {
	p := newProgress(os.Stderr)
	defer p.stopAndClear()
	p.add(newSpinner(""))
	ctx, cancel := interruptible()
	defer cancel()

	state := &displayState{}
	var thinking strings.Builder
	var latest map[string]any
	opened, closed := false, false
	stopped := false
	plain := !isConsole(os.Stdout)
	var apiErr string

	extra := map[string]any{"prompt": o.prompt, "stream": true}
	if o.system != "" {
		extra["system"] = o.system
	}
	if len(o.images) > 0 {
		extra["images"] = o.images
	}
	err := stream(ctx, "/api/generate", o.body(extra), func(ev map[string]any) bool {
		if e := str(ev, "error"); e != "" {
			apiErr = e
			return false
		}
		latest = ev
		content := str(ev, "response")
		if content != "" || !o.hideThinking {
			if !stopped {
				p.stopAndClear()
				stopped = true
			}
		}
		if t := str(ev, "thinking"); t != "" && !o.hideThinking {
			if !opened {
				fmt.Print(thinkOpen(plain))
				opened, closed = true, false
			}
			thinking.WriteString(t)
			displayResponse(t, o.wordWrap, state)
		}
		calls := list(ev, "tool_calls")
		if opened && !closed && (content != "" || len(calls) > 0) {
			if !strings.HasSuffix(thinking.String(), "\n") {
				fmt.Println()
			}
			fmt.Print(thinkClose(plain))
			opened, closed = false, true
			state = &displayState{}
		}
		displayResponse(content, o.wordWrap, state)
		if len(calls) > 0 {
			fmt.Print(renderToolCalls(calls, plain))
		}
		return true
	})
	p.stopAndClear()
	if apiErr != "" {
		return errors.New(apiErr)
	}
	if errors.Is(err, context.Canceled) || ctx.Err() != nil {
		return nil
	}
	if err != nil {
		return err
	}
	if o.prompt != "" {
		fmt.Println()
		fmt.Println()
	}
	if latest == nil {
		return nil
	}
	if done, _ := latest["done"].(bool); !done {
		return nil
	}
	if o.verbose {
		summary(latest)
	}
	return nil
}

func embedOnce(o runOptions, truncate *bool, dimensions int) error {
	body := map[string]any{"model": o.model, "input": o.prompt}
	if o.keepAlive != nil {
		body["keep_alive"] = o.keepAlive
	}
	if truncate != nil {
		body["truncate"] = *truncate
	}
	if dimensions > 0 {
		body["dimensions"] = dimensions
	}
	d, r, err := callJSON("POST", "/api/embed", body, 10*time.Minute)
	if err != nil {
		return err
	}
	if r.StatusCode >= 400 {
		return errors.New(first(str(d, "error"), r.Status))
	}
	embs := list(d, "embeddings")
	if len(embs) == 0 {
		return errors.New("no embeddings returned")
	}
	out, err := json.Marshal(embs[0])
	if err != nil {
		return err
	}
	fmt.Println(string(out))
	return nil
}

// ----------------------------------------------------------- the REPL

var fileRe = regexp.MustCompile(`(?:[a-zA-Z]:)?(?:\./|/|\\)[\S\\ ]+?\.(?i:jpg|jpeg|png|webp|wav)\b`)

func encodeB64(b []byte) string { return base64.StdEncoding.EncodeToString(b) }

func normalizeFilePath(fp string) string {
	return strings.NewReplacer(
		"\\ ", " ", "\\(", "(", "\\)", ")", "\\[", "[", "\\]", "]", "\\{", "{", "\\}", "}",
		"\\$", "$", "\\&", "&", "\\;", ";", "\\'", "'", "\\\\", "\\", "\\*", "*", "\\?", "?", "\\~", "~",
	).Replace(fp)
}

// Splits a prompt into its text and the base64 of any image or audio files
// named in it.
func extractFileData(input string) (string, []string, error) {
	var imgs []string
	for _, fp := range fileRe.FindAllString(input, -1) {
		nfp := normalizeFilePath(fp)
		b, err := os.ReadFile(nfp)
		if err != nil {
			fmt.Printf("Couldn't process image: %q\n", err)
			return "", nil, err
		}
		fmt.Printf("Added image '%s'\n", nfp)
		input = strings.ReplaceAll(input, fp, "")
		imgs = append(imgs, encodeB64(b))
	}
	return strings.TrimSpace(input), imgs, nil
}

const (
	mlNone = iota
	mlPrompt
	mlSystem
)

func usageMain(multiModal bool) {
	e := os.Stderr
	fmt.Fprintln(e, "Available Commands:")
	fmt.Fprintln(e, "  /set            Set session variables")
	fmt.Fprintln(e, "  /show           Show model information")
	fmt.Fprintln(e, "  /load <model>   Load a session or model")
	fmt.Fprintln(e, "  /save <model>   Save your current session")
	fmt.Fprintln(e, "  /clear          Clear session context")
	fmt.Fprintln(e, "  /bye            Exit")
	fmt.Fprintln(e, "  /?, /help       Help for a command")
	fmt.Fprintln(e, "  /? shortcuts    Help for keyboard shortcuts")
	fmt.Fprintln(e, "")
	fmt.Fprintln(e, "Use \"\"\" to begin a multi-line message.")
	if multiModal {
		fmt.Fprintf(e, "Use %s to include .jpg, .png, .webp images, or .wav audio files.\n", filepath.FromSlash("/path/to/file"))
	}
	fmt.Fprintln(e, "")
}

func usageSet() {
	e := os.Stderr
	fmt.Fprintln(e, "Available Commands:")
	fmt.Fprintln(e, "  /set parameter ...     Set a parameter")
	fmt.Fprintln(e, "  /set system <string>   Set system message")
	fmt.Fprintln(e, "  /set history           Enable history")
	fmt.Fprintln(e, "  /set nohistory         Disable history")
	fmt.Fprintln(e, "  /set wordwrap          Enable wordwrap")
	fmt.Fprintln(e, "  /set nowordwrap        Disable wordwrap")
	fmt.Fprintln(e, "  /set format json       Enable JSON mode")
	fmt.Fprintln(e, "  /set noformat          Disable formatting")
	fmt.Fprintln(e, "  /set verbose           Show LLM stats")
	fmt.Fprintln(e, "  /set quiet             Disable LLM stats")
	fmt.Fprintln(e, "  /set think             Enable thinking")
	fmt.Fprintln(e, "  /set nothink           Disable thinking")
	fmt.Fprintln(e, "")
}

func usageShortcuts() {
	e := os.Stderr
	fmt.Fprintln(e, "Available keyboard shortcuts:")
	fmt.Fprintln(e, "  Ctrl + a            Move to the beginning of the line (Home)")
	fmt.Fprintln(e, "  Ctrl + e            Move to the end of the line (End)")
	fmt.Fprintln(e, "   Alt + b            Move back (left) one word")
	fmt.Fprintln(e, "   Alt + f            Move forward (right) one word")
	fmt.Fprintln(e, "  Ctrl + k            Delete the sentence after the cursor")
	fmt.Fprintln(e, "  Ctrl + u            Delete the sentence before the cursor")
	fmt.Fprintln(e, "  Ctrl + w            Delete the word before the cursor")
	fmt.Fprintln(e, "")
	fmt.Fprintln(e, "  Ctrl + l            Clear the screen")
	fmt.Fprintln(e, "  Ctrl + g            Open default editor to compose a prompt")
	fmt.Fprintln(e, "  Ctrl + c            Stop the model from responding")
	fmt.Fprintln(e, "  Ctrl + d            Exit llmash (/bye)")
	fmt.Fprintln(e, "")
}

func usageShow() {
	e := os.Stderr
	fmt.Fprintln(e, "Available Commands:")
	fmt.Fprintln(e, "  /show info         Show details for this model")
	fmt.Fprintln(e, "  /show license      Show model license")
	fmt.Fprintln(e, "  /show modelfile    Show Modelfile for this model")
	fmt.Fprintln(e, "  /show parameters   Show parameters for this model")
	fmt.Fprintln(e, "  /show system       Show system message")
	fmt.Fprintln(e, "  /show template     Show prompt template")
	fmt.Fprintln(e, "")
}

func usageParameters() {
	e := os.Stderr
	fmt.Fprintln(e, "Available Parameters:")
	fmt.Fprintln(e, "  /set parameter seed <int>             Random number seed")
	fmt.Fprintln(e, "  /set parameter num_predict <int>      Max number of tokens to predict")
	fmt.Fprintln(e, "  /set parameter top_k <int>            Pick from top k num of tokens")
	fmt.Fprintln(e, "  /set parameter top_p <float>          Pick token based on sum of probabilities")
	fmt.Fprintln(e, "  /set parameter min_p <float>          Pick token based on top token probability * min_p")
	fmt.Fprintln(e, "  /set parameter num_ctx <int>          Set the context size")
	fmt.Fprintln(e, "  /set parameter temperature <float>    Set creativity level")
	fmt.Fprintln(e, "  /set parameter repeat_penalty <float> How strongly to penalize repetitions")
	fmt.Fprintln(e, "  /set parameter repeat_last_n <int>    Set how far back to look for repetitions")
	fmt.Fprintln(e, "  /set parameter num_gpu <int>          The number of layers to send to the GPU")
	fmt.Fprintln(e, "  /set parameter stop <string> <string> ...   Set the stop parameters")
	fmt.Fprintln(e, "")
}

var intParams = map[string]bool{"seed": true, "num_predict": true, "top_k": true, "num_ctx": true,
	"repeat_last_n": true, "num_gpu": true, "num_keep": true, "num_batch": true, "num_thread": true,
	"main_gpu": true, "mirostat": true}
var boolParams = map[string]bool{"numa": true, "low_vram": true, "use_mmap": true, "use_mlock": true}

// The value types ollama's api.FormatParams gives a `/set parameter`.
func formatParam(key string, vals []string) (any, error) {
	switch {
	case key == "stop":
		return vals, nil
	case intParams[key]:
		n, err := strconv.Atoi(vals[0])
		if err != nil {
			return nil, fmt.Errorf("invalid int value %v", vals)
		}
		return n, nil
	case boolParams[key]:
		b, err := strconv.ParseBool(vals[0])
		if err != nil {
			return nil, fmt.Errorf("invalid bool value %v", vals)
		}
		return b, nil
	}
	f, err := strconv.ParseFloat(vals[0], 64)
	if err != nil {
		return nil, fmt.Errorf("invalid float value %v", vals)
	}
	return f, nil
}

func generateInteractive(o runOptions) {
	ed := newEditor()
	thinkSet := o.think != nil
	fmt.Print(startBracketedPaste)
	defer fmt.Print(endBracketedPaste)

	var sb strings.Builder
	multiline := mlNone

	setSystem := func(text string) {
		o.system = text
		nm := message{"role": "system", "content": text}
		if n := len(o.messages); n > 0 && str(o.messages[n-1], "role") == "system" {
			o.messages[n-1] = nm
		} else {
			o.messages = append(o.messages, nm)
		}
	}

	for {
		line, err := ed.readline()
		switch {
		case errors.Is(err, io.EOF):
			return
		case errors.Is(err, errInterrupt):
			if line == "" {
				fmt.Println("\nUse Ctrl + d or /bye to exit.")
			}
			ed.useAlt = false
			sb.Reset()
			multiline = mlNone
			continue
		case errors.Is(err, errEditPrompt):
			sb.Reset()
			content, err := editInExternalEditor(line)
			if err != nil {
				fmt.Fprintf(os.Stderr, "error: %v\n", err)
				continue
			}
			if strings.TrimSpace(content) == "" {
				continue
			}
			ed.prefill = content
			continue
		case err != nil:
			die("error: %v", err)
		}

		switch {
		case multiline != mlNone:
			before, ok := strings.CutSuffix(line, `"""`)
			sb.WriteString(before)
			if !ok {
				fmt.Fprintln(&sb)
				ed.useAlt = true
				continue
			}
			if multiline == mlSystem {
				setSystem(sb.String())
				fmt.Println("Set system message.")
				sb.Reset()
			}
			multiline = mlNone
			ed.useAlt = false
		case strings.HasPrefix(line, `"""`):
			rest := strings.TrimPrefix(line, `"""`)
			rest, ok := strings.CutSuffix(rest, `"""`)
			sb.WriteString(rest)
			if !ok {
				fmt.Fprintln(&sb)
				multiline = mlPrompt
				ed.useAlt = true
			}
		case ed.pasting:
			fmt.Fprintln(&sb, line)
			continue
		case strings.HasPrefix(line, "/list"):
			args := strings.Fields(line)
			cmdList(args[1:])
		case strings.HasPrefix(line, "/load"):
			args := strings.Fields(line)
			if len(args) != 2 {
				fmt.Println("Usage:\n  /load <modelname>")
				continue
			}
			orig := o.copy()
			o.model = args[1]
			o.messages = nil
			o.loadedMessages = nil
			fmt.Printf("Loading model '%s'\n", o.model)
			info, code := showModel(o.model)
			if code != 200 {
				fmt.Printf("Couldn't find model '%s'\n", o.model)
				o = orig.copy()
				continue
			}
			o.parentModel = str(sub(info, "details"), "parent_model")
			inferThinking(info, &o, thinkSet)
			o.multiModal = hasCap(info, "vision") || hasCap(info, "audio")
			if err := loadOrUnloadModel(&o); err != nil {
				if strings.Contains(err.Error(), "not found") {
					fmt.Printf("Couldn't find model '%s'\n", o.model)
					o = orig.copy()
					continue
				}
				fmt.Printf("error: %v\n", err)
				continue
			}
			continue
		case strings.HasPrefix(line, "/save"):
			args := strings.Fields(line)
			if len(args) != 2 {
				fmt.Println("Usage:\n  /save <modelname>")
				continue
			}
			msgs := make([]any, 0, len(o.loadedMessages)+len(o.messages))
			for _, m := range append(append([]message{}, o.loadedMessages...), o.messages...) {
				msgs = append(msgs, map[string]any(m))
			}
			req := map[string]any{"model": args[1], "from": first(o.parentModel, o.model)}
			if o.system != "" {
				req["system"] = o.system
			}
			if len(o.options) > 0 {
				req["parameters"] = o.options
			}
			if len(msgs) > 0 {
				req["messages"] = msgs
			}
			if e := createModel(req); e != "" {
				fmt.Printf("error: %s\n", e)
				continue
			}
			fmt.Printf("Created new model '%s'\n", args[1])
			continue
		case strings.HasPrefix(line, "/clear"):
			o.messages = nil
			if o.system != "" {
				o.messages = append(o.messages, message{"role": "system", "content": o.system})
			}
			fmt.Println("Cleared session context")
			continue
		case strings.HasPrefix(line, "/set"):
			args := strings.Fields(line)
			if len(args) <= 1 {
				usageSet()
				break
			}
			switch args[1] {
			case "history":
				ed.hist.enabled = true
			case "nohistory":
				ed.hist.enabled = false
			case "wordwrap":
				o.wordWrap = true
				fmt.Println("Set 'wordwrap' mode.")
			case "nowordwrap":
				o.wordWrap = false
				fmt.Println("Set 'nowordwrap' mode.")
			case "verbose":
				o.verbose = true
				fmt.Println("Set 'verbose' mode.")
			case "quiet":
				o.verbose = false
				fmt.Println("Set 'quiet' mode.")
			case "think":
				level := ""
				if len(args) > 2 {
					level = args[2]
				}
				if level != "" {
					o.think = level
					fmt.Printf("Set 'think' mode to '%s'.\n", level)
				} else {
					o.think = true
					fmt.Println("Set 'think' mode.")
				}
				thinkSet = true
				ensureThinkingSupport(o.model)
			case "nothink":
				o.think = false
				thinkSet = true
				ensureThinkingSupport(o.model)
				fmt.Println("Set 'nothink' mode.")
			case "format":
				if len(args) < 3 || args[2] != "json" {
					fmt.Println("Invalid or missing format. For 'json' mode use '/set format json'")
				} else {
					o.format = args[2]
					fmt.Printf("Set format to '%s' mode.\n", args[2])
				}
			case "noformat":
				o.format = ""
				fmt.Println("Disabled format.")
			case "parameter":
				if len(args) < 4 {
					usageParameters()
					continue
				}
				params := args[3:]
				v, err := formatParam(args[2], params)
				if err != nil {
					fmt.Printf("Couldn't set parameter: %q\n", err)
					continue
				}
				fmt.Printf("Set parameter '%s' to '%s'\n", args[2], strings.Join(params, ", "))
				if o.options == nil {
					o.options = map[string]any{}
				}
				o.options[args[2]] = v
			case "system":
				if len(args) < 3 {
					usageSet()
					continue
				}
				multiline = mlSystem
				rest := strings.Join(args[2:], " ")
				rest, ok := strings.CutPrefix(rest, `"""`)
				if !ok {
					multiline = mlNone
				} else if rest, ok = strings.CutSuffix(rest, `"""`); ok {
					multiline = mlNone
				}
				sb.WriteString(rest)
				if multiline != mlNone {
					ed.useAlt = true
					continue
				}
				setSystem(sb.String())
				fmt.Println("Set system message.")
				sb.Reset()
				continue
			default:
				fmt.Printf("Unknown command '/set %s'. Type /? for help\n", args[1])
			}
		case strings.HasPrefix(line, "/show"):
			args := strings.Fields(line)
			if len(args) <= 1 {
				usageShow()
				break
			}
			info, code := showModel(o.model)
			if code != 200 {
				fmt.Println("error: couldn't get model")
				continue
			}
			switch args[1] {
			case "info":
				showInfo(info, false, os.Stderr)
			case "license":
				if str(info, "license") == "" {
					fmt.Println("No license was specified for this model.")
				} else {
					fmt.Println(str(info, "license"))
				}
			case "modelfile":
				fmt.Println(str(info, "modelfile"))
			case "parameters":
				fmt.Println("Model defined parameters:")
				if str(info, "parameters") == "" {
					fmt.Println("  No additional parameters were specified for this model.")
				} else {
					for _, l := range strings.Split(str(info, "parameters"), "\n") {
						fmt.Printf("  %s\n", l)
					}
				}
				fmt.Println()
				if len(o.options) > 0 {
					fmt.Println("User defined parameters:")
					keys := make([]string, 0, len(o.options))
					for k := range o.options {
						keys = append(keys, k)
					}
					sort.Strings(keys)
					for _, k := range keys {
						fmt.Printf("  %-*s %v\n", 30, k, o.options[k])
					}
					fmt.Println()
				}
			case "system":
				switch {
				case o.system != "":
					fmt.Println(o.system + "\n")
				case str(info, "system") != "":
					fmt.Println(str(info, "system") + "\n")
				default:
					fmt.Println("No system message was specified for this model.")
				}
			case "template":
				if str(info, "template") != "" {
					fmt.Println(str(info, "template"))
				} else {
					fmt.Println("No prompt template was specified for this model.")
				}
			default:
				fmt.Printf("Unknown command '/show %s'. Type /? for help\n", args[1])
			}
		case strings.HasPrefix(line, "/help"), strings.HasPrefix(line, "/?"):
			args := strings.Fields(line)
			if len(args) > 1 {
				switch args[1] {
				case "set", "/set":
					usageSet()
				case "show", "/show":
					usageShow()
				case "shortcut", "shortcuts":
					usageShortcuts()
				}
			} else {
				usageMain(o.multiModal)
			}
		case strings.HasPrefix(line, "/exit"), strings.HasPrefix(line, "/bye"):
			return
		case strings.HasPrefix(line, "/"):
			args := strings.Fields(line)
			isFile := false
			if o.multiModal {
				for _, f := range fileRe.FindAllString(line, -1) {
					if strings.HasPrefix(f, args[0]) {
						isFile = true
						break
					}
				}
			}
			if !isFile {
				fmt.Printf("Unknown command '%s'. Type /? for help\n", args[0])
				continue
			}
			sb.WriteString(line)
		default:
			sb.WriteString(line)
		}

		if sb.Len() > 0 && multiline == mlNone {
			nm := message{"role": "user", "content": sb.String()}
			if o.multiModal {
				text, imgs, err := extractFileData(sb.String())
				if err != nil {
					sb.Reset()
					continue
				}
				nm["content"] = text
				if len(imgs) > 0 {
					nm["images"] = imgs
				}
			}
			o.messages = append(o.messages, nm)
			assistant, err := chatOnce(o)
			if err != nil {
				if strings.Contains(err.Error(), "does not support thinking") ||
					strings.Contains(err.Error(), "invalid think value") {
					fmt.Printf("error: %v\n", err)
					o.messages = o.messages[:len(o.messages)-1]
					sb.Reset()
					continue
				}
				fmt.Fprintf(os.Stderr, "error: %v\n", err)
				o.messages = o.messages[:len(o.messages)-1]
				sb.Reset()
				continue
			}
			if assistant != nil {
				o.messages = append(o.messages, assistant)
			}
			sb.Reset()
		}
	}
}

func ensureThinkingSupport(model string) {
	info, code := showModel(model)
	if code != 200 {
		return
	}
	if !hasCap(info, "thinking") {
		fmt.Fprintf(os.Stderr, "warning: model %q does not support thinking output\n", model)
	}
}

func createModel(req map[string]any) string {
	var bad string
	err := stream(context.Background(), "/api/create", req, func(ev map[string]any) bool {
		if e := str(ev, "error"); e != "" {
			bad = e
			return false
		}
		return true
	})
	if bad != "" {
		return bad
	}
	if err != nil {
		return err.Error()
	}
	return ""
}

// --------------------------------------------------------------- entry

func cmdRun(o runOpts) {
	needServer()
	interactive := true
	opts := runOptions{
		model:        o.model,
		options:      map[string]any{},
		wordWrap:     !o.nowordwrap,
		showConnect:  true,
		format:       o.format,
		hideThinking: o.hidethinking,
		verbose:      o.verbose,
	}
	switch {
	case !o.thinkSet:
	case o.think == "true":
		opts.think = true
	case o.think == "false":
		opts.think = false
	default:
		opts.think = o.think
	}
	if o.ctx > 0 {
		opts.options["num_ctx"] = o.ctx
	}
	if o.temperature != nil {
		opts.options["temperature"] = *o.temperature
	}
	if o.keepalive != "" {
		opts.keepAlive = o.keepalive
	}

	prompt := o.prompt
	if !isConsole(os.Stdin) {
		in, err := io.ReadAll(os.Stdin)
		if err != nil {
			die("Error: %v", err)
		}
		if len(in) > 0 {
			prompt = strings.TrimRight(string(in), "\r\n") + " " + prompt
			prompt = strings.TrimSpace(prompt)
		}
		opts.showConnect = false
		opts.wordWrap = false
		interactive = false
	}
	opts.prompt = prompt
	if prompt != "" {
		interactive = false
	}
	if !isConsole(os.Stdout) {
		interactive = false
	}

	info := showOrPull(opts.model)
	opts.parentModel = str(sub(info, "details"), "parent_model")
	inferThinking(info, &opts, o.thinkSet)
	opts.multiModal = hasCap(info, "vision") || hasCap(info, "audio")

	if hasCap(info, "embedding") {
		if opts.prompt == "" {
			die("Error: embedding models require input text. Usage: %s run %s \"your text here\"", prog, opts.model)
		}
		if err := embedOnce(opts, o.truncate, o.dimensions); err != nil {
			die("Error: %v", err)
		}
		return
	}

	if interactive {
		if err := loadOrUnloadModel(&opts); err != nil {
			die("Error: %v", err)
		}
		for _, m := range opts.loadedMessages {
			fmt.Printf("%s: %s\n\n", str(m, "role"), str(m, "content"))
		}
		generateInteractive(opts)
		return
	}
	if opts.multiModal {
		text, imgs, err := extractFileData(opts.prompt)
		if err != nil {
			exit(1)
		}
		opts.prompt, opts.images = text, imgs
	}
	if err := generateOnce(opts); err != nil {
		die("Error: %v", err)
	}
}
