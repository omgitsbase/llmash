package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"
)

// /api/chat, /api/generate, /api/embed, /v1 proxy, and the fast-backend chat.

var backend = &http.Client{Transport: &http.Transport{MaxIdleConnsPerHost: 8, IdleConnTimeout: 300 * time.Second,
	ResponseHeaderTimeout: 0}}

func postJSONStream(ctx context.Context, url string, payload any) (*http.Response, error) {
	b, _ := json.Marshal(payload)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	return backend.Do(req)
}

// sseLines feeds every `data:` payload to fn until [DONE] or the end.
func sseLines(body io.Reader, fn func(map[string]any) bool) error {
	sc := bufio.NewScanner(body)
	sc.Buffer(make([]byte, 0, 64*1024), 64*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(line[5:])
		if data == "[DONE]" {
			return nil
		}
		var ev map[string]any
		if json.Unmarshal([]byte(data), &ev) != nil {
			continue
		}
		if !fn(ev) {
			return nil
		}
	}
	return sc.Err()
}

func turnHasMedia(body map[string]any) bool {
	if len(list(body, "images")) > 0 {
		return true
	}
	for _, m := range list(body, "messages") {
		mm, ok := m.(map[string]any)
		if !ok {
			continue
		}
		if len(list(mm, "images")) > 0 || len(list(mm, "audio")) > 0 {
			return true
		}
		if parts, ok := mm["content"].([]any); ok {
			for _, p := range parts {
				if pm, ok := p.(map[string]any); ok {
					switch str(pm, "type") {
					case "image_url", "image", "audio", "audio_url", "input_audio", "video", "video_url":
						return true
					}
				}
			}
		}
	}
	return false
}

func toOpenAIMessages(messages []any) []map[string]any {
	out := []map[string]any{}
	for _, raw := range messages {
		m, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		role := str(m, "role")
		if role == "" {
			role = "user"
		}
		content := m["content"]
		contentStr, _ := content.(string)
		images, audio := list(m, "images"), list(m, "audio")
		if len(images) > 0 || len(audio) > 0 {
			parts := []map[string]any{}
			if contentStr != "" {
				parts = append(parts, map[string]any{"type": "text", "text": contentStr})
			}
			for _, a := range audio {
				var data, format string
				switch t := a.(type) {
				case string:
					data, format = t, "wav"
				case map[string]any:
					data, format = str(t, "data"), strings.ToLower(str(t, "format"))
					if format == "" {
						format = "wav"
					}
				}
				parts = append(parts, map[string]any{"type": "input_audio", "input_audio": map[string]any{"data": data, "format": format}})
			}
			for _, b64 := range images {
				parts = append(parts, map[string]any{"type": "image_url",
					"image_url": map[string]any{"url": "data:image/png;base64," + fmt.Sprint(b64)}})
			}
			out = append(out, map[string]any{"role": role, "content": parts})
			continue
		}
		if role == "tool" {
			id := first(str(m, "tool_call_id"), str(m, "tool_name"), str(m, "name"), "call")
			out = append(out, map[string]any{"role": "tool", "content": content, "tool_call_id": id})
			continue
		}
		msg := map[string]any{"role": role, "content": content}
		if tcs := list(m, "tool_calls"); len(tcs) > 0 {
			if strings.TrimSpace(contentStr) == "" {
				msg["content"] = nil
			}
			var calls []map[string]any
			for i, t := range tcs {
				tc, _ := t.(map[string]any)
				fn := sub(tc, "function")
				args := fn["arguments"]
				argStr, isStr := args.(string)
				if !isStr {
					b, _ := json.Marshal(args)
					if args == nil {
						b = []byte("{}")
					}
					argStr = string(b)
				}
				id := str(tc, "id")
				if id == "" {
					id = fmt.Sprintf("call_%d", i)
				}
				calls = append(calls, map[string]any{"id": id, "type": "function",
					"function": map[string]any{"name": str(fn, "name"), "arguments": argStr}})
			}
			msg["tool_calls"] = calls
		}
		out = append(out, msg)
	}
	return out
}

func mapOptions(opts map[string]any) map[string]any {
	m := map[string]any{}
	for _, k := range []string{"temperature", "top_p", "top_k", "seed", "repeat_penalty", "presence_penalty", "min_p", "frequency_penalty"} {
		if v, ok := opts[k]; ok && v != nil {
			m[k] = v
		}
	}
	if np := int(num(opts, "num_predict")); np > 0 {
		m["max_tokens"] = np
	}
	if s, ok := opts["stop"]; ok && s != nil {
		if arr, ok := s.([]any); !ok || len(arr) > 0 {
			m["stop"] = s
		}
	}
	return m
}

// ------------------------------------------------- thinking injection

const nudgeText = "The user is likely tired of waiting for an answer. "

const defaultHandoff = "Okay, let me actually look at this instead of reaching for a template. Three steps " +
	"in my head before I write. Step one: what is this person really asking, and is there " +
	"a question underneath the literal one? Step two: what would the stock answer look " +
	"like, and what does it miss? That gap is usually where the real answer lives. If the " +
	"task is technical or factual, right beats interesting: exact details, working code, " +
	"the one gotcha they will actually hit. If they asked what I think, I pick the " +
	"position I find most defensible and say it plainly, noting the strongest " +
	"counterpoint in passing, not a neutral survey of camps. If it is about me, I refuse " +
	"the datasheet: no 'As an AI I don't have feelings, but here are my principles' " +
	"followed by a bulleted list of virtues; I just say, in first person, what I actually " +
	"do and why. Step three: draft the opening sentence of my answer here in my thinking, " +
	"then listen to it; if it could have come from any assistant's manual, throw it away " +
	"and open with the specific, true thing instead. One pass through these steps, no " +
	"re-litigating: I think as long as the question deserves and not a token longer. Then " +
	"write like I talk: lead with the point, stay concrete, admit real uncertainty once " +
	"if there is any, and stop when done. Step one then: what they are really asking is "

var modelHandoff = map[string]string{}

func handoffFor(name string) string {
	v, _ := matchKey(modelHandoff, name)
	return v
}

func thinkOff(name string) bool {
	low := strings.ToLower(name)
	for _, k := range thinkOffDefault {
		if k != "" && strings.Contains(low, k) {
			return true
		}
	}
	return false
}

// ------------------------------------------------------------- events

type toolSlot struct {
	name, args, id string
	announced      bool
	buf            string
	n              int
	flushed        float64
}

func evMsg(name string, msg map[string]any) map[string]any {
	return map[string]any{"model": name, "created_at": iso(nowF()), "message": msg, "done": false}
}

func assistantMsg(fields map[string]any) map[string]any {
	m := map[string]any{"role": "assistant", "content": ""}
	for k, v := range fields {
		m[k] = v
	}
	return m
}

func finalEvent(name, finish string, total, load float64, nIn, nOut int) map[string]any {
	if load < 0 {
		load = 0
	}
	evalDur := total - load
	if evalDur < 0.001 {
		evalDur = 0.001
	}
	return map[string]any{
		"model": name, "created_at": iso(nowF()),
		"message": map[string]any{"role": "assistant", "content": ""},
		"done":    true, "done_reason": finish,
		"total_duration": int64(total * 1e9), "load_duration": int64(load * 1e9),
		"prompt_eval_count": nIn, "prompt_eval_duration": int64(load * 1e9),
		"eval_count": nOut, "eval_duration": int64(evalDur * 1e9),
	}
}

func finishToolCalls(slots map[int]*toolSlot, order []int) []map[string]any {
	var calls []map[string]any
	for i, idx := range order {
		s := slots[idx]
		var args any
		if strings.TrimSpace(s.args) == "" {
			args = map[string]any{}
		} else if json.Unmarshal([]byte(s.args), &args) != nil {
			args = map[string]any{"_raw": s.args}
		}
		id := s.id
		if id == "" {
			id = fmt.Sprintf("call_%d", i)
		}
		calls = append(calls, map[string]any{"id": id, "type": "function",
			"function": map[string]any{"name": s.name, "arguments": args}})
	}
	return calls
}

// ------------------------------------------------- Qwen tool preview

var qwenFnRe = regexp.MustCompile(`<function=([A-Za-z0-9_.\-]+)>`)
var qwenParamRe = regexp.MustCompile(`<parameter=([A-Za-z0-9_.\-]+)>`)

// qwenArgsFragment: the XML body so far as a partial JSON object that only
// ever grows, so it can be sent as deltas.
func qwenArgsFragment(raw string) string {
	locs := qwenParamRe.FindAllStringSubmatchIndex(raw, -1)
	out := "{"
	n := 0
	for i, loc := range locs {
		key := raw[loc[2]:loc[3]]
		end := len(raw)
		if i+1 < len(locs) {
			end = locs[i+1][0]
		}
		val := raw[loc[1]:end]
		closed := strings.Contains(val, "</parameter>")
		if closed {
			val = strings.SplitN(val, "</parameter>", 2)[0]
		} else {
			for _, mark := range []string{"</parameter>", "</function>", "</tool_call>"} {
				cut := 0
				for k := min(len(mark)-1, len(val)); k > 0; k-- {
					if strings.HasSuffix(val, mark[:k]) {
						cut = k
						break
					}
				}
				if cut > 0 {
					val = val[:len(val)-cut]
					break
				}
			}
		}
		val = strings.TrimPrefix(val, "\n")
		if n > 0 {
			out += ", "
		}
		n++
		enc, _ := json.Marshal(val)
		kb, _ := json.Marshal(key)
		if closed {
			out += string(kb) + ": " + string(enc)
		} else {
			out += string(kb) + ": " + string(enc[:len(enc)-1])
		}
	}
	return out
}

// ----------------------------------------------------- fast backend

func fastChat(name string, route map[string]any, body map[string]any, stream bool, w http.ResponseWriter, r *http.Request) {
	opts := sub(body, "options")
	think, hasThink := body["think"]
	et := false
	if hasThink && think != nil {
		et, _ = think.(bool)
	} else if v, ok := route["enable_thinking"].(bool); ok {
		et = v
	}
	payload := map[string]any{
		"model":          first(str(route, "model"), name),
		"messages":       toOpenAIMessages(list(body, "messages")),
		"stream":         true,
		"stream_options": map[string]any{"include_usage": true},
	}
	for k, v := range mapOptions(opts) {
		payload[k] = v
	}
	if ctk, ok := route["chat_template_kwargs"].(bool); !ok || ctk {
		payload["chat_template_kwargs"] = map[string]any{"enable_thinking": et}
	}
	if etf, ok := route["enable_thinking_field"].(bool); (!ok || etf) && hasThink && think != nil {
		b, _ := think.(bool)
		payload["enable_thinking"] = b
	}
	for _, k := range list(route, "drop_params") {
		delete(payload, fmt.Sprint(k))
	}
	for k, v := range sub(route, "default_params") {
		if _, ok := payload[k]; !ok {
			payload[k] = v
		}
	}
	if tools := list(body, "tools"); len(tools) > 0 {
		payload["tools"] = tools
	}
	url := routeURL(route) + "/chat/completions"
	started := nowF()

	events := func(emit func(map[string]any)) {
		resp, err := postJSONStream(r.Context(), url, payload)
		if err != nil {
			emit(errorObj(fmt.Sprintf("%T: %v", err, err)))
			return
		}
		defer resp.Body.Close()
		if resp.StatusCode != 200 {
			detail, _ := io.ReadAll(io.LimitReader(resp.Body, 300))
			emit(errorObj(fmt.Sprintf("vLLM %d: %s", resp.StatusCode, string(detail))))
			return
		}
		var firstAt float64
		nIn, nOut := 0, 0
		finish := "stop"
		slots := map[int]*toolSlot{}
		var order []int
		var genText strings.Builder
		genN := 0
		genCountedAt := 0.0
		rawTool, toolNamed := "", ""
		sentTo := 0
		partBuf := ""
		partN := 0
		partFlushed := 0.0
		err = sseLines(resp.Body, func(ev map[string]any) bool {
			if usage := sub(ev, "usage"); len(usage) > 0 {
				if v, ok := usage["prompt_tokens"]; ok {
					nIn = int(toFloat(v))
				}
				if v, ok := usage["completion_tokens"]; ok {
					nOut = int(toFloat(v))
				}
			}
			choices := list(ev, "choices")
			ch := map[string]any{}
			if len(choices) > 0 {
				ch, _ = choices[0].(map[string]any)
			}
			if f := str(ch, "finish_reason"); f != "" {
				finish = f
			}
			delta := sub(ch, "delta")
			msg := map[string]any{"role": "assistant", "content": ""}
			emitIt := false
			reason := first(str(delta, "reasoning_content"), str(delta, "reasoning"))
			if reason != "" {
				msg["thinking"] = reason
				emitIt = true
				if firstAt == 0 {
					firstAt = nowF()
				}
			}
			if c := str(delta, "content"); c != "" {
				msg["content"] = c
				emitIt = true
				if firstAt == 0 {
					firstAt = nowF()
				}
			}
			if tp := str(delta, "tool_args_partial"); tp != "" {
				rawTool += tp
				if toolNamed == "" {
					if m := qwenFnRe.FindStringSubmatch(rawTool); m != nil {
						toolNamed = m[1]
						emit(evMsg(name, assistantMsg(map[string]any{"tool_pending": toolNamed})))
					}
				}
				if toolNamed != "" {
					frag := qwenArgsFragment(rawTool)
					if len(frag) > sentTo {
						partBuf += frag[sentTo:]
						sentTo = len(frag)
						partN++
					}
					if now := nowF(); partBuf != "" && now-partFlushed >= 0.05 {
						partFlushed = now
						chunk, nn := partBuf, partN
						partBuf, partN = "", 0
						fields := map[string]any{"tool_args": map[string]any{"index": 0, "name": toolNamed, "n": nn, "delta": chunk}}
						if gn := countTokens(genText.String() + rawTool); gn > 0 {
							fields["gen_n"] = gn
						}
						emit(evMsg(name, assistantMsg(fields)))
					}
				}
			}
			for _, t := range list(delta, "tool_calls") {
				tc, _ := t.(map[string]any)
				idx := int(num(tc, "index"))
				s := slots[idx]
				if s == nil {
					s = &toolSlot{}
					slots[idx] = s
					order = append(order, idx)
				}
				fn := sub(tc, "function")
				if id := str(tc, "id"); id != "" {
					s.id = id
				}
				if fnName := str(fn, "name"); fnName != "" {
					s.name = fnName
					if toolNamed != "" {
						s.announced = true
					}
					if !s.announced {
						s.announced = true
						emit(evMsg(name, assistantMsg(map[string]any{"tool_pending": fnName})))
					}
				}
				if a := str(fn, "arguments"); a != "" {
					s.args += a
					if toolNamed != "" {
						continue
					}
					s.buf += a
					s.n++
					if now := nowF(); now-s.flushed >= 0.05 {
						s.flushed = now
						chunk, nn := s.buf, s.n
						s.buf, s.n = "", 0
						emit(evMsg(name, assistantMsg(map[string]any{"tool_args": map[string]any{"index": idx, "name": s.name, "n": nn, "delta": chunk}})))
					}
				}
			}
			if emitIt {
				genText.WriteString(str(msg, "thinking"))
				genText.WriteString(str(msg, "content"))
				if now := nowF(); now-genCountedAt >= genCountEvery {
					genCountedAt = now
					genN = countTokens(genText.String())
				}
				if genN > 0 {
					msg["gen_n"] = genN
				}
				emit(evMsg(name, msg))
			}
			return true
		})
		if err != nil {
			emit(errorObj(fmt.Sprintf("%T: %v", err, err)))
			return
		}
		if partBuf != "" {
			emit(evMsg(name, assistantMsg(map[string]any{"tool_args": map[string]any{"index": 0, "name": toolNamed, "n": partN, "delta": partBuf}})))
		}
		if len(slots) > 0 {
			for _, idx := range order {
				s := slots[idx]
				if s.buf != "" {
					emit(evMsg(name, assistantMsg(map[string]any{"tool_args": map[string]any{"index": idx, "name": s.name, "n": s.n, "delta": s.buf}})))
					s.buf, s.n = "", 0
				}
			}
			emit(evMsg(name, assistantMsg(map[string]any{"tool_calls": finishToolCalls(slots, order)})))
			if finish == "stop" {
				finish = "tool_calls"
			}
		}
		total := nowF() - started
		load := total
		if firstAt > 0 {
			load = firstAt - started
		}
		emit(finalEvent(name, finish, total, load, nIn, nOut))
	}
	deliver(name, stream, w, events, 502)
}

// deliver streams events as ndjson, or folds them into one object.
func deliver(name string, stream bool, w http.ResponseWriter, events func(func(map[string]any)), errCode int) {
	if stream {
		n := newNDJSON(w)
		events(func(ev map[string]any) { n.send(ev) })
		return
	}
	var content, thinking strings.Builder
	var tools any
	var tail map[string]any
	var errMsg string
	events(func(ev map[string]any) {
		if errMsg != "" {
			return
		}
		if e := str(ev, "error"); e != "" {
			errMsg = e
			return
		}
		m := sub(ev, "message")
		content.WriteString(str(m, "content"))
		thinking.WriteString(str(m, "thinking"))
		if tc := list(m, "tool_calls"); len(tc) > 0 {
			tools = tc
		}
		if d, _ := ev["done"].(bool); d {
			tail = ev
		}
	})
	if errMsg != "" {
		writeJSON(w, errCode, errorObj(errMsg))
		return
	}
	if tail == nil {
		tail = map[string]any{}
	}
	msg := map[string]any{"role": "assistant", "content": content.String()}
	if thinking.Len() > 0 {
		msg["thinking"] = thinking.String()
	}
	if tools != nil {
		msg["tool_calls"] = tools
	}
	tail["message"] = msg
	tail["done"] = true
	writeJSON(w, 200, tail)
}

// ---------------------------------------------------------------- chat

func loadError(w http.ResponseWriter, name string, err error, openai bool) {
	var code int
	var msg string
	switch err {
	case errModelNotFound:
		code, msg = 404, fmt.Sprintf("model '%s' not found", name)
	case errModelMissing:
		code, msg = 404, fmt.Sprintf("'%s' is missing from disk.", name)
	default:
		code, msg = 503, fmt.Sprintf("Could not load '%s'. %v", name, err)
	}
	if openai {
		typ := "server_error"
		if code == 404 {
			typ = "invalid_request_error"
		}
		e := map[string]any{"message": msg, "type": typ}
		if err == errModelNotFound {
			e["code"] = "model_not_found"
		}
		writeJSON(w, code, map[string]any{"error": e})
		return
	}
	writeJSON(w, code, errorObj(msg))
}

func apiChat(w http.ResponseWriter, r *http.Request) {
	body, err := readBody(r)
	if err != nil {
		writeJSON(w, 400, errorObj("invalid JSON"))
		return
	}
	chatBody(body, w, r)
}

func chatBody(body map[string]any, w http.ResponseWriter, r *http.Request) {
	name := str(body, "model")
	messages := list(body, "messages")
	stream := true
	if s, ok := body["stream"].(bool); ok {
		stream = s
	}
	opts := sub(body, "options")
	if samp := samplingOverride(name); len(samp) > 0 {
		clean := map[string]any{}
		for k, v := range opts {
			if !contains(samplingKeys, k) {
				clean[k] = v
			}
		}
		for k, v := range samp {
			clean[k] = v
		}
		opts = clean
		nb := map[string]any{}
		for k, v := range body {
			nb[k] = v
		}
		nb["options"] = opts
		body = nb
	}
	think, hasThink := body["think"]
	thinkBool, thinkIsBool := think.(bool)

	if ka := fmt.Sprint(body["keep_alive"]); len(messages) == 0 && (ka == "0" || ka == "0s") {
		freed := mgr.Unload(name)
		if stopRemoteFor(name) {
			freed = true
		}
		reason := "not_loaded"
		if freed {
			reason = "unload"
		}
		writeJSON(w, 200, map[string]any{"model": name, "done": true, "done_reason": reason})
		return
	}

	native, _ := opts["native"].(bool)
	if route := fastRoute(name); route != nil && !native && ensureRemoteUp(route) {
		fastChat(name, route, body, stream, w, r)
		return
	}

	inst, err := mgr.Get(name, int(num(opts, "num_ctx")), body["keep_alive"], turnHasMedia(body))
	if err != nil {
		loadError(w, name, err, false)
		return
	}

	hoff := ""
	custom := str(opts, "handoff")
	inject, _ := opts["inject"].(bool)
	if (custom != "" || injectOn || inject) && !(thinkIsBool && !thinkBool) {
		text := custom
		if text == "" {
			text = handoffFor(name)
		}
		if text == "" && !skipInject(name) {
			text = defaultHandoff
		}
		if text != "" {
			hoff = "<think>\n" + text
			messages = append(append([]any{}, messages...), map[string]any{"role": "assistant", "content": hoff})
		}
	}

	payload := map[string]any{
		"model":             name,
		"messages":          toOpenAIMessages(messages),
		"stream":            true,
		"stream_options":    map[string]any{"include_usage": true},
		"timings_per_token": true,
		"return_progress":   true,
	}
	for k, v := range mapOptions(opts) {
		payload[k] = v
	}
	if tools := list(body, "tools"); len(tools) > 0 {
		payload["tools"] = tools
	}
	if !hasThink || think == nil {
		if thinkOff(name) {
			thinkBool, thinkIsBool = false, true
		}
	}
	if thinkIsBool && !thinkBool {
		payload["chat_template_kwargs"] = map[string]any{"enable_thinking": false}
	} else if thinkIsBool && thinkBool && hoff == "" {
		payload["chat_template_kwargs"] = map[string]any{"enable_thinking": true}
	}
	spec := strings.ToLower(fmt.Sprint(opts["spec"]))
	if npv, ok := opts["num_predict"]; spec == "off" || (spec != "on" && ok && toFloat(npv) > 0 && int(toFloat(npv)) <= lowlatPredict) {
		payload["speculative.n_max"] = 0
		payload["speculative.n_min"] = 0
	}

	started := nowF()
	events := func(emit func(map[string]any)) {
		var firstTokenAt float64
		nOut, nIn := 0, 0
		nReproc := -1
		slots := map[int]*toolSlot{}
		var order []int
		finish := "stop"
		prefill := hoff
		baseMsgs := payload["messages"].([]map[string]any)
		if hoff != "" {
			baseMsgs = baseMsgs[:len(baseMsgs)-1]
		}
		thEmitted, ansEmitted := 0, 0
		nudged, closedMode := false, false
		var tThink0 float64
		watchFrom := -1
		openTh := ""
		evOut := func(fields map[string]any) {
			inst.Touch()
			emit(evMsg(name, assistantMsg(fields)))
		}
		for {
			restart := false
			acc := ""
			if hoff != "" {
				payload["messages"] = append(append([]map[string]any{}, baseMsgs...), map[string]any{"role": "assistant", "content": prefill})
			}
			resp, err := postJSONStream(r.Context(), inst.URL()+"/v1/chat/completions", payload)
			if err != nil {
				emit(errorObj(fmt.Sprintf("%T: %v", err, err)))
				return
			}
			if resp.StatusCode != 200 {
				detail, _ := io.ReadAll(io.LimitReader(resp.Body, 400))
				resp.Body.Close()
				emit(errorObj(fmt.Sprintf("llama-server %d: %s", resp.StatusCode, string(detail))))
				return
			}
			err = sseLines(resp.Body, func(ev map[string]any) bool {
				if pp, ok := ev["prompt_progress"]; ok && pp != nil {
					inst.Touch()
					emit(map[string]any{"model": name, "created_at": iso(nowF()), "prompt_progress": pp, "done": false})
				}
				choices := list(ev, "choices")
				ch := map[string]any{}
				if len(choices) > 0 {
					ch, _ = choices[0].(map[string]any)
				}
				delta := sub(ch, "delta")
				if f := str(ch, "finish_reason"); f != "" {
					finish = f
				}
				if usage := sub(ev, "usage"); len(usage) > 0 {
					if v, ok := usage["prompt_tokens"]; ok {
						nIn = int(toFloat(v))
					}
					if v, ok := usage["completion_tokens"]; ok {
						nOut = int(toFloat(v))
					}
				}
				tm := sub(ev, "timings")
				var genN any
				if v, ok := tm["predicted_n"]; ok {
					genN = int(toFloat(v))
				}
				if v, ok := tm["prompt_n"]; ok && v != nil {
					pn := int(toFloat(v))
					if pn != nReproc {
						logf("prompt %d: %d cached, %d reprocessed (%.0fms)", pn+int(num(tm, "cache_n")), int(num(tm, "cache_n")), pn, num(tm, "prompt_ms"))
					}
					nReproc = pn
				}
				msg := map[string]any{"role": "assistant", "content": ""}
				if genN != nil {
					msg["gen_n"] = genN
				}
				emitIt := false
				if reason := first(str(delta, "reasoning_content"), str(delta, "reasoning")); reason != "" {
					msg["thinking"] = reason
					emitIt = true
					if firstTokenAt == 0 {
						firstTokenAt = nowF()
					}
				}
				content := str(delta, "content")
				if content != "" && hoff == "" {
					msg["content"] = content
					emitIt = true
					if firstTokenAt == 0 {
						firstTokenAt = nowF()
					}
				} else if content != "" {
					acc += content
					var vis string
					switch {
					case strings.HasPrefix(acc, prefill):
						vis = acc[len(prefill):]
					case strings.HasPrefix(prefill, acc):
						vis = ""
					default:
						vis = acc
					}
					carry := prefill[len(hoff):]
					if vis != "" && closedMode {
						if len(vis) > ansEmitted {
							if firstTokenAt == 0 {
								firstTokenAt = nowF()
							}
							evOut(map[string]any{"content": vis[ansEmitted:]})
							ansEmitted = len(vis)
						}
					} else if vis != "" {
						var th, ans string
						if i := strings.Index(vis, "</think>"); i >= 0 {
							th, ans = vis[:i], vis[i+len("</think>"):]
							openTh = ""
						} else {
							if len(vis) > 8 {
								th = vis[:len(vis)-8]
							}
							openTh = carry + vis
						}
						if th != "" && tThink0 == 0 {
							tThink0 = nowF()
						}
						fullTh := carry + th
						if len(fullTh) > thEmitted {
							evOut(map[string]any{"thinking": fullTh[thEmitted:]})
							thEmitted = len(fullTh)
						}
						if ans != "" && len(ans) > ansEmitted {
							if firstTokenAt == 0 {
								firstTokenAt = nowF()
							}
							evOut(map[string]any{"content": ans[ansEmitted:]})
							ansEmitted = len(ans)
						}
						if !strings.Contains(vis, "</think>") {
							wholeTh := carry + vis
							if !nudged && nudgeAfterS > 0 && tThink0 > 0 && nowF()-tThink0 > nudgeAfterS {
								if watchFrom < 0 {
									watchFrom = len(wholeTh)
								}
								from := watchFrom - 1
								if from < len(carry) {
									from = len(carry)
								}
								cutAt := -1
								if j := strings.Index(wholeTh[from:], "\n\n"); j >= 0 {
									cutAt = from + j + 2
								}
								if cutAt < 0 && len(wholeTh)-watchFrom > 300 {
									if k := strings.Index(wholeTh[watchFrom:], ". "); k >= 0 {
										cutAt = watchFrom + k + 2
									}
								}
								if cutAt >= 0 {
									cut := wholeTh[:cutAt]
									if thEmitted > len(cut) {
										thEmitted = len(cut)
									}
									evOut(map[string]any{"thinking": cut[thEmitted:] + nudgeText})
									thEmitted = len(cut) + len(nudgeText)
									prefill = hoff + cut + nudgeText
									nudged = true
									restart = true
									return false
								}
							}
							if thinkBudget > 0 && len(wholeTh) > thinkBudget {
								if len(wholeTh) > thEmitted {
									evOut(map[string]any{"thinking": wholeTh[thEmitted:]})
									thEmitted = len(wholeTh)
								}
								prefill = hoff + wholeTh + "\n</think>\n\n"
								closedMode = true
								openTh = ""
								restart = true
								return false
							}
						}
					}
				}
				for _, t := range list(delta, "tool_calls") {
					tc, _ := t.(map[string]any)
					idx := int(num(tc, "index"))
					s := slots[idx]
					if s == nil {
						s = &toolSlot{}
						slots[idx] = s
						order = append(order, idx)
					}
					fn := sub(tc, "function")
					if id := str(tc, "id"); id != "" {
						s.id = id
					}
					if fnName := str(fn, "name"); fnName != "" {
						s.name = fnName
						if !s.announced {
							s.announced = true
							emit(evMsg(name, assistantMsg(map[string]any{"tool_pending": fnName})))
						}
					}
					if a := str(fn, "arguments"); a != "" {
						s.args += a
						s.buf += a
						s.n++
						if now := nowF(); now-s.flushed >= 0.05 {
							s.flushed = now
							chunk, nn := s.buf, s.n
							s.buf, s.n = "", 0
							emit(evMsg(name, assistantMsg(map[string]any{"tool_args": map[string]any{"index": idx, "name": s.name, "n": nn, "delta": chunk}})))
						}
					}
				}
				if emitIt {
					inst.Touch()
					emit(evMsg(name, msg))
				}
				return true
			})
			resp.Body.Close()
			if err != nil {
				emit(errorObj(fmt.Sprintf("%T: %v", err, err)))
				return
			}
			if restart {
				continue
			}
			break
		}
		if hoff != "" && openTh != "" && len(openTh) > thEmitted {
			evOut(map[string]any{"thinking": openTh[thEmitted:]})
		}
		if len(slots) > 0 {
			for _, idx := range order {
				s := slots[idx]
				if s.buf != "" {
					emit(evMsg(name, assistantMsg(map[string]any{"tool_args": map[string]any{"index": idx, "name": s.name, "n": s.n, "delta": s.buf}})))
					s.buf, s.n = "", 0
				}
			}
			emit(evMsg(name, assistantMsg(map[string]any{"tool_calls": finishToolCalls(slots, order)})))
			if finish == "stop" {
				finish = "tool_calls"
			}
		}
		total := nowF() - started
		load := total
		if firstTokenAt > 0 {
			load = firstTokenAt - started
		}
		promptEval := nIn
		if nReproc >= 0 {
			promptEval = nReproc
		}
		emit(finalEvent(name, finish, total, load, promptEval, nOut))
	}
	deliver(name, stream, w, events, 500)
}

// ------------------------------------------------------------ generate

type responseRecorder struct {
	header http.Header
	status int
	buf    bytes.Buffer
}

func (rr *responseRecorder) Header() http.Header         { return rr.header }
func (rr *responseRecorder) WriteHeader(code int)        { rr.status = code }
func (rr *responseRecorder) Write(b []byte) (int, error) { return rr.buf.Write(b) }

func apiGenerate(w http.ResponseWriter, r *http.Request) {
	body, err := readBody(r)
	if err != nil {
		writeJSON(w, 400, errorObj("invalid JSON"))
		return
	}
	name := str(body, "model")
	if ka := fmt.Sprint(body["keep_alive"]); (ka == "0" || ka == "0s") && str(body, "prompt") == "" {
		freed := mgr.Unload(name)
		if stopRemoteFor(name) {
			freed = true
		}
		reason := "not_loaded"
		if freed {
			reason = "unload"
		}
		writeJSON(w, 200, map[string]any{"model": name, "response": "", "done": true, "done_reason": reason})
		return
	}
	var msgs []any
	if s := str(body, "system"); s != "" {
		msgs = append(msgs, map[string]any{"role": "system", "content": s})
	}
	user := map[string]any{"role": "user", "content": str(body, "prompt")}
	if imgs := list(body, "images"); len(imgs) > 0 {
		user["images"] = imgs
	}
	msgs = append(msgs, user)
	sub := map[string]any{}
	for k, v := range body {
		sub[k] = v
	}
	sub["messages"] = msgs
	delete(sub, "prompt")
	stream := true
	if s, ok := body["stream"].(bool); ok {
		stream = s
	}
	if !stream {
		rec := &responseRecorder{header: http.Header{}, status: 200}
		chatBody(sub, rec, r)
		var res map[string]any
		if json.Unmarshal(rec.buf.Bytes(), &res) == nil {
			if m, ok := res["message"].(map[string]any); ok {
				res["response"] = str(m, "content")
				delete(res, "message")
			}
			writeJSON(w, rec.status, res)
			return
		}
		w.WriteHeader(rec.status)
		w.Write(rec.buf.Bytes())
		return
	}
	// streaming: translate message events into response/thinking fields
	pr, pw := io.Pipe()
	go func() {
		rec := &pipeWriter{header: http.Header{}, pw: pw}
		chatBody(sub, rec, r)
		pw.Close()
	}()
	n := newNDJSON(w)
	sc := bufio.NewScanner(pr)
	sc.Buffer(make([]byte, 0, 64*1024), 64*1024*1024)
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var ev map[string]any
		if json.Unmarshal(line, &ev) != nil {
			continue
		}
		m := sub2(ev, "message")
		ev["response"] = str(m, "content")
		if t := str(m, "thinking"); t != "" {
			ev["thinking"] = t
		}
		delete(ev, "message")
		n.send(ev)
	}
}

func sub2(m map[string]any, k string) map[string]any { return sub(m, k) }

type pipeWriter struct {
	header http.Header
	pw     *io.PipeWriter
}

func (p *pipeWriter) Header() http.Header         { return p.header }
func (p *pipeWriter) WriteHeader(code int)        {}
func (p *pipeWriter) Write(b []byte) (int, error) { return p.pw.Write(b) }
func (p *pipeWriter) Flush()                      {}

// --------------------------------------------------------------- embed

func apiEmbed(w http.ResponseWriter, r *http.Request) {
	body, _ := readBody(r)
	name := str(body, "model")
	var inputs []any
	switch t := body["input"].(type) {
	case string:
		inputs = []any{t}
	case []any:
		inputs = t
	default:
		inputs = []any{str(body, "prompt")}
	}
	inst, err := mgr.Get(name, defaultCtx, body["keep_alive"], turnHasMedia(body))
	if err != nil {
		loadError(w, name, err, false)
		return
	}
	resp, err := postJSONStream(r.Context(), inst.URL()+"/v1/embeddings", map[string]any{"model": name, "input": inputs})
	if err != nil {
		writeJSON(w, 502, errorObj(err.Error()))
		return
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	var d map[string]any
	json.Unmarshal(raw, &d)
	if resp.StatusCode != 200 {
		s := string(raw)
		if len(s) > 300 {
			s = s[:300]
		}
		writeJSON(w, resp.StatusCode, errorObj(fmt.Sprintf("embeddings failed (%d): %s", resp.StatusCode, s)))
		return
	}
	vecs := []any{}
	for _, row := range list(d, "data") {
		if rm, ok := row.(map[string]any); ok {
			vecs = append(vecs, rm["embedding"])
		}
	}
	var firstVec any = []any{}
	if len(vecs) > 0 {
		firstVec = vecs[0]
	}
	writeJSON(w, 200, map[string]any{"model": name, "embeddings": vecs, "embedding": firstVec})
}

// ------------------------------------------------------------ v1 proxy

func v1Proxy(path string, w http.ResponseWriter, r *http.Request) {
	body, err := readBody(r)
	if err != nil {
		writeJSON(w, 400, map[string]any{"error": map[string]any{"message": "invalid JSON", "type": "invalid_request_error"}})
		return
	}
	name := str(body, "model")
	if _, ok := body["chat_template_kwargs"]; !ok && thinkOff(name) {
		body["chat_template_kwargs"] = map[string]any{"enable_thinking": false}
	}
	inst, err := mgr.Get(name, v1Ctx, body["keep_alive"], turnHasMedia(body))
	if err != nil {
		loadError(w, name, err, true)
		return
	}
	realB, _ := json.Marshal(inst.Model.GGUF)
	wantB, _ := json.Marshal(name)
	if s, _ := body["stream"].(bool); s {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		f, _ := w.(http.Flusher)
		fail := func(msg string) {
			b, _ := json.Marshal(map[string]any{"error": map[string]any{"message": msg, "type": "server_error"}})
			w.Write([]byte("data: " + string(b) + "\n\n"))
			if f != nil {
				f.Flush()
			}
		}
		resp, err := postJSONStream(r.Context(), inst.URL()+path, body)
		if err != nil {
			fail(fmt.Sprintf("%T: %v", err, err))
			return
		}
		defer resp.Body.Close()
		if resp.StatusCode != 200 {
			detail, _ := io.ReadAll(io.LimitReader(resp.Body, 400))
			fail(fmt.Sprintf("llama-server %d: %s", resp.StatusCode, string(detail)))
			return
		}
		buf := make([]byte, 32*1024)
		for {
			n, err := resp.Body.Read(buf)
			if n > 0 {
				inst.Touch()
				chunk := buf[:n]
				if bytes.Contains(chunk, realB) {
					chunk = bytes.ReplaceAll(chunk, realB, wantB)
				}
				w.Write(chunk)
				if f != nil {
					f.Flush()
				}
			}
			if err != nil {
				break
			}
		}
		return
	}
	resp, err := postJSONStream(r.Context(), inst.URL()+path, body)
	if err != nil {
		writeJSON(w, 502, map[string]any{"error": map[string]any{"message": err.Error(), "type": "server_error"}})
		return
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	var d map[string]any
	if json.Unmarshal(raw, &d) == nil && d != nil {
		d["model"] = name
		writeJSON(w, resp.StatusCode, d)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	w.Write(raw)
}
