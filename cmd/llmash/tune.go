package main

import (
	"fmt"
	"strconv"
	"strings"
)

// Optimizations that apply to every model, decided at launch from what the
// model is and what the machine has free. Anything set by hand for a model in
// local.json's launch_extra wins: those flags are appended after these, and
// llama.cpp keeps the last occurrence.

var tuneOff = map[string]bool{}

func tuneEnabled(name string) bool {
	if len(tuneOff) == 0 {
		for _, k := range csv(envStr("LLMASH_TUNE_OFF", "")) {
			tuneOff[strings.ToLower(strings.TrimSpace(k))] = true
		}
		tuneOff["_loaded"] = true
	}
	return !tuneOff[name]
}

// autoTune returns the flags to add and a one-line account of why.
func (in *Instance) autoTune() ([]string, string) {
	var flags []string
	var why []string
	add := func(note string, xs ...string) {
		flags = append(flags, xs...)
		why = append(why, note)
	}

	// A conversation that comes back should not pay for its prompt twice.
	// cache-reuse lets a prefix that has shifted be reused by moving the KV
	// entries instead of recomputing them.
	if n := envInt("LLMASH_CACHE_REUSE", 256); n > 0 && tuneEnabled("cache-reuse") {
		add(fmt.Sprintf("cache-reuse %d", n), "--cache-reuse", strconv.Itoa(n))
	}

	// Prompt caches for slots that are not resident live in host RAM. The
	// default is 8 GB; give it a quarter of what is free, within reason.
	if tuneEnabled("cache-ram") {
		mib := envInt("LLMASH_CACHE_RAM_MB", 0)
		if mib == 0 {
			mib = int(freeRAMGB() * 1024 / 4)
			if mib < 8192 {
				mib = 8192
			}
			if mib > 32768 {
				mib = 32768
			}
		}
		if mib > 0 {
			add(fmt.Sprintf("cache-ram %d MiB", mib), "-cram", strconv.Itoa(mib))
		}
	}

	// Prompt processing runs in physical batches; the stock 512 leaves a big
	// card idle. Only widened when there is room left after the weights.
	if tuneEnabled("batch") {
		ub := envInt("LLMASH_UBATCH", 0)
		b := envInt("LLMASH_BATCH", 0)
		if ub == 0 && freeVRAMGB() > 24 {
			ub, b = 2048, 4096
		}
		if ub > 0 {
			if b < ub {
				b = ub * 2
			}
			add(fmt.Sprintf("batch %d/%d", b, ub), "-b", strconv.Itoa(b), "-ub", strconv.Itoa(ub))
		}
	}

	// A desktop has other things running; keep the server above them.
	if p := envInt("LLMASH_PRIO", 1); p > 0 && tuneEnabled("prio") {
		add(fmt.Sprintf("prio %d", p), "--prio", strconv.Itoa(p))
	}
	return flags, strings.Join(why, ", ")
}

// dropOverridden removes any flag the model's own launch_extra already sets,
// so hand configuration keeps the last word without duplicate arguments.
func dropOverridden(tuned, extra []string) []string {
	set := map[string]bool{}
	for _, x := range extra {
		if strings.HasPrefix(x, "-") {
			set[x] = true
		}
	}
	if len(set) == 0 {
		return tuned
	}
	var out []string
	for i := 0; i < len(tuned); i++ {
		if !strings.HasPrefix(tuned[i], "-") {
			out = append(out, tuned[i])
			continue
		}
		skip := set[tuned[i]]
		n := 1
		for n < len(tuned)-i && !strings.HasPrefix(tuned[i+n], "-") {
			n++
		}
		if !skip {
			out = append(out, tuned[i:i+n]...)
		}
		i += n - 1
	}
	return out
}

// specWhy names the drafter in the log the way the flag reads.
func specWhy(kind string) string {
	if kind == "" {
		return ""
	}
	if strings.HasPrefix(kind, "ngram") {
		return "self-speculation " + kind
	}
	return "speculation " + kind
}
