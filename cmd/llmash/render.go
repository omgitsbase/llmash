package main

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"
)

// The `list` and `ps` tables. The server renders the unfiltered ones at
// /cli/list and /cli/ps and into cache/, and the command line prints them as
// they are; a filter argument makes the command render its own.

func short12(digest string) string {
	d := strings.TrimPrefix(digest, "sha256:")
	if len(d) > 12 {
		d = d[:12]
	}
	return d
}

func renderList(rows []map[string]any) string {
	t := newTable("NAME", "ID", "SIZE", "MODIFIED")
	for _, m := range rows {
		t.add(str(m, "name"), short12(str(m, "digest")),
			humanBytes(int64(num(m, "size"))), humanTimeISO(str(m, "modified_at"), "Never"))
	}
	return t.String()
}

func renderPs(rows []map[string]any) string {
	t := newTable("NAME", "ID", "SIZE", "PROCESSOR", "CONTEXT", "UNTIL")
	for _, m := range rows {
		size, vram := int64(num(m, "size")), int64(num(m, "size_vram"))
		var proc string
		switch {
		case vram == 0:
			proc = "100% CPU"
		case vram == size:
			proc = "100% GPU"
		case vram > size || size == 0:
			proc = "Unknown"
		default:
			cpu := math.Round(float64(size-vram) / float64(size) * 100)
			proc = fmt.Sprintf("%d%%/%d%% CPU/GPU", int(cpu), int(100-cpu))
		}
		until := "Never"
		if exp, err := time.Parse(time.RFC3339Nano, str(m, "expires_at")); err == nil {
			if time.Since(exp) > 0 {
				until = "Stopping..."
			} else {
				until = humanTime(exp, "Never")
			}
		}
		t.add(str(m, "name"), short12(str(m, "digest")), humanBytes(size), proc,
			strconv.Itoa(int(num(m, "context_length"))), until)
	}
	return t.String()
}
