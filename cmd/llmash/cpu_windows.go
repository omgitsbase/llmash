package main

import (
	"syscall"
	"unsafe"
)

// The cores worth giving inference threads, and a mask that puts one thread on
// each of them.
//
// A hybrid Intel chip reports its performance cores in a higher efficiency
// class than its efficient ones, and llama.cpp's default counts every logical
// processor: on a 13900KF that is 24 threads against 8 performance cores.
// Measured there with llama 3B Q4_K_M, tokens a second: 19.4 at the default 24,
// 21.0 at 8, and 22.97 at 8 pinned one to a core. Two threads on one core share
// the units they are both waiting on, and an efficient core finishes late and
// holds up the rest.
//
// On a chip with one efficiency class this is the physical core count, which is
// what you want there too. Returns 0 and 0 when the topology cannot be read,
// which means leave llama.cpp's default alone rather than guess at it.
func perfCoresAndMask() (int, uint64) {
	const relationProcessorCore = 0

	kernel32 := syscall.NewLazyDLL("kernel32.dll")
	proc := kernel32.NewProc("GetLogicalProcessorInformationEx")
	if proc.Find() != nil {
		return 0, 0
	}

	var size uint32
	// the first call asks how much room the answer needs
	proc.Call(uintptr(relationProcessorCore), 0, uintptr(unsafe.Pointer(&size)))
	if size == 0 {
		return 0, 0
	}
	buf := make([]byte, size)
	r, _, _ := proc.Call(uintptr(relationProcessorCore), uintptr(unsafe.Pointer(&buf[0])), uintptr(unsafe.Pointer(&size)))
	if r == 0 {
		return 0, 0
	}

	// SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX is Relationship uint32, Size
	// uint32, then PROCESSOR_RELATIONSHIP: Flags byte, EfficiencyClass byte,
	// 20 reserved, GroupCount uint16, then GROUP_AFFINITY, which is a
	// 64-bit mask followed by a group number.
	type core struct {
		class int
		first uint64 // one bit: the first logical processor of this core
	}
	var cores []core

	for off := uint32(0); off+8 <= size; {
		rel := *(*uint32)(unsafe.Pointer(&buf[off]))
		sz := *(*uint32)(unsafe.Pointer(&buf[off+4]))
		if sz == 0 || off+sz > size {
			break
		}
		if rel == relationProcessorCore && off+48 <= size {
			groups := *(*uint16)(unsafe.Pointer(&buf[off+30]))
			group := *(*uint16)(unsafe.Pointer(&buf[off+40]))
			// a machine with more than one processor group is left alone:
			// an affinity mask only addresses the group it belongs to
			if groups == 1 && group == 0 {
				mask := *(*uint64)(unsafe.Pointer(&buf[off+32]))
				if mask != 0 {
					cores = append(cores, core{class: int(buf[off+9]), first: mask & (^mask + 1)})
				}
			}
		}
		off += sz
	}
	if len(cores) == 0 {
		return 0, 0
	}

	best := cores[0].class
	for _, c := range cores {
		if c.class > best {
			best = c.class
		}
	}
	n, mask := 0, uint64(0)
	for _, c := range cores {
		if c.class == best {
			n++
			mask |= c.first
		}
	}
	return n, mask
}
