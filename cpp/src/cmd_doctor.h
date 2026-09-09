#pragma once

// `llmash doctor`: a diagnostic report that fixes nothing.

namespace llmash {

// Loads its own Config/Registry, the way the Go command does, prints the
// report, and calls std::exit(1) if anything is a hard failure (matching
// doctor.go's own exit(1) on a FAIL row).
void cmd_doctor();

} // namespace llmash
