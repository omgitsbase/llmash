#pragma once

// Bringing a downloaded model to a GSQ-RCO allocation, a chunk at a time.
// IST-DASLab publishes the search result as `<tensor>: <TYPE>` lines, which
// is what llama-quantize's --tensor-type-file takes. The allocation belongs
// to the architecture, not the checkpoint.

#include "config.h"
#include "pull.h"

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace llmash {
namespace gsq {

struct Allocation {
    std::string name;
    std::string repo;
    std::string alloc;   // path in that repo to the .rco-allocation.txt
    std::string imatrix;
    double      bpw   = 0;
    std::string arch;
    int64_t     embd  = 0; // embedding_length, to tell same-arch models apart
    std::string fallback;  // what llama-quantize uses for anything unnamed
};

const std::vector<Allocation> & known();

// "GSQ-3.5". A quality name like any other, which is how it reaches the
// picker and comes back on a pull.
std::string quant_name(double bpw);

// The bit width a GSQ quality asks for, 0 when the name is not one.
double bpw_of_quant(const std::string & quant);

// The allocations published for this architecture, widest first.
std::vector<Allocation> all_for_arch(const std::string & arch, int64_t embd);
const Allocation *      for_model(const std::string & arch, int64_t embd, double bpw);

// What to requantize FROM: Q8_0 first, the baseline the allocation targets.
const std::vector<std::string> & source_preference();

// `<tensor>: <TYPE>` lines, lower-cased.
std::map<std::string, std::string> parse_allocation(const std::string & text);

// ------------------------------------------------------------------ GGUF

struct TensorEntry {
    std::string          name;
    std::vector<int64_t> dims;
    uint32_t             type   = 0;
    int64_t              offset = 0; // from the start of its part's data
    int64_t              bytes  = 0;
    size_t               part   = 0; // which shard it lives in
};

// One metadata entry, kept as the bytes it was read as so a chunk can carry
// it through unchanged.
struct KvEntry {
    std::string key;
    std::string raw;
};

struct Layout {
    uint32_t                 version = 0;
    std::vector<KvEntry>     kv;
    std::vector<TensorEntry> tensors;
    std::vector<int64_t>     data_start; // per part, where tensor data begins
    int64_t                  align = 32;
    std::string              error;
};

// The whole header: every metadata entry and every tensor with the bytes it
// occupies.
Layout read_layout(const std::string & path);
Layout layout_from(const std::string & head);

// A second shard's tensors, appended to the first's. Its own metadata is a
// stub (the split bookkeeping only), so only the tensors are taken.
bool append_part(Layout & l, const std::string & head);

// A chunk is a run of consecutive tensors: llama-quantize takes a file
// holding only some of a model's tensors, as long as the metadata is intact.
struct Chunk {
    size_t  first = 0, last = 0; // inclusive
    int64_t bytes = 0;
};
std::vector<Chunk> plan_chunks(const Layout & l, int64_t target_bytes);

// The chunk's GGUF header; returns its length, where the data begins.
int64_t write_chunk_header(const std::string & path, const Layout & l, const Chunk & c, std::string & err);

// Where tensor `i` of the chunk sits in the source, and how many bytes.
struct Piece {
    size_t  part = 0;
    int64_t from = 0, bytes = 0;
    int64_t into = 0; // offset in the chunk file
};
std::vector<Piece> chunk_pieces(const Layout & l, const Chunk & c, int64_t header_bytes);

// One file out of the quantized chunks, each removed as it is appended.
bool assemble(const std::vector<std::string> & chunks, const std::string & dest, const ProgressFn & progress,
              std::string & err);

// ------------------------------------------------------------ quantizing

// The --tensor-type-file lines, and how much of the model is covered. A
// tensor the allocation keeps at full precision counts as covered but must
// not reach the file; llama-quantize rejects it.
struct Plan {
    std::vector<std::string> lines;
    int                      covered = 0;
    int                      total   = 0;
    std::map<std::string, int> by_type;
};
Plan plan_types(const Layout & l, const std::map<std::string, std::string> & alloc);

// llama-quantize beside llama-server, or "".
std::string quantize_exe(const Config & cfg);

// Runs it over one chunk. `progress` is called with the tensor it is on and
// how many there are.
bool quantize(const std::string & exe, const std::string & src, const std::string & dst,
              const std::string & type_file, const std::string & imatrix, const std::string & fallback,
              const std::function<void(int, int)> & progress, std::string & err);

} // namespace gsq
} // namespace llmash
