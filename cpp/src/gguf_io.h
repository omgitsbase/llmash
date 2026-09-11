#pragma once

// Reading a GGUF's header and writing one back out, for the requantizer.

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace llmash {
namespace ggufio {

struct TensorEntry {
    std::string          name;
    std::vector<int64_t> dims;
    uint32_t             type   = 0;
    int64_t              offset = 0; // from the start of its part's data
    int64_t              bytes  = 0;
    size_t               part   = 0; // which shard it lives in

    int64_t rows() const;      // every dimension but the first
    int64_t experts() const;   // the third dimension, or 1
};

// A metadata entry, kept as the bytes it was read as so it can be written
// back unchanged.
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

    // Where a tensor's data sits in the file it came from.
    int64_t file_offset(const TensorEntry & t) const { return data_start[t.part] + t.offset; }
};

Layout read_layout(const std::string & path);
Layout layout_from(const std::string & head);

// A later shard's tensors appended to the first's. Its own metadata is
// split bookkeeping, so only the tensors are taken.
bool append_part(Layout & l, const std::string & head);

// A header for the whole model, written before the types are known: an
// entry's size does not depend on its type, so the types and offsets are
// filled in afterwards by patch_entry.
int64_t write_header(const std::string & path, const Layout & l, std::string & err);
bool    patch_entry(std::ostream & out, const Layout & l, size_t tensor, uint32_t type, int64_t offset);

} // namespace ggufio
} // namespace llmash
