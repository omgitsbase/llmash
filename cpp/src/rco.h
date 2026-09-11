#pragma once

// Choosing a quantization type per tensor under a total size budget.
//
// The budget is the only thing coupling the tensors, so in its Lagrangian
// form each one independently takes the type with the lowest
// error + lambda * bytes, and lambda is bisected until the total lands on
// the budget. The error is measured with ggml's own quantizer, loaded from
// the runtime beside llama-server.

#include "config.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace llmash {
namespace rco {

enum GgmlType : int {
    GT_F32 = 0, GT_F16 = 1, GT_Q8_0 = 8, GT_Q2_K = 10, GT_Q3_K = 11, GT_Q4_K = 12, GT_Q5_K = 13,
    GT_Q6_K = 14, GT_IQ2_XXS = 16, GT_IQ2_XS = 17, GT_IQ3_XXS = 18, GT_IQ4_NL = 20, GT_IQ3_S = 21,
    GT_IQ2_S = 22, GT_IQ4_XS = 23, GT_IQ1_M = 29, GT_BF16 = 30,
};

// ggml-base, found beside llama-server. Every entry point is checked against
// what it should answer for known types before anything uses it.
class Ggml {
public:
    ~Ggml();
    bool load(const Config & cfg, std::string & err);
    bool ok() const { return lib_ != nullptr; }

    int64_t      block(int type) const;
    size_t       type_size(int type) const;
    size_t       row_size(int type, int64_t n_per_row) const;
    const char * name(int type) const;
    bool         quantized(int type) const;

    size_t quantize(int type, const float * src, void * dst, int64_t nrows, int64_t n_per_row,
                    const float * imatrix) const;
    void   dequantize(int type, const void * src, float * dst, int64_t n) const;

private:
    void * lib_    = nullptr;
    void * traits_ = nullptr;
    void * quant_  = nullptr;
    void * row_    = nullptr;
    void * blck_   = nullptr;
    void * tsize_  = nullptr;
};

// The types a search may assign, widest first.
const std::vector<int> & candidates();

// Those of them a build at this width may use.
std::vector<int> candidates_for(const Ggml & g, double bpw);

// What a custom build aims for by default: what an IQ3_S build weighs.
// llama.cpp's names understate it, an "IQ3_XS" file measuring 3.85 bits a
// weight over the whole model, and a budget set to the name instead of the
// size hands the comparison a head start.
constexpr double DEFAULT_BPW = 3.9;

double bits_of_type(const Ggml & g, int type);

// What one tensor would cost at one type.
struct Cost {
    int     type  = 0;
    double  bits  = 0;
    double  error = 0; // imatrix-weighted output error, relative to the output
    int64_t bytes = 0;
};

// The least a tensor may be given whatever the budget says, because the
// search cannot see what it would cost. 0 for all but a couple.
double floor_bits(const std::string & tensor);

// Measures `sample_rows` rows spread through the tensor and scales their
// error to the whole, which makes what it costs to decide independent of
// how big the tensor is.
std::vector<Cost> measure(const Ggml & g, const float * data, int64_t nrows, int64_t n_per_row,
                          const std::vector<int> & types, const float * imatrix, int64_t sample_rows,
                          int nthread);

struct Measured {
    std::string       name;
    int64_t           elements   = 0;
    double            floor_bits = 0;
    std::vector<Cost> costs;
};

double solve_lambda(const std::vector<Measured> & m, int64_t budget_bytes);
int    pick(const Measured & m, double lambda);

// The type for every tensor: the bisection, then the budget's remainder
// handed to whichever tensors gain the most from it.
std::map<std::string, int> allocate(const std::vector<Measured> & m, int64_t budget_bytes);

// llama.cpp's imatrix file: per tensor, the mean square of each input
// channel over a calibration run, which is what tells the quantizer which
// columns to spend its bits on. Both the GGUF form and the older one.
class Imatrix {
public:
    bool   load(const std::string & path, std::string & err);
    bool   empty() const { return by_tensor_.empty(); }
    size_t size() const { return by_tensor_.size(); }

    // The weights for one tensor, or null. `expert` slices a 3-D tensor's
    // per-expert block the way llama-quantize does.
    const float * of(const std::string & tensor, int64_t n_per_row, int64_t expert = 0) const;

private:
    std::map<std::string, std::vector<float>> by_tensor_;
};

} // namespace rco
} // namespace llmash
