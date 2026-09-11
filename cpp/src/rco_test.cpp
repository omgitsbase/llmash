// Whitebox test for rco.cpp. The quantizer itself lives in a DLL the runtime
// ships, so the parts that need it are skipped when it is not there; the
// solver is exercised either way.
#include "rco.cpp"

#include <cmath>
#include <cstdio>

using namespace llmash;
using namespace llmash::rco;

namespace {

int failures = 0;

void check(bool ok, const char * what) {
    if (!ok) {
        failures++;
        std::printf("FAIL %s\n", what);
    }
}

Measured with_costs(const char * name, int64_t elements, std::vector<Cost> costs) {
    Measured m;
    m.name     = name;
    m.elements = elements;
    m.costs    = std::move(costs);
    return m;
}

int64_t total_bytes(const std::vector<Measured> & ms, double lambda) {
    int64_t sum = 0;
    for (const Measured & m : ms) {
        const int t = pick(m, lambda);
        for (const Cost & c : m.costs) {
            if (c.type == t) {
                sum += c.bytes;
            }
        }
    }
    return sum;
}

} // namespace

int main() {
    // Two tensors, three types each: cheap ones cost more error.
    const std::vector<Measured> ms{
        with_costs("a", 1000, {{GT_Q6_K, 0.001, 800}, {GT_IQ4_XS, 0.010, 530}, {GT_IQ2_S, 0.100, 300}}),
        with_costs("b", 1000, {{GT_Q6_K, 0.002, 800}, {GT_IQ4_XS, 0.004, 530}, {GT_IQ2_S, 0.500, 300}}),
    };

    check(pick(ms[0], 0.0) == GT_Q6_K, "with bytes free, the best type wins");
    check(pick(ms[0], 1e9) == GT_IQ2_S, "with bytes dear, the smallest wins");

    const int64_t budget = 1100;
    const double  lambda = solve_lambda(ms, budget);
    const int64_t got    = total_bytes(ms, lambda);
    check(got <= budget, "the solved multiplier lands inside the budget");
    check(got > budget * 3 / 4, "and does not leave most of it unspent");

    // b's error climbs far faster as the type narrows, so it is the one
    // that keeps its bits.
    const int a = pick(ms[0], lambda), b = pick(ms[1], lambda);
    check(a != GT_Q6_K || b == GT_Q6_K, "the tensor that suffers most keeps the wider type");

    check(total_bytes(ms, solve_lambda(ms, 600)) <= 600, "a tight budget is still met");
    check(total_bytes(ms, solve_lambda(ms, 1600)) <= 1600, "a loose one spends what it may");

    Config      cfg;
    cfg.llama_bin = env_str("LLAMA_BIN");
    Ggml        g;
    std::string err;
    if (!cfg.llama_bin.empty() && g.load(cfg, err)) {
        check(g.block(GT_Q8_0) == 32, "ggml reports q8_0 in blocks of 32");
        check(g.type_size(GT_Q8_0) == 34, "and 34 bytes to the block");
        check(std::string(g.name(GT_IQ4_XS)) == "iq4_xs", "and names iq4_xs");
        check(g.quantized(GT_IQ4_XS) && !g.quantized(GT_F32), "and knows which types are quantized");

        // A ramp through every candidate: wider types must come back closer.
        const int64_t      n_per = 512, rows = 64;
        std::vector<float> src(static_cast<size_t>(n_per * rows));
        for (size_t i = 0; i < src.size(); i++) {
            src[i] = std::sin(static_cast<double>(i) * 0.01) * 0.2f;
        }
        std::vector<float> im(static_cast<size_t>(n_per), 1.0f);
        const std::vector<Cost> costs =
            measure(g, src.data(), rows, n_per, candidates(), im.data(), rows, 4);
        check(costs.size() == candidates().size(), "every candidate is measured");
        bool all_positive = true, ordered = true;
        for (size_t i = 0; i < costs.size(); i++) {
            all_positive = all_positive && costs[i].error > 0 && costs[i].bytes > 0;
            if (i > 0) {
                ordered = ordered && costs[i].bytes <= costs[i - 1].bytes;
            }
        }
        check(all_positive, "each has an error and a size");
        check(ordered, "and the candidates run from widest to narrowest");
        check(costs.front().error < costs.back().error, "the widest type reconstructs the closest");
    } else {
        std::printf("  (ggml not loaded: %s)\n", err.empty() ? "LLAMA_BIN unset" : err.c_str());
    }

    std::printf("%s (%d failure%s)\n", failures == 0 ? "all passed" : "FAILURES", failures,
                failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
