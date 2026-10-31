#pragma once

// What `llmash ablate` draws while llama-ablate runs: a row per stage, the layer the refusal direction was read from
// on a strip of the model's layers, and each search pass's KL as a sparkline.

#include <cstddef>
#include <string>
#include <vector>

namespace llmash {

// The value right after "key=" on one of llama-ablate's progress lines (space-delimited).
std::string ablate_field(const std::string & s, const char * key);

struct AblateView {
    enum Stage { kLoad, kDirection, kBaseline, kSearch, kCheck, kDone };

    std::string         model;
    int                 stage      = kLoad;
    size_t              cap_done   = 0;
    size_t              cap_total  = 0;
    int                 layer      = -1;
    int                 n_layer    = 0;
    double              separation = 0.0;
    int                 passes     = 0;
    std::vector<double> kl;  // each search pass's, in order
    double              best_kl    = -1.0;
    double              seconds    = 0.0;  // the tool's own total, once done

    // one of the tool's lines, its "ablate: " prefix taken off
    void feed(const std::string & line);

    // the rows to draw, each at most `width` columns; `frame` turns the spinner, `color` adds the terminal's colours
    std::vector<std::string> rows(int width, int frame, double elapsed, bool color) const;
};

}  // namespace llmash
