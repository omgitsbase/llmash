#include "config.h"
#include "registry.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace llmash;

namespace {

std::string human_size(uint64_t bytes) {
    const char * unit[] = {"B", "KB", "MB", "GB", "TB"};
    double v = static_cast<double>(bytes);
    int    u = 0;
    while (v >= 1024.0 && u < 4) {
        v /= 1024.0;
        u++;
    }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.1f %s", v, unit[u]);
    return buf;
}

int cmd_list(Registry & reg) {
    const auto models = reg.all();
    if (models.empty()) {
        std::printf("no models found\n");
        return 0;
    }
    std::printf("%-40s %-10s %10s  %s\n", "NAME", "QUANT", "SIZE", "");
    for (const auto & m : models) {
        std::printf("%-40s %-10s %10s  %s\n", m.name.c_str(), m.quant.c_str(),
                    human_size(m.size).c_str(), m.has_mtp ? "drafts for itself" : "");
    }
    return 0;
}

int cmd_models(const Config & cfg, Registry & reg) {
    std::printf("store        %s\n", cfg.models_root.empty() ? "(none)" : cfg.models_root.c_str());
    for (const auto & d : reg.library_dirs()) {
        std::printf("reading      %s\n", d.c_str());
    }
    std::printf("models       %zu\n", reg.all().size());
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    const Config cfg = load_config();
    Registry     reg(cfg);

    const std::string cmd = argc > 1 ? argv[1] : "help";

    if (cmd == "list" || cmd == "ls") {
        return cmd_list(reg);
    }
    if (cmd == "models") {
        return cmd_models(cfg, reg);
    }
    if (cmd == "-v" || cmd == "--version") {
        std::printf("llmash 0.4.0\n");
        return 0;
    }

    std::printf("usage: llmash <command>\n\n");
    std::printf("  list     the models that can be served\n");
    std::printf("  models   the folders being read\n");
    return cmd == "help" || cmd == "-h" || cmd == "--help" ? 0 : 1;
}
