// Copyright 2026 Facundo Gomez Prates
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "bfl/bfl.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace bfl {

// Layered MUX model. Each MUX selects from the preceding layer with
// S_effective = S_dynamic XOR S_memory. Topology is fixed after construction.
struct GreedyConfig {
    std::size_t input_bits = 0;
    std::size_t output_bits = 0;
    std::size_t hidden_layers = 2;
    std::size_t hidden_width = 32;
    std::uint32_t seed = 42;
};

struct GreedyTrainConfig {
    std::size_t max_epochs = 100;
    std::size_t candidate_limit = 64; // Zero means no limit.
    std::size_t patience = 10; // Zero means no early stop for stagnation.
    std::uint32_t seed = 1234;
    bool shuffle_candidates = true;
};

struct GreedyMetrics {
    std::size_t error_bits = 0;
    double bit_accuracy = 0;
    double exact_accuracy = 0;
};

struct GreedyTrainReport {
    std::size_t epochs_ran = 0;
    std::size_t initial_error_bits = 0;
    std::size_t final_error_bits = 0;
    std::size_t tested_flips = 0;
    std::size_t accepted_flips = 0;
};

class GreedyModel {
public:
    explicit GreedyModel(GreedyConfig config);

    const GreedyConfig& config() const noexcept { return config_; }
    std::size_t node_count() const noexcept { return nodes_.size(); }
    std::size_t model_storage_bytes() const noexcept { return nodes_.size() * sizeof(std::uint64_t); }
    std::vector<Bit> predict(const std::vector<Bit>& input) const;
    std::vector<std::vector<Bit>> predict_batch(const std::vector<std::vector<Bit>>& inputs) const;
    GreedyMetrics evaluate(const Dataset& data) const;
    GreedyTrainReport train(const Dataset& data, const GreedyTrainConfig& config = {});

    void reset() noexcept;
    std::vector<Bit> export_state() const;
    void import_state(const std::vector<Bit>& state);
    void save(const std::string& path) const;
    static GreedyModel load(const std::string& path);

private:
    GreedyConfig config_;
    std::vector<std::size_t> layer_widths_;
    std::vector<std::size_t> layer_offsets_;
    std::vector<std::uint64_t> nodes_;
};

} // namespace bfl
