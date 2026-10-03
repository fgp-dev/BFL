// Copyright 2026 Facundo Gomez Prates
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "bfl/bfl.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
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
    enum class UpdateRule { SingleBit, MaskedChangeUsed };
    static constexpr std::size_t recommended_unused_error_weight = 17;
    std::size_t max_epochs = 100;
    std::size_t candidate_limit = 64; // Zero means no limit.
    std::size_t patience = 10; // Zero means no early stop for stagnation.
    std::uint32_t seed = 1234;
    bool shuffle_candidates = true;
    // Optional search extensions. Defaults reproduce the original greedy rule.
    // When an epoch accepts no single flip, try up to this many random pairs of
    // candidate bits and keep a pair only if the training error falls.
    std::size_t pair_flips = 0;
    // Simulated annealing: a flip that does not lower the error is kept with
    // probability exp(-increase / temperature). Zero disables it. The best
    // state seen is restored when training ends.
    double initial_temperature = 0;
    double cooling = 0.95; // Temperature multiplier applied after each epoch.
    UpdateRule update_rule = UpdateRule::SingleBit;
    // Masked rule minimizes error_weight * error_bits + used_weight * usage term.
    // Unset uses 10 for USED and the validation-selected 17 for NOT(USED).
    std::optional<std::size_t> error_weight = std::nullopt;
    std::size_t used_weight = 1;
    // When true, replace used_bits with the count of inactive MUXes per example.
    bool penalize_unused = false;

    // Minimize error_bits + (numerator / denominator) * unused_bits,
    // using the equivalent all-integer score denominator * errors + numerator * unused.
    void set_unused_lambda(std::size_t numerator, std::size_t denominator) {
        if (!denominator) throw std::invalid_argument("Lambda denominator must be positive");
        update_rule = UpdateRule::MaskedChangeUsed;
        penalize_unused = true;
        error_weight = denominator;
        used_weight = numerator;
    }
};

struct GreedyMetrics {
    std::size_t error_bits = 0;
    double bit_accuracy = 0;
    double exact_accuracy = 0;
    std::size_t used_bits = 0; // Sum of distinct active MUXes per example.
};

struct GreedyTrainReport {
    std::size_t epochs_ran = 0;
    std::size_t initial_error_bits = 0;
    std::size_t final_error_bits = 0;
    std::size_t initial_used_bits = 0;
    std::size_t final_used_bits = 0;
    std::size_t tested_flips = 0;
    std::size_t accepted_flips = 0;
    std::size_t training_working_bytes = 0;
};

class GreedyModel {
public:
    explicit GreedyModel(GreedyConfig config);

    const GreedyConfig& config() const noexcept { return config_; }
    std::size_t node_count() const noexcept { return nodes_.size(); }
    std::size_t model_storage_bytes() const noexcept { return nodes_.size() * sizeof(std::uint64_t); }
    std::vector<Bit> predict(const std::vector<Bit>& input) const;
    // Caller supplies input_bits, output_bits, and node_count() bytes respectively.
    void predict_into(const Bit* input, Bit* output, Bit* scratch) const noexcept;
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
