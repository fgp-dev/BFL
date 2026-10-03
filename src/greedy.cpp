// Copyright 2026 Facundo Gomez Prates
// SPDX-License-Identifier: Apache-2.0
#include "bfl/greedy.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace bfl {
namespace {
constexpr std::uint64_t source_mask = (std::uint64_t{1} << 21) - 1;
constexpr std::uint64_t state_mask = std::uint64_t{1} << 63;

std::uint32_t rng_next(std::uint32_t& state) {
    std::uint32_t x = state ? state : 0x9E3779B9u;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;
    return x;
}

std::uint32_t rng_range(std::uint32_t& state, std::uint32_t count) {
    return count ? rng_next(state) % count : 0;
}

std::uint64_t pack_node(std::uint32_t a, std::uint32_t b, std::uint32_t selector) {
    return std::uint64_t{a} | (std::uint64_t{b} << 21) |
           (std::uint64_t{selector} << 42);
}

std::uint32_t source_a(std::uint64_t node) { return node & source_mask; }
std::uint32_t source_b(std::uint64_t node) { return (node >> 21) & source_mask; }
std::uint32_t source_s(std::uint64_t node) { return (node >> 42) & source_mask; }

void require_bit(Bit bit) {
    if (bit > 1) throw std::invalid_argument("Expected a binary value");
}

void validate_data(const GreedyConfig& config, const Dataset& data) {
    if (data.empty()) throw std::invalid_argument("Dataset must not be empty");
    for (const auto& example : data) {
        if (example.input.size() != config.input_bits ||
            example.target.size() != config.output_bits)
            throw std::invalid_argument("Dataset shape does not match model");
        for (Bit bit : example.input) require_bit(bit);
        for (Bit bit : example.target) require_bit(bit);
    }
}

struct PackedData {
    std::size_t samples = 0;
    std::vector<std::uint64_t> inputs, targets, valid;
};

PackedData pack_data(const GreedyConfig& config, const Dataset& data) {
    validate_data(config, data);
    PackedData packed;
    packed.samples = data.size();
    const auto blocks = (data.size() + 63) / 64;
    packed.inputs.assign(blocks * config.input_bits, 0);
    packed.targets.assign(blocks * config.output_bits, 0);
    packed.valid.resize(blocks);
    for (std::size_t block = 0; block < blocks; ++block) {
        const auto count = std::min<std::size_t>(64, data.size() - block * 64);
        packed.valid[block] = count == 64 ? UINT64_MAX : ((std::uint64_t{1} << count) - 1);
        for (std::size_t lane = 0; lane < count; ++lane) {
            const auto& example = data[block * 64 + lane];
            const auto lane_bit = std::uint64_t{1} << lane;
            for (std::size_t i = 0; i < config.input_bits; ++i)
                if (example.input[i]) packed.inputs[block * config.input_bits + i] |= lane_bit;
            for (std::size_t i = 0; i < config.output_bits; ++i)
                if (example.target[i]) packed.targets[block * config.output_bits + i] |= lane_bit;
        }
    }
    return packed;
}

std::size_t popcount(std::uint64_t value) {
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<std::size_t>(__builtin_popcountll(value));
#else
    std::size_t count = 0;
    while (value) { value &= value - 1; ++count; }
    return count;
#endif
}

void forward_block(const std::vector<std::size_t>& widths,
                   const std::vector<std::size_t>& offsets,
                   const std::vector<std::uint64_t>& nodes,
                   const std::uint64_t* input, std::uint64_t valid,
                   std::vector<std::uint64_t>& values,
                   std::vector<std::uint64_t>& selectors) {
    for (std::size_t layer = 0; layer < widths.size(); ++layer) {
        const auto offset = offsets[layer];
        const auto previous = layer ? offsets[layer - 1] : 0;
        for (std::size_t local = 0; local < widths[layer]; ++local) {
            const auto id = offset + local;
            const auto node = nodes[id];
            const auto a = layer ? values[previous + source_a(node)] : input[source_a(node)];
            const auto b = layer ? values[previous + source_b(node)] : input[source_b(node)];
            const auto dynamic = layer ? values[previous + source_s(node)] : input[source_s(node)];
            const auto selector = (dynamic ^ ((node & state_mask) ? valid : 0)) & valid;
            selectors[id] = selector;
            values[id] = ((a & ~selector) | (b & selector)) & valid;
        }
    }
}

// USED contains the active MUXes reached from the requested output lanes.
// The selector source and the selected data source both affect a MUX's value.
void trace_used(const std::vector<std::size_t>& widths,
                const std::vector<std::size_t>& offsets,
                const std::vector<std::uint64_t>& nodes,
                const std::vector<std::uint64_t>& selectors,
                std::uint64_t valid, std::vector<std::uint64_t>& used) {
    for (std::size_t layer = widths.size(); layer-- > 1;) {
        const auto offset = offsets[layer];
        const auto previous = offsets[layer - 1];
        for (std::size_t local = 0; local < widths[layer]; ++local) {
            const auto id = offset + local;
            const auto active = used[id] & valid;
            if (!active) continue;
            const auto selected = selectors[id] & valid;
            const auto node = nodes[id];
            used[previous + source_s(node)] |= active;
            used[previous + source_a(node)] |= active & ~selected & valid;
            used[previous + source_b(node)] |= active & selected;
        }
    }
}

GreedyMetrics evaluate_packed(const GreedyConfig& config,
                              const std::vector<std::size_t>& widths,
                              const std::vector<std::size_t>& offsets,
                              const std::vector<std::uint64_t>& nodes,
                              const PackedData& data) {
    std::vector<std::uint64_t> values(nodes.size()), selectors(nodes.size());
    std::size_t errors = 0, exact = 0, used_bits = 0;
    std::vector<std::uint64_t> used(nodes.size());
    const auto output_offset = offsets.back();
    for (std::size_t block = 0; block < data.valid.size(); ++block) {
        const auto valid = data.valid[block];
        forward_block(widths, offsets, nodes,
                      data.inputs.data() + block * config.input_bits,
                      valid, values, selectors);
        std::fill(used.begin(), used.end(), 0);
        for (std::size_t bit = 0; bit < config.output_bits; ++bit)
            used[output_offset + bit] = valid;
        trace_used(widths, offsets, nodes, selectors, valid, used);
        for (const auto lanes : used) used_bits += popcount(lanes);
        std::uint64_t wrong = 0;
        for (std::size_t bit = 0; bit < config.output_bits; ++bit) {
            const auto error = (values[output_offset + bit] ^
                data.targets[block * config.output_bits + bit]) & valid;
            errors += popcount(error);
            wrong |= error;
        }
        exact += popcount(valid & ~wrong);
    }
    return {errors,
            1.0 - double(errors) / (data.samples * config.output_bits),
            double(exact) / data.samples, used_bits};
}

std::vector<std::vector<std::size_t>> collect_masked_candidates(
    const GreedyConfig& config, const std::vector<std::size_t>& widths,
    const std::vector<std::size_t>& offsets,
    const std::vector<std::uint64_t>& nodes, const PackedData& data) {
    std::vector<std::uint64_t> values(nodes.size()), selectors(nodes.size()), used(nodes.size());
    std::vector<std::vector<std::size_t>> candidates;
    const auto output_offset = offsets.back();
    for (std::size_t block = 0; block < data.valid.size(); ++block) {
        const auto valid = data.valid[block];
        forward_block(widths, offsets, nodes,
                      data.inputs.data() + block * config.input_bits,
                      valid, values, selectors);
        for (std::size_t bit = 0; bit < config.output_bits; ++bit) {
            auto errors = (values[output_offset + bit] ^
                data.targets[block * config.output_bits + bit]) & valid;
            while (errors) {
                const auto lane = errors & (~errors + 1);
                errors &= errors - 1;
                std::fill(used.begin(), used.end(), 0);
                used[output_offset + bit] = lane;
                trace_used(widths, offsets, nodes, selectors, lane, used);
                std::vector<std::size_t> flip;
                for (std::size_t id = 0; id < nodes.size(); ++id)
                    if (used[id]) flip.push_back(id);
                if (!flip.empty()) candidates.push_back(std::move(flip));
            }
        }
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    return candidates;
}

std::size_t objective(const GreedyMetrics& metrics, std::size_t error_weight,
                      std::size_t used_weight, bool penalize_unused,
                      std::size_t available_bits) {
    constexpr auto maximum = std::numeric_limits<std::size_t>::max();
    if (penalize_unused && metrics.used_bits > available_bits)
        throw std::logic_error("USED exceeds available selector bits");
    const auto usage = penalize_unused ? available_bits - metrics.used_bits : metrics.used_bits;
    if ((error_weight && metrics.error_bits > maximum / error_weight) ||
        (used_weight && usage >
            (maximum - error_weight * metrics.error_bits) / used_weight))
        throw std::overflow_error("Greedy objective is too large");
    return error_weight * metrics.error_bits + used_weight * usage;
}

std::vector<std::size_t> collect_candidates(const GreedyConfig& config,
                                            const std::vector<std::size_t>& widths,
                                            const std::vector<std::size_t>& offsets,
                                            const std::vector<std::uint64_t>& nodes,
                                            const PackedData& data) {
    std::vector<std::uint64_t> values(nodes.size()), selectors(nodes.size()), used(nodes.size());
    std::vector<Bit> mask(nodes.size());
    for (std::size_t block = 0; block < data.valid.size(); ++block) {
        std::fill(used.begin(), used.end(), 0);
        const auto valid = data.valid[block];
        forward_block(widths, offsets, nodes,
                      data.inputs.data() + block * config.input_bits,
                      valid, values, selectors);
        const auto output_offset = offsets.back();
        for (std::size_t bit = 0; bit < config.output_bits; ++bit)
            used[output_offset + bit] |= (values[output_offset + bit] ^
                data.targets[block * config.output_bits + bit]) & valid;

        for (std::size_t layer = widths.size(); layer-- > 1;) {
            const auto offset = offsets[layer];
            const auto previous = offsets[layer - 1];
            for (std::size_t local = 0; local < widths[layer]; ++local) {
                const auto id = offset + local;
                const auto active = used[id] & valid;
                if (!active) continue;
                const auto selected = selectors[id] & valid;
                const auto node = nodes[id];
                used[previous + source_s(node)] |= active;
                used[previous + source_a(node)] |= active & ~selected & valid;
                used[previous + source_b(node)] |= active & selected;
            }
        }
        for (std::size_t id = 0; id < nodes.size(); ++id)
            if (used[id]) mask[id] = 1;
    }
    std::vector<std::size_t> candidates;
    for (std::size_t id = 0; id < mask.size(); ++id)
        if (mask[id]) candidates.push_back(id);
    return candidates;
}

void write_u64(std::ostream& out, std::uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) out.put(static_cast<char>((value >> (8 * i)) & 0xff));
    if (!out) throw std::runtime_error("Could not write model");
}

std::uint64_t read_u64(std::istream& in) {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) {
        const int byte = in.get();
        if (byte == std::char_traits<char>::eof()) throw std::runtime_error("Truncated model");
        value |= std::uint64_t{static_cast<unsigned char>(byte)} << (8 * i);
    }
    return value;
}

std::uint32_t read_u32(std::istream& in) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i < 4; ++i) {
        const int byte = in.get();
        if (byte == std::char_traits<char>::eof()) throw std::runtime_error("Truncated model");
        value |= std::uint32_t{static_cast<unsigned char>(byte)} << (8 * i);
    }
    return value;
}

std::size_t read_size(std::istream& in) {
    const auto value = read_u64(in);
    if (value > std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("Model dimension is too large");
    return static_cast<std::size_t>(value);
}
} // namespace

GreedyModel::GreedyModel(GreedyConfig config) : config_(config) {
    if (!config.input_bits || !config.output_bits ||
        (config.hidden_layers && !config.hidden_width) ||
        config.input_bits > source_mask + 1 || config.hidden_width > source_mask + 1)
        throw std::invalid_argument("Invalid greedy model shape");
    if (config.hidden_layers == std::numeric_limits<std::size_t>::max())
        throw std::length_error("Too many layers");
    layer_widths_.assign(config.hidden_layers, config.hidden_width);
    layer_widths_.push_back(config.output_bits);
    layer_offsets_.reserve(layer_widths_.size());
    std::size_t total = 0;
    for (const auto width : layer_widths_) {
        if (width > nodes_.max_size() - total) throw std::length_error("Too many nodes");
        layer_offsets_.push_back(total);
        total += width;
    }
    nodes_.resize(total);
    std::uint32_t rng = config.seed ? config.seed : 1u;
    for (std::size_t layer = 0; layer < layer_widths_.size(); ++layer) {
        const auto count = static_cast<std::uint32_t>(
            layer ? layer_widths_[layer - 1] : config.input_bits);
        for (std::size_t local = 0; local < layer_widths_[layer]; ++local) {
            const auto a = rng_range(rng, count);
            std::uint32_t b = a, selector = a;
            if (count > 1) {
                do { b = rng_range(rng, count); } while (b == a);
                if (count == 2) selector = rng_range(rng, count);
                else do { selector = rng_range(rng, count); }
                     while (selector == a || selector == b);
            }
            nodes_[layer_offsets_[layer] + local] = pack_node(a, b, selector);
        }
    }
}

std::vector<Bit> GreedyModel::predict(const std::vector<Bit>& input) const {
    if (input.size() != config_.input_bits) throw std::invalid_argument("Input shape does not match model");
    for (Bit bit : input) require_bit(bit);
    std::vector<Bit> values(nodes_.size());
    std::vector<Bit> output(config_.output_bits);
    predict_into(input.data(), output.data(), values.data());
    return output;
}

void GreedyModel::predict_into(const Bit* input, Bit* output, Bit* values) const noexcept {
    for (std::size_t layer = 0; layer < layer_widths_.size(); ++layer) {
        const auto previous = layer ? layer_offsets_[layer - 1] : 0;
        for (std::size_t local = 0; local < layer_widths_[layer]; ++local) {
            const auto id = layer_offsets_[layer] + local;
            const auto node = nodes_[id];
            const Bit a = layer ? values[previous + source_a(node)] : input[source_a(node)];
            const Bit b = layer ? values[previous + source_b(node)] : input[source_b(node)];
            const Bit dynamic = layer ? values[previous + source_s(node)] : input[source_s(node)];
            values[id] = (dynamic ^ Bit((node & state_mask) != 0)) ? b : a;
        }
    }
    for (std::size_t bit = 0; bit < config_.output_bits; ++bit)
        output[bit] = values[layer_offsets_.back() + bit];
}

std::vector<std::vector<Bit>> GreedyModel::predict_batch(
    const std::vector<std::vector<Bit>>& inputs) const {
    std::vector<std::vector<Bit>> outputs;
    outputs.reserve(inputs.size());
    for (const auto& input : inputs) outputs.push_back(predict(input));
    return outputs;
}

GreedyMetrics GreedyModel::evaluate(const Dataset& data) const {
    return evaluate_packed(config_, layer_widths_, layer_offsets_, nodes_, pack_data(config_, data));
}

GreedyTrainReport GreedyModel::train(const Dataset& data, const GreedyTrainConfig& config) {
    const auto packed = pack_data(config_, data);
    auto current = evaluate_packed(config_, layer_widths_, layer_offsets_, nodes_, packed);
    GreedyTrainReport report;
    report.initial_error_bits = current.error_bits;
    report.initial_used_bits = current.used_bits;
    const bool masked = config.update_rule == GreedyTrainConfig::UpdateRule::MaskedChangeUsed;
    const auto error_weight = config.error_weight.value_or(config.penalize_unused ?
        GreedyTrainConfig::recommended_unused_error_weight : std::size_t{10});
    std::size_t available_bits = 0;
    if (masked && config.penalize_unused) {
        if (packed.samples > std::numeric_limits<std::size_t>::max() / nodes_.size())
            throw std::overflow_error("Dataset selector count is too large");
        available_bits = packed.samples * nodes_.size();
    }
    auto current_score = masked ? objective(current, error_weight, config.used_weight,
        config.penalize_unused, available_bits) : current.error_bits;
    if (!masked) {
        const bool anneal = config.initial_temperature > 0;
        double temperature = config.initial_temperature;
        std::size_t best_error = current.error_bits;
        std::vector<std::uint64_t> best_nodes;
        if (anneal) best_nodes = nodes_;
        std::size_t stagnant = 0;
        std::uint32_t rng = config.seed ? config.seed : 1u;
        for (std::size_t epoch = 1; epoch <= config.max_epochs && best_error; ++epoch) {
            report.epochs_ran = epoch;
            auto pool = collect_candidates(config_, layer_widths_, layer_offsets_, nodes_, packed);
            if (config.shuffle_candidates && pool.size() > 1) {
                for (std::size_t i = pool.size() - 1; i > 0; --i) {
                    const auto j = rng_range(rng, static_cast<std::uint32_t>(i + 1));
                    std::swap(pool[i], pool[j]);
                }
            }
            auto candidates = pool;
            if (config.candidate_limit && candidates.size() > config.candidate_limit)
                candidates.resize(config.candidate_limit);
            report.training_working_bytes = std::max(report.training_working_bytes,
                (packed.inputs.capacity() + packed.targets.capacity() + packed.valid.capacity() +
                 3 * nodes_.size()) * sizeof(std::uint64_t) +
                (pool.capacity() + candidates.capacity()) * sizeof(std::size_t) +
                best_nodes.capacity() * sizeof(std::uint64_t));
            bool improved = false;
            const auto note = [&] {
                if (current.error_bits < best_error) {
                    best_error = current.error_bits;
                    if (anneal) best_nodes = nodes_;
                    improved = true;
                }
            };
            for (const auto id : candidates) {
                ++report.tested_flips;
                nodes_[id] ^= state_mask;
                const auto trial = evaluate_packed(config_, layer_widths_, layer_offsets_, nodes_, packed);
                bool keep = trial.error_bits < current.error_bits;
                if (!keep && anneal) {
                    const double increase = double(trial.error_bits) - double(current.error_bits);
                    const double draw = rng_next(rng) / 4294967296.0;
                    keep = increase <= 0 || draw < std::exp(-increase / temperature);
                }
                if (keep) {
                    current = trial;
                    ++report.accepted_flips;
                    note();
                } else {
                    nodes_[id] ^= state_mask;
                }
                if (!best_error) break;
            }
            if (!improved && best_error && config.pair_flips && pool.size() > 1) {
                for (std::size_t k = 0; k < config.pair_flips; ++k) {
                    const auto i = rng_range(rng, static_cast<std::uint32_t>(pool.size()));
                    auto j = rng_range(rng, static_cast<std::uint32_t>(pool.size() - 1));
                    if (j >= i) ++j;
                    ++report.tested_flips;
                    nodes_[pool[i]] ^= state_mask;
                    nodes_[pool[j]] ^= state_mask;
                    const auto trial = evaluate_packed(config_, layer_widths_, layer_offsets_, nodes_, packed);
                    if (trial.error_bits < current.error_bits) {
                        current = trial;
                        ++report.accepted_flips;
                        note();
                        if (!best_error) break;
                    } else {
                        nodes_[pool[i]] ^= state_mask;
                        nodes_[pool[j]] ^= state_mask;
                    }
                }
            }
            if (anneal) temperature *= config.cooling;
            stagnant = improved ? 0 : stagnant + 1;
            if (config.patience && stagnant >= config.patience) break;
        }
        if (anneal && best_error < current.error_bits) {
            nodes_ = best_nodes;
            current = evaluate_packed(config_, layer_widths_, layer_offsets_, nodes_, packed);
        }
        report.final_error_bits = current.error_bits;
        report.final_used_bits = current.used_bits;
        return report;
    }
    std::size_t stagnant = 0;
    std::uint32_t rng = config.seed ? config.seed : 1u;
    auto shuffle = [&](std::vector<std::vector<std::size_t>>& candidates) {
        if (config.shuffle_candidates && candidates.size() > 1) {
            for (std::size_t i = candidates.size() - 1; i > 0; --i) {
                const auto j = rng_range(rng, static_cast<std::uint32_t>(i + 1));
                std::swap(candidates[i], candidates[j]);
            }
        }
    };
    for (std::size_t epoch = 1; epoch <= config.max_epochs && current.error_bits; ++epoch) {
        report.epochs_ran = epoch;
        std::vector<std::vector<std::size_t>> candidates;
        if (masked) {
            candidates = collect_masked_candidates(
                config_, layer_widths_, layer_offsets_, nodes_, packed);
        } else {
            for (const auto id : collect_candidates(
                     config_, layer_widths_, layer_offsets_, nodes_, packed))
                candidates.push_back({id});
        }
        shuffle(candidates);
        if (!masked && config.candidate_limit && candidates.size() > config.candidate_limit)
            candidates.resize(config.candidate_limit);
        std::size_t candidate_bytes = candidates.capacity() * sizeof(std::vector<std::size_t>);
        for (const auto& candidate : candidates)
            candidate_bytes += candidate.capacity() * sizeof(std::size_t);
        report.training_working_bytes = std::max(report.training_working_bytes,
            (packed.inputs.capacity() + packed.targets.capacity() + packed.valid.capacity() +
             3 * nodes_.size()) * sizeof(std::uint64_t) + candidate_bytes);
        std::size_t accepted = 0;
        std::size_t index = 0, tested = 0;
        while (index < candidates.size() &&
               (!masked || !config.candidate_limit || tested < config.candidate_limit)) {
            const auto& flip = candidates[index++];
            ++report.tested_flips;
            ++tested;
            for (const auto id : flip) nodes_[id] ^= state_mask;
            const auto trial = evaluate_packed(config_, layer_widths_, layer_offsets_, nodes_, packed);
            const auto trial_score = masked ? objective(trial, error_weight, config.used_weight,
                config.penalize_unused, available_bits) : trial.error_bits;
            const bool keep = trial_score < current_score;
            if (keep) {
                current = trial;
                current_score = trial_score;
                ++accepted;
                ++report.accepted_flips;
            } else {
                for (const auto id : flip) nodes_[id] ^= state_mask;
            }
            if (!current.error_bits) break;
            if (keep && masked) {
                candidates = collect_masked_candidates(
                    config_, layer_widths_, layer_offsets_, nodes_, packed);
                shuffle(candidates);
                index = 0;
            }
        }
        stagnant = accepted ? 0 : stagnant + 1;
        if (config.patience && stagnant >= config.patience) break;
    }
    report.final_error_bits = current.error_bits;
    report.final_used_bits = current.used_bits;
    return report;
}

void GreedyModel::reset() noexcept {
    for (auto& node : nodes_) node &= ~state_mask;
}

std::vector<Bit> GreedyModel::export_state() const {
    std::vector<Bit> state;
    state.reserve(nodes_.size());
    for (const auto node : nodes_) state.push_back((node & state_mask) != 0);
    return state;
}

void GreedyModel::import_state(const std::vector<Bit>& state) {
    if (state.size() != nodes_.size()) throw std::invalid_argument("State shape does not match model");
    for (Bit bit : state) require_bit(bit);
    for (std::size_t i = 0; i < state.size(); ++i)
        nodes_[i] = (nodes_[i] & ~state_mask) | (state[i] ? state_mask : 0);
}

void GreedyModel::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Could not open model file for writing");
    out.write("BFLG", 4);
    write_u64(out, 1);
    write_u64(out, config_.input_bits);
    write_u64(out, config_.output_bits);
    write_u64(out, config_.hidden_layers);
    write_u64(out, config_.hidden_width);
    write_u64(out, config_.seed);
    write_u64(out, nodes_.size());
    for (auto node : nodes_) write_u64(out, node);
}

GreedyModel GreedyModel::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Could not open model file for reading");
    char magic[4];
    if (!in.read(magic, 4))
        throw std::runtime_error("Invalid greedy model file");
    const bool legacy = std::string(magic, 4) == "BFLM";
    if (!legacy && std::string(magic, 4) != "BFLG")
        throw std::runtime_error("Invalid greedy model file");
    const auto version = legacy ? std::uint64_t{read_u32(in)} : read_u64(in);
    if (legacy ? (version != 1 && version != 2) : version != 1)
        throw std::runtime_error("Unsupported greedy model version");
    GreedyConfig config;
    config.input_bits = read_size(in);
    config.output_bits = read_size(in);
    config.hidden_layers = read_size(in);
    config.hidden_width = read_size(in);
    const auto seed = legacy ? std::uint64_t{read_u32(in)} : read_u64(in);
    if (seed > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("Invalid greedy model seed");
    config.seed = static_cast<std::uint32_t>(seed);
    if (legacy) read_u32(in); // Reserved field in the earlier C file header.
    const auto count = legacy ? 0 : read_u64(in);
    GreedyModel model(config);
    if (!legacy && count != model.nodes_.size())
        throw std::runtime_error("Invalid greedy model node count");
    for (std::size_t layer = 0; layer < model.layer_widths_.size(); ++layer) {
        const auto preceding = layer ? model.layer_widths_[layer - 1] : config.input_bits;
        for (std::size_t local = 0; local < model.layer_widths_[layer]; ++local) {
            std::uint64_t node;
            if (legacy && version == 1) {
                const auto a = read_u32(in), b = read_u32(in), selector = read_u32(in);
                const int state = in.get();
                if (state == std::char_traits<char>::eof() ||
                    a > source_mask || b > source_mask || selector > source_mask)
                    throw std::runtime_error("Invalid greedy model node");
                node = pack_node(a, b, selector) | (state ? state_mask : 0);
            } else {
                node = read_u64(in);
            }
            if (source_a(node) >= preceding || source_b(node) >= preceding ||
                source_s(node) >= preceding)
                throw std::runtime_error("Invalid greedy model source index");
            model.nodes_[model.layer_offsets_[layer] + local] = node;
        }
    }
    if (in.get() != std::char_traits<char>::eof())
        throw std::runtime_error("Trailing data in greedy model file");
    return model;
}

} // namespace bfl
