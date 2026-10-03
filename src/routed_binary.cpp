// Copyright 2026 Facundo Gomez Prates
// SPDX-License-Identifier: Apache-2.0
#include "bfl/routed_binary.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace bfl {
namespace {
constexpr std::uint64_t feature_limit = (std::uint64_t{1} << 16) - 1;

std::size_t count_bits(std::uint64_t value) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<std::size_t>(__builtin_popcountll(value));
#else
    std::size_t result = 0;
    while (value) { value &= value - 1; ++result; }
    return result;
#endif
}

Bit predict_state(std::uint64_t active, std::uint64_t state) noexcept {
    const auto positive = count_bits(active & state);
    const auto total = count_bits(active);
    return static_cast<Bit>(2 * positive >= total);
}

std::vector<std::uint64_t> make_candidates() {
    std::vector<std::uint64_t> candidates{0};
    for (std::size_t i = 0; i < RoutedBinaryModel::feature_bits; ++i)
        candidates.push_back(std::uint64_t{1} << i);
    for (std::size_t i = 0; i < RoutedBinaryModel::feature_bits; ++i)
        for (std::size_t j = i + 1; j < RoutedBinaryModel::feature_bits; ++j)
            candidates.push_back((std::uint64_t{1} << i) | (std::uint64_t{1} << j));
    for (std::size_t i = 0; i < RoutedBinaryModel::feature_bits; ++i)
        for (std::size_t j = i + 1; j < RoutedBinaryModel::feature_bits; ++j)
            for (std::size_t k = j + 1; k < RoutedBinaryModel::feature_bits; ++k)
                candidates.push_back((std::uint64_t{1} << i) |
                    (std::uint64_t{1} << j) | (std::uint64_t{1} << k));
    return candidates;
}

struct PreparedSample { std::uint64_t active; Bit target; };
} // namespace

std::uint64_t binary_error_mask(std::uint64_t target, std::uint64_t prediction,
                                std::uint64_t valid) noexcept {
    return (target ^ prediction) & valid;
}

std::uint64_t binary_forget_mask(std::uint64_t before, std::uint64_t after,
                                 std::uint64_t valid) noexcept {
    return (~before & after) & valid;
}

bool binary_score_less(BinaryCandidateScore left, BinaryCandidateScore right) noexcept {
    if (left.err != right.err) return left.err < right.err;
    if (left.forget != right.forget) return left.forget < right.forget;
    return left.change_cost < right.change_cost;
}

RoutedBinaryModel::RoutedBinaryModel(std::uint64_t initial_state, bool mask_with_used)
    : mask_with_used_(mask_with_used) {
    for (std::size_t route = 0; route < routes; ++route)
        states_[route] = (initial_state ^
            (0x9E3779B97F4A7C15ULL * (route + 1))) & feature_limit;
}

std::uint64_t RoutedBinaryModel::feature_mask(
    const std::array<Bit, input_bits>& input) noexcept {
    std::uint64_t result = 1;
    for (std::size_t i = 0; i < input_bits; ++i)
        result |= std::uint64_t{input[i] & 1u} << (i + 1);
    for (std::size_t i = 0; i < 5; ++i)
        result |= std::uint64_t{(input[2 * i] & input[2 * i + 1]) & 1u} << (11 + i);
    return result;
}

std::size_t RoutedBinaryModel::route_of(
    const std::array<Bit, input_bits>& input) noexcept {
    std::size_t result = 0;
    for (std::size_t i = 0; i < route_bits; ++i)
        result |= std::size_t{input[i] & 1u} << i;
    return result;
}

Bit RoutedBinaryModel::predict(const std::array<Bit, input_bits>& input) const noexcept {
    return predict_state(feature_mask(input), states_[route_of(input)]);
}

RoutedBinaryTrainReport RoutedBinaryModel::train(
    const std::vector<RoutedBinarySample>& data, std::mt19937& rng, std::size_t epochs) {
    if (data.empty()) throw std::invalid_argument("Training data must not be empty");
    std::array<std::vector<PreparedSample>, routes> memory;
    std::vector<std::size_t> order(data.size());
    std::iota(order.begin(), order.end(), 0);
    for (const auto& sample : data) {
        if (sample.target > 1) throw std::invalid_argument("Target must be binary");
        for (Bit bit : sample.input)
            if (bit > 1) throw std::invalid_argument("Input must be binary");
        memory[route_of(sample.input)].push_back({feature_mask(sample.input), sample.target});
    }
    const auto candidates = make_candidates();
    RoutedBinaryTrainReport report;
    report.training_working_bytes = order.capacity() * sizeof(std::size_t) +
        candidates.capacity() * sizeof(std::uint64_t);
    for (const auto& route : memory)
        report.training_working_bytes += route.capacity() * sizeof(PreparedSample);
    auto errors = [&]() {
        std::size_t result = 0;
        for (std::size_t route = 0; route < routes; ++route)
            for (const auto& sample : memory[route]) {
                result += sample.target ^ predict_state(sample.active, states_[route]);
                report.popcount_operations += 2;
            }
        return result;
    };
    report.initial_errors = errors();
    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), rng);
        for (const auto index : order) {
            const auto route = route_of(data[index].input);
            const auto used = feature_mask(data[index].input);
            const auto before_state = states_[route];
            const auto& examples = memory[route];
            std::vector<Bit> before(examples.size());
            std::size_t before_errors = 0;
            for (std::size_t i = 0; i < examples.size(); ++i) {
                before[i] = examples[i].target ^
                    predict_state(examples[i].active, before_state);
                report.popcount_operations += 2;
                before_errors += before[i];
            }
            report.training_working_bytes = std::max(report.training_working_bytes,
                order.capacity() * sizeof(std::size_t) +
                candidates.capacity() * sizeof(std::uint64_t) +
                before.capacity() * sizeof(Bit) + [&]() {
                    std::size_t bytes = 0;
                    for (const auto& group : memory)
                        bytes += group.capacity() * sizeof(PreparedSample);
                    return bytes;
                }());
            BinaryCandidateScore best{before_errors, 0, 0};
            std::uint64_t best_effective = 0;
            for (const auto change : candidates) {
                const auto effective = mask_with_used_ ? (change & used) : change;
                const auto after_state = before_state ^ effective;
                BinaryCandidateScore score{0, 0, count_bits(effective)};
                ++report.popcount_operations;
                for (std::size_t i = 0; i < examples.size(); ++i) {
                    const Bit after = examples[i].target ^
                        predict_state(examples[i].active, after_state);
                    report.popcount_operations += 2;
                    score.err += after;
                    score.forget += static_cast<Bit>((~before[i]) & after & 1u);
                }
                ++report.candidate_evaluations;
                report.err_before_sum += before_errors;
                report.err_after_sum += score.err;
                if (binary_score_less(score, best)) {
                    if (score.err < best.err) ++report.err_improves;
                    else if (score.forget < best.forget) ++report.err_ties_forget_improves;
                    else ++report.err_forget_ties_change_improves;
                    best = score;
                    best_effective = effective;
                }
            }
            report.rejected_candidates += candidates.size() - (best_effective != 0);
            if (best_effective) {
                // The official update is S' = S XOR (CHANGE AND USED).
                states_[route] ^= best_effective;
                ++report.accepted_updates;
                report.effective_changed_bits += best.change_cost;
                report.forget_count += best.forget;
            }
        }
        ++report.epochs;
    }
    report.final_errors = errors();
    return report;
}
} // namespace bfl
