// Copyright 2026 Facundo Gomez Prates
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "bfl/bfl.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace bfl {

struct BinaryCandidateScore {
    std::size_t err = 0, forget = 0, change_cost = 0;
};

std::uint64_t binary_error_mask(std::uint64_t target, std::uint64_t prediction,
                                std::uint64_t valid) noexcept;
std::uint64_t binary_forget_mask(std::uint64_t before, std::uint64_t after,
                                 std::uint64_t valid) noexcept;
bool binary_score_less(BinaryCandidateScore left, BinaryCandidateScore right) noexcept;

// Binary feature model from the routed CHANGE/USED experiment.
// This architecture is distinct from the layered MUX GreedyModel.
struct RoutedBinarySample {
    std::array<Bit, 10> input{};
    Bit target = 0;
};

struct RoutedBinaryTrainReport {
    std::size_t epochs = 0, candidate_evaluations = 0, popcount_operations = 0;
    std::size_t accepted_updates = 0, rejected_candidates = 0;
    std::size_t effective_changed_bits = 0, forget_count = 0;
    std::size_t err_improves = 0, err_ties_forget_improves = 0;
    std::size_t err_forget_ties_change_improves = 0;
    std::size_t err_before_sum = 0, err_after_sum = 0;
    std::size_t initial_errors = 0, final_errors = 0;
    std::size_t training_working_bytes = 0;
};

class RoutedBinaryModel {
public:
    static constexpr std::size_t input_bits = 10;
    static constexpr std::size_t feature_bits = 16;
    static constexpr std::size_t route_bits = 5;
    static constexpr std::size_t routes = 1u << route_bits;
    static constexpr std::size_t max_candidate_flips = 3;

    explicit RoutedBinaryModel(std::uint64_t initial_state, bool mask_with_used = true);
    Bit predict(const std::array<Bit, input_bits>& input) const noexcept;
    RoutedBinaryTrainReport train(const std::vector<RoutedBinarySample>& data,
                                  std::mt19937& rng, std::size_t epochs = 20);
    static std::uint64_t feature_mask(const std::array<Bit, input_bits>& input) noexcept;
    static std::size_t route_of(const std::array<Bit, input_bits>& input) noexcept;
    const std::array<std::uint64_t, routes>& states() const noexcept { return states_; }
    static constexpr std::size_t model_storage_bytes() noexcept {
        return routes * sizeof(std::uint64_t);
    }
private:
    std::array<std::uint64_t, routes> states_{};
    bool mask_with_used_ = true;
};

} // namespace bfl
