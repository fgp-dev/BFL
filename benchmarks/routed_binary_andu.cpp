// Copyright 2026 Facundo Gomez Prates
// SPDX-License-Identifier: Apache-2.0
// Library-backed reproduction of the root AND U versus no-AND-U experiment.
#include "bfl/routed_binary.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace {
using Sample = bfl::RoutedBinarySample;

bfl::Bit target(const std::array<bfl::Bit, 10>& x) {
    const int score = -1 + x[0] + x[1] + (x[0] & x[1]) +
        (x[2] & x[3]) + (x[4] & x[5]) - x[6] - x[7] - (x[8] & x[9]);
    return static_cast<bfl::Bit>(score >= 1);
}

double accuracy(const bfl::RoutedBinaryModel& model, const std::vector<Sample>& data) {
    std::size_t correct = 0;
    for (const auto& sample : data) correct += model.predict(sample.input) == sample.target;
    return 100.0 * correct / data.size();
}
} // namespace

int main(int argc, char** argv) {
    const std::uint32_t seed = argc > 1 ? static_cast<std::uint32_t>(std::stoul(argv[1])) : 12345u;
    std::mt19937 rng(seed);
    std::vector<Sample> all;
    all.reserve(1024);
    for (unsigned value = 0; value < 1024; ++value) {
        Sample sample;
        for (std::size_t bit = 0; bit < 10; ++bit) sample.input[bit] = (value >> bit) & 1u;
        sample.target = target(sample.input);
        all.push_back(sample);
    }
    std::shuffle(all.begin(), all.end(), rng);
    const std::vector<Sample> train(all.begin(), all.begin() + 600);
    const std::vector<Sample> test(all.begin() + 600, all.end());
    std::uint64_t initial_state = 0;
    for (std::size_t bit = 0; bit < bfl::RoutedBinaryModel::feature_bits; ++bit)
        if (rng() & 1u) initial_state |= std::uint64_t{1} << bit;
    bfl::RoutedBinaryModel with_u(initial_state, true), without_u(initial_state, false);
    auto with_rng = rng, without_rng = rng;
    const auto with_report = with_u.train(train, with_rng, 20);
    const auto without_report = without_u.train(train, without_rng, 20);
    std::cout << "seed=" << seed << "\ntrain_n=600\ntest_n=424\nroutes=32\n\n";
    auto print = [&](const char* name, const bfl::RoutedBinaryModel& model,
                     const bfl::RoutedBinaryTrainReport& report, bool blank_line) {
        std::cout << name << "\ntrain=" << accuracy(model, train) <<
            "\ntest=" << accuracy(model, test) <<
            "\nupdates=" << report.accepted_updates <<
            "\neffective_flips=" << report.effective_changed_bits <<
            "\nforget_events=" << report.forget_count << '\n';
        if (blank_line) std::cout << '\n';
    };
    print("WITH_AND_U", with_u, with_report, true);
    print("WITHOUT_AND_U", without_u, without_report, false);
}
