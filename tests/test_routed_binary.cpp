// Copyright 2026 Facundo Gomez Prates
// SPDX-License-Identifier: Apache-2.0
#include "bfl/routed_binary.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>

int main() {
    using bfl::Bit;
    using bfl::RoutedBinaryModel;
    using bfl::RoutedBinarySample;
    assert(bfl::binary_error_mask(0b1010, 0b1100, 0b1111) == 0b0110);
    // Lanes: correct->wrong, correct->correct, wrong->correct, wrong->wrong.
    assert(bfl::binary_forget_mask(0b1100, 0b1001, 0b1111) == 0b0001);
    assert(bfl::update_state(1, 1, 0) == 1);
    assert(bfl::update_state(1, 1, 1) == 0);
    assert(bfl::binary_score_less({2, 5, 9}, {3, 0, 0}));
    assert(bfl::binary_score_less({2, 1, 9}, {2, 2, 0}));
    assert(bfl::binary_score_less({2, 1, 2}, {2, 1, 3}));
    assert(!bfl::binary_score_less({2, 1, 3}, {2, 1, 3}));
    const std::array<Bit, 10> input{{1, 1, 0, 1, 1, 1, 0, 0, 1, 0}};
    const auto used = RoutedBinaryModel::feature_mask(input);
    assert(used & 1u);
    assert(used & (std::uint64_t{1} << 11));
    assert(used & (std::uint64_t{1} << 13));
    assert(!(used & (std::uint64_t{1} << 12)));
    assert(RoutedBinaryModel::route_of(input) == 0b11011);
    RoutedBinaryModel model(12345);
    const auto state = model.states()[RoutedBinaryModel::route_of(input)];
    int score = 0;
    for (std::size_t i = 0; i < 16; ++i)
        if (used & (std::uint64_t{1} << i))
            score += (state & (std::uint64_t{1} << i)) ? 1 : -1;
    assert(model.predict(input) == static_cast<Bit>(score >= 0));

    std::vector<RoutedBinarySample> data{{input, Bit(model.predict(input) ^ 1)}};
    const auto old_states = model.states();
    std::mt19937 rng(7);
    const auto report = model.train(data, rng, 1);
    assert(report.candidate_evaluations == 697);
    assert(report.accepted_updates <= 1);
    for (std::size_t route = 0; route < RoutedBinaryModel::routes; ++route) {
        const auto changed = old_states[route] ^ model.states()[route];
        if (route == RoutedBinaryModel::route_of(input)) assert((changed & ~used) == 0);
        else assert(changed == 0);
    }
    assert(report.effective_changed_bits <= 3);
    std::cout << "All routed binary BFL tests passed\n";
}
