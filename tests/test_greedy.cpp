// Copyright 2026 Facundo Gomez Prates
// SPDX-License-Identifier: Apache-2.0
#include "bfl/greedy.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

using namespace bfl;

template<class F> void throws(F action) {
    bool caught = false;
    try { action(); } catch (const std::exception&) { caught = true; }
    assert(caught);
}

void test_packed_evaluation_and_batch() {
    GreedyModel model({5, 1, 2, 16, 42});
    Dataset data;
    std::vector<std::vector<Bit>> inputs;
    for (int value = 0; value < 16; ++value) {
        std::vector<Bit> input(5);
        for (int bit = 0; bit < 4; ++bit) input[bit] = (value >> bit) & 1;
        input[4] = 1;
        inputs.push_back(input);
        data.push_back({input, {Bit(input[0] | input[1])}});
    }
    const auto predictions = model.predict_batch(inputs);
    const auto metrics = model.evaluate(data);
    std::size_t errors = 0;
    for (std::size_t i = 0; i < data.size(); ++i) {
        assert(predictions[i] == model.predict(data[i].input));
        std::vector<Bit> scratch(model.node_count());
        Bit output = 0;
        model.predict_into(data[i].input.data(), &output, scratch.data());
        assert(output == predictions[i][0]);
        errors += predictions[i][0] != data[i].target[0];
    }
    assert(metrics.error_bits == errors);
    assert(metrics.bit_accuracy == 1.0 - double(errors) / data.size());
    throws([&] { model.predict({0, 1}); });
    throws([&] { model.evaluate({{{0, 0, 0, 0, 2}, {0}}}); });
}

void test_training_and_state() {
    GreedyModel model({2, 1, 3, 32, 44});
    const Dataset xor_data = {
        {{0, 0}, {0}}, {{0, 1}, {1}}, {{1, 0}, {1}}, {{1, 1}, {0}}
    };
    const auto initial = model.evaluate(xor_data).error_bits;
    const auto report = model.train(xor_data, {8, 64, 3, 7, true});
    const auto final = model.evaluate(xor_data).error_bits;
    assert(initial == 3 && final == 1); // Reference algorithm fixture.
    assert(report.initial_error_bits == initial && report.final_error_bits == final);
    assert(report.accepted_flips > 0 && report.tested_flips >= report.accepted_flips);

    const auto state = model.export_state();
    GreedyModel copy({2, 1, 3, 32, 44});
    copy.import_state(state);
    assert(copy.predict_batch({{0,0},{0,1},{1,0},{1,1}}) ==
           model.predict_batch({{0,0},{0,1},{1,0},{1,1}}));
    copy.reset();
    for (Bit bit : copy.export_state()) assert(bit == 0);
    throws([&] { copy.import_state({0}); });
    auto invalid = state;
    invalid[0] = 2;
    throws([&] { copy.import_state(invalid); });
}

void test_masked_change_used_rule() {
    GreedyTrainConfig config;
    config.update_rule = GreedyTrainConfig::UpdateRule::MaskedChangeUsed;
    config.max_epochs = 1;
    config.shuffle_candidates = false;
    GreedyModel model({2, 1, 0, 0, 5});
    const std::vector<Bit> input{0, 1};
    const Bit before = model.predict(input)[0];
    const Dataset correctable = {{input, {Bit(before ^ 1)}}};
    const auto first = model.evaluate(correctable);
    assert(first.error_bits == 1 && first.used_bits == 1);
    const auto state = model.export_state();
    const auto report = model.train(correctable, config);
    const auto after = model.evaluate(correctable);
    assert(report.tested_flips == 1 && report.accepted_flips == 1);
    assert(report.initial_used_bits == 1 && report.final_used_bits == 1);
    assert(after.error_bits == 0 && after.used_bits == 1);
    assert(model.export_state()[0] == Bit(state[0] ^ 1));

    GreedyModel tied({2, 1, 0, 0, 5});
    const Dataset conflicting = {{input, {before}}, {input, {Bit(before ^ 1)}}};
    const auto tied_state = tied.export_state();
    const auto tied_report = tied.train(conflicting, config);
    assert(tied_report.tested_flips == 1 && tied_report.accepted_flips == 0);
    assert(tied.export_state() == tied_state);
    assert(tied.evaluate(conflicting).used_bits == 2);

    GreedyModel multi({3, 2, 2, 8, 42});
    Dataset data;
    for (int value = 0; value < 8; ++value) {
        const Bit a = value & 1, b = (value >> 1) & 1, c = (value >> 2) & 1;
        data.push_back({{a, b, c}, {Bit(a ^ b), Bit(b | c)}});
    }
    const auto initial = multi.evaluate(data);
    config.max_epochs = 5;
    const auto multi_report = multi.train(data, config);
    const auto final = multi.evaluate(data);
    assert(multi_report.initial_error_bits == initial.error_bits);
    assert(multi_report.final_error_bits == final.error_bits);
    assert(multi_report.initial_used_bits == initial.used_bits);
    assert(multi_report.final_used_bits == final.used_bits);
    assert(10 * final.error_bits + final.used_bits <=
           10 * initial.error_bits + initial.used_bits);

    GreedyModel inverse({3, 2, 2, 8, 42});
    config.penalize_unused = true;
    config.error_weight = 10;
    const auto inverse_initial = inverse.evaluate(data);
    const auto inverse_report = inverse.train(data, config);
    const auto inverse_final = inverse.evaluate(data);
    const auto available = data.size() * inverse.node_count();
    assert(inverse_report.initial_used_bits == inverse_initial.used_bits);
    assert(inverse_report.final_used_bits == inverse_final.used_bits);
    assert(10 * inverse_final.error_bits + available - inverse_final.used_bits <=
           10 * inverse_initial.error_bits + available - inverse_initial.used_bits);

    GreedyModel equal_weight({3, 2, 2, 8, 42});
    config.error_weight = 1;
    const auto equal_initial = equal_weight.evaluate(data);
    equal_weight.train(data, config);
    const auto equal_final = equal_weight.evaluate(data);
    assert(equal_final.error_bits + available - equal_final.used_bits <=
           equal_initial.error_bits + available - equal_initial.used_bits);

    GreedyModel recommended({3, 2, 2, 8, 42});
    GreedyModel explicit_weight({3, 2, 2, 8, 42});
    config.error_weight.reset();
    recommended.train(data, config);
    config.set_unused_lambda(1, GreedyTrainConfig::recommended_unused_error_weight);
    explicit_weight.train(data, config);
    assert(recommended.export_state() == explicit_weight.export_state());
    throws([&] { config.set_unused_lambda(1, 0); });
}

void test_persistence() {
    GreedyModel model({2, 1, 3, 32, 44});
    const Dataset data = {{{0,0},{0}}, {{0,1},{1}}, {{1,0},{1}}, {{1,1},{0}}};
    model.train(data, {8, 64, 3, 7, true});
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path() /
                      ("bfl_greedy_test_" + std::to_string(stamp) + ".bflg");
    model.save(path.string());
    const auto loaded = GreedyModel::load(path.string());
    assert(loaded.export_state() == model.export_state());
    assert(loaded.predict_batch({{0,0},{0,1},{1,0},{1,1}}) ==
           model.predict_batch({{0,0},{0,1},{1,0},{1,1}}));

    std::ifstream saved(path, std::ios::binary);
    const std::vector<unsigned char> bytes(std::istreambuf_iterator<char>{saved}, {});
    assert(bytes.size() == 60 + model.model_storage_bytes());
    const auto legacy_path = path.string() + ".bfl";
    const auto legacy_v1_path = path.string() + ".v1.bfl";
    auto write_u32 = [](std::ofstream& out, std::uint32_t value) {
        for (unsigned i = 0; i < 4; ++i) out.put(char((value >> (8 * i)) & 0xff));
    };
    auto write_u64 = [](std::ofstream& out, std::uint64_t value) {
        for (unsigned i = 0; i < 8; ++i) out.put(char((value >> (8 * i)) & 0xff));
    };
    for (int version : {1, 2}) {
        const auto& legacy_file = version == 1 ? legacy_v1_path : legacy_path;
        std::ofstream out(legacy_file, std::ios::binary);
        out.write("BFLM", 4);
        write_u32(out, version);
        write_u64(out, 2); write_u64(out, 1);
        write_u64(out, 3); write_u64(out, 32);
        write_u32(out, 44); write_u32(out, 0);
        if (version == 2) {
            out.write(reinterpret_cast<const char*>(bytes.data() + 60),
                      bytes.size() - 60);
        } else {
            for (std::size_t offset = 60; offset < bytes.size(); offset += 8) {
                std::uint64_t node = 0;
                for (unsigned i = 0; i < 8; ++i)
                    node |= std::uint64_t{bytes[offset + i]} << (8 * i);
                write_u32(out, node & ((1u << 21) - 1));
                write_u32(out, (node >> 21) & ((1u << 21) - 1));
                write_u32(out, (node >> 42) & ((1u << 21) - 1));
                out.put(char(node >> 63));
            }
        }
        out.close();
        const auto legacy = GreedyModel::load(legacy_file);
        assert(legacy.export_state() == model.export_state());
        assert(legacy.predict_batch({{0,0},{0,1},{1,0},{1,1}}) ==
               model.predict_batch({{0,0},{0,1},{1,0},{1,1}}));
        std::filesystem::remove(legacy_file);
    }
    std::filesystem::remove(path);
}

Dataset xor_dataset() {
    return {{{0, 0}, {0}}, {{0, 1}, {1}}, {{1, 0}, {1}}, {{1, 1}, {0}}};
}

void test_search_extensions_never_return_a_worse_state() {
    const auto data = xor_dataset();
    for (std::uint32_t seed = 1; seed <= 8; ++seed) {
        GreedyModel model({2, 1, 3, 32, seed});
        GreedyTrainConfig config;
        config.max_epochs = 30;
        config.patience = 0;
        config.seed = seed;
        config.initial_temperature = 2.0;
        config.pair_flips = 16;
        const auto report = model.train(data, config);
        assert(report.final_error_bits <= report.initial_error_bits);
        assert(model.evaluate(data).error_bits == report.final_error_bits);
    }
}

void test_defaults_match_the_original_greedy_rule() {
    const auto data = xor_dataset();
    GreedyModel a({2, 1, 3, 32, 44}), b({2, 1, 3, 32, 44});
    GreedyTrainConfig plain;
    plain.max_epochs = 8; plain.candidate_limit = 64; plain.patience = 3; plain.seed = 7;
    auto explicit_off = plain;
    explicit_off.pair_flips = 0;
    explicit_off.initial_temperature = 0;
    a.train(data, plain);
    b.train(data, explicit_off);
    assert(a.export_state() == b.export_state());
}

void test_pair_flips_reduce_error_when_single_flips_stall() {
    const auto data = xor_dataset();
    std::size_t improved = 0;
    for (std::uint32_t seed = 1; seed <= 20; ++seed) {
        GreedyModel single({2, 1, 3, 32, seed}), paired({2, 1, 3, 32, seed});
        GreedyTrainConfig config;
        config.max_epochs = 30;
        config.patience = 0;
        config.seed = seed;
        single.train(data, config);
        config.pair_flips = 64;
        paired.train(data, config);
        assert(paired.evaluate(data).error_bits <= single.evaluate(data).error_bits + 1);
        improved += paired.evaluate(data).error_bits < single.evaluate(data).error_bits;
    }
    assert(improved > 0);
}

int main() {
    test_packed_evaluation_and_batch();
    test_training_and_state();
    test_masked_change_used_rule();
    test_persistence();
    test_defaults_match_the_original_greedy_rule();
    test_search_extensions_never_return_a_worse_state();
    test_pair_flips_reduce_error_when_single_flips_stall();
    std::cout << "All greedy BFL tests passed\n";
}
