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

int main() {
    test_packed_evaluation_and_batch();
    test_training_and_state();
    test_persistence();
    std::cout << "All greedy BFL tests passed\n";
}
