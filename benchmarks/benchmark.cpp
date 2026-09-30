// Copyright 2026 Facundo Gomez Prates
// SPDX-License-Identifier: Apache-2.0

#include "bfl/bfl.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
using namespace bfl;

using Clock = chrono::steady_clock;

namespace {

volatile uint64_t inference_sink = 0;

vector<uint64_t> parse_list(const string& value) {
    vector<uint64_t> result;
    istringstream stream(value);
    string item;
    while (getline(stream, item, ',')) {
        if (item.empty()) throw invalid_argument("Empty list item");
        size_t end = 0;
        const auto number = stoull(item, &end);
        if (end != item.size()) throw invalid_argument("Invalid list item");
        result.push_back(number);
    }
    if (result.empty()) throw invalid_argument("List must not be empty");
    return result;
}

vector<Bit> input_bits(size_t value, size_t width) {
    vector<Bit> bits(width);
    for (size_t i = 0; i < width; ++i) bits[i] = (value >> i) & 1;
    return bits;
}

vector<Bit> target_bits(const vector<Bit>& input) {
    const auto ones = accumulate(input.begin(), input.end(), size_t{0});
    return {static_cast<Bit>(ones & 1),
            static_cast<Bit>(ones >= (input.size() + 1) / 2)};
}

struct Split {
    Dataset train;
    Dataset test;
};

Split make_split(size_t width, uint64_t seed) {
    // Shuffling the finite domain makes train/test inputs disjoint for every seed.
    const size_t domain = size_t{1} << width;
    vector<size_t> values(domain);
    iota(values.begin(), values.end(), 0);
    mt19937_64 rng(seed);
    shuffle(values.begin(), values.end(), rng);
    Split split;
    for (size_t i = 0; i < 3 * domain / 4; ++i) {
        auto input = input_bits(values[i], width);
        Example example{input, target_bits(input)};
        if (i < domain / 2) split.train.push_back(move(example));
        else split.test.push_back(move(example));
    }
    return split;
}

Graph make_lookup_graph(size_t width, size_t route_prefix_bits) {
    // Adaptive leaves store direct selector states; internal MUXes select by input bits.
    Graph graph(route_prefix_bits);
    vector<NodeId> inputs;
    for (size_t i = 0; i < width; ++i) inputs.push_back(graph.add_input());
    const auto zero = graph.add_constant(0);
    const auto one = graph.add_constant(1);
    for (size_t output = 0; output < 2; ++output) {
        vector<NodeId> level;
        for (size_t i = 0; i < (size_t{1} << width); ++i) {
            level.push_back(graph.add_adaptive_mux(zero, one));
        }
        for (size_t bit = 0; bit < width; ++bit) {
            vector<NodeId> next;
            for (size_t i = 0; i < level.size(); i += 2) {
                next.push_back(graph.add_signal_mux(level[i], level[i + 1], inputs[bit]));
            }
            level = move(next);
        }
        graph.add_output(level[0]);
    }
    return graph;
}

struct MajorityBaseline {
    vector<Bit> majority;

    void fit(const Dataset& data) {
        majority.assign(data[0].target.size(), 0);
        for (size_t j = 0; j < majority.size(); ++j) {
            size_t ones = 0;
            for (const auto& example : data) ones += example.target[j];
            majority[j] = 2 * ones >= data.size();
        }
    }
    const vector<Bit>& predict(const vector<Bit>&) const { return majority; }
    vector<vector<Bit>> predict_batch(const vector<vector<Bit>>& inputs) const {
        return vector<vector<Bit>>(inputs.size(), majority);
    }
    size_t memory_bytes() const { return sizeof(*this) + majority.capacity() * sizeof(Bit); }
};

struct Metrics {
    size_t loss = 0;
    size_t exact = 0;
    size_t bits = 0;
    size_t samples = 0;
};

template <class Model> Metrics measure(const Model& model, const Dataset& data) {
    Metrics result;
    result.samples = data.size();
    for (const auto& example : data) {
        const auto output = model.predict(example.input);
        size_t errors = 0;
        for (size_t j = 0; j < output.size(); ++j) errors += output[j] ^ example.target[j];
        result.loss += errors;
        result.exact += errors == 0;
        result.bits += output.size();
    }
    return result;
}

template <class Model> double inference_ns(const Model& model, const Dataset& data,
                                            size_t repeats) {
    const auto start = Clock::now();
    uint64_t checksum = 0;
    for (size_t repeat = 0; repeat < repeats; ++repeat) {
        for (const auto& example : data) {
            const auto output = model.predict(example.input);
            for (Bit bit : output) checksum += bit;
        }
    }
    inference_sink = checksum;
    const auto elapsed = chrono::duration<double, nano>(Clock::now() - start).count();
    return elapsed / (repeats * data.size());
}

template <class Model> double batch_inference_ns(const Model& model, const Dataset& data,
                                                  size_t repeats) {
    vector<vector<Bit>> inputs;
    inputs.reserve(data.size());
    for (const auto& example : data) inputs.push_back(example.input);
    const auto start = Clock::now();
    uint64_t checksum = 0;
    for (size_t repeat = 0; repeat < repeats; ++repeat) {
        const auto output = model.predict_batch(inputs);
        for (const auto& row : output) for (Bit bit : row) checksum += bit;
    }
    inference_sink = checksum;
    const auto elapsed = chrono::duration<double, nano>(Clock::now() - start).count();
    return elapsed / (repeats * data.size());
}

void print_row(size_t width, uint64_t seed, const char* model,
               size_t route_prefix_bits,
               const Dataset& train, const Dataset& test,
               const Metrics& train_metrics, const Metrics& test_metrics,
               double training_ms, double inference_per_sample_ns,
               double batch_inference_per_sample_ns,
               size_t comparisons, size_t state_updates, size_t distinct_routes,
               size_t distinct_adaptive_routes, size_t adaptive_addresses,
               size_t used_adaptive_addresses,
               double mean_examples_per_route, size_t nodes, size_t states, size_t bytes,
               size_t epochs, size_t repeats) {
    cout << width << ',' << seed << ',' << model << ',' << route_prefix_bits
              << ',' << train.size() << ',' << test.size()
              << ',' << nodes << ',' << states << ',' << bytes << ',' << epochs << ',' << repeats
              << ',' << training_ms << ',' << inference_per_sample_ns
              << ',' << batch_inference_per_sample_ns
              << ',' << train_metrics.loss
              << ',' << double(train_metrics.bits - train_metrics.loss) / train_metrics.bits
              << ',' << double(train_metrics.exact) / train_metrics.samples
              << ',' << test_metrics.loss
              << ',' << double(test_metrics.bits - test_metrics.loss) / test_metrics.bits
              << ',' << double(test_metrics.exact) / test_metrics.samples
              << ',' << comparisons << ',' << state_updates << ',' << distinct_routes
              << ',' << distinct_adaptive_routes << ',' << adaptive_addresses
              << ',' << used_adaptive_addresses
              << ',' << mean_examples_per_route << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        vector<uint64_t> sizes{4, 6, 8};
        vector<uint64_t> seeds{1, 2, 3};
        size_t epochs = 1;
        size_t repeats = 1000;
        size_t route_prefix_bits = 0;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 >= argc) throw invalid_argument("Missing argument value");
            const string flag = argv[i];
            const string value = argv[i + 1];
            if (flag == "--sizes") sizes = parse_list(value);
            else if (flag == "--seeds") seeds = parse_list(value);
            else if (flag == "--epochs") epochs = stoull(value);
            else if (flag == "--repeats") repeats = stoull(value);
            else if (flag == "--route-prefix-bits") route_prefix_bits = stoull(value);
            else throw invalid_argument("Unknown argument: " + flag);
        }
        if (repeats == 0) throw invalid_argument("Repeats must be positive");
        cout << fixed << setprecision(6);
        if (route_prefix_bits > 64) throw invalid_argument("Route prefix cannot exceed 64 bits");
        cout << "input_bits,seed,model,route_prefix_bits,train_samples,test_samples,nodes,selector_states,model_bytes,epochs,inference_repeats,training_ms,inference_ns_per_sample,batch_inference_ns_per_sample,train_loss,train_bit_accuracy,train_exact_accuracy,test_loss,test_bit_accuracy,test_exact_accuracy,comparisons,state_updates,distinct_routes,distinct_adaptive_routes,adaptive_addresses,used_adaptive_addresses,mean_examples_per_route\n";
        for (uint64_t width : sizes) {
            if (width < 2 || width > 12) throw invalid_argument("Input size must be in [2,12]");
            for (uint64_t seed : seeds) {
                const auto split = make_split(width, seed);
                auto graph = make_lookup_graph(width, route_prefix_bits);
                const auto bfl_start = Clock::now();
                const auto report = train(graph, split.train, {epochs});
                const double bfl_ms = chrono::duration<double, milli>(Clock::now() - bfl_start).count();
                const auto bfl_train = measure(graph, split.train);
                const auto bfl_test = measure(graph, split.test);
                print_row(width, seed, "bfl", route_prefix_bits, split.train, split.test, bfl_train, bfl_test,
                          bfl_ms, inference_ns(graph, split.test, repeats),
                          batch_inference_ns(graph, split.test, repeats),
                          report.comparisons, report.state_updates, report.distinct_routes,
                          report.distinct_adaptive_routes, report.adaptive_addresses,
                          report.used_adaptive_addresses,
                          report.mean_examples_per_route, graph.node_count(), graph.state_count(),
                          graph.memory_bytes(), epochs, repeats);

                MajorityBaseline baseline;
                const auto base_start = Clock::now();
                baseline.fit(split.train);
                const double base_ms = chrono::duration<double, milli>(Clock::now() - base_start).count();
                const auto base_train = measure(baseline, split.train);
                const auto base_test = measure(baseline, split.test);
                print_row(width, seed, "majority", route_prefix_bits, split.train, split.test, base_train, base_test,
                          base_ms, inference_ns(baseline, split.test, repeats),
                          batch_inference_ns(baseline, split.test, repeats),
                          0, 0, 0, 0, 0, 0, 0, 0, 2, baseline.memory_bytes(), 0, repeats);
            }
        }
    } catch (const exception& error) {
        cerr << "Benchmark error: " << error.what() << '\n';
        return 1;
    }
}
