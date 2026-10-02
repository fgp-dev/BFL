// Copyright 2026 Facundo Gomez Prates
// SPDX-License-Identifier: Apache-2.0
// Reproducible held-out comparison of BFL's C++ learning variants and an MLP.
#include "bfl/bfl.hpp"
#include "bfl/greedy.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
constexpr std::size_t variable_bits = 8;
constexpr std::size_t input_bits = variable_bits + 1; // Both models receive a constant-one column.
constexpr std::size_t hidden = 16;
using Input = std::array<bfl::Bit, input_bits>;
struct Sample { Input input; bfl::Bit target; };
struct Split { std::vector<Sample> train, validation, test; };

bfl::Bit target_for(const std::string& task, const Input& x) {
    if (task == "or2") return x[0] | x[1];
    if (task == "xor2") return x[0] ^ x[1];
    if (task == "majority3") return (x[0] + x[1] + x[2]) >= 2;
    throw std::invalid_argument("Unknown task");
}

Split make_split(const std::string& task, std::uint64_t seed) {
    std::vector<std::size_t> domain(1u << variable_bits);
    std::iota(domain.begin(), domain.end(), 0);
    std::mt19937_64 rng(seed);
    std::shuffle(domain.begin(), domain.end(), rng);
    Split split;
    for (std::size_t i = 0; i < domain.size(); ++i) {
        Input x{};
        for (std::size_t bit = 0; bit < variable_bits; ++bit) x[bit] = (domain[i] >> bit) & 1u;
        x[variable_bits] = 1;
        Sample sample{x, target_for(task, x)};
        if (i < domain.size() / 2) split.train.push_back(sample);
        else if (i < 3 * domain.size() / 4) split.validation.push_back(sample);
        else split.test.push_back(sample);
    }
    return split;
}

struct Metrics { std::size_t errors = 0; double accuracy = 0; };

Metrics greedy_metrics(const bfl::GreedyModel& model, const bfl::Dataset& data) {
    const auto result = model.evaluate(data);
    return {result.error_bits, result.bit_accuracy};
}

struct MLP {
    static constexpr std::size_t b1 = hidden * input_bits;
    static constexpr std::size_t w2 = b1 + hidden;
    static constexpr std::size_t b2 = w2 + hidden;
    std::array<double, b2 + 1> weights{}, first{}, second{};

    explicit MLP(std::uint64_t seed) {
        std::mt19937_64 rng(seed);
        std::normal_distribution<double> init(0.0, std::sqrt(2.0 / (input_bits + hidden)));
        for (std::size_t i = 0; i < b1; ++i) weights[i] = init(rng);
        for (std::size_t i = w2; i < b2; ++i) weights[i] = init(rng);
    }

    double logit(const Input& x, std::array<double, hidden>& activation) const {
        double result = weights[b2];
        for (std::size_t h = 0; h < hidden; ++h) {
            double z = weights[b1 + h];
            for (std::size_t i = 0; i < input_bits; ++i) z += weights[h * input_bits + i] * x[i];
            activation[h] = std::tanh(z);
            result += weights[w2 + h] * activation[h];
        }
        return result;
    }

    bfl::Bit predict(const Input& x) const {
        std::array<double, hidden> activation{};
        return logit(x, activation) >= 0;
    }

    void fit(const std::vector<Sample>& data) {
        constexpr std::size_t epochs = 1500;
        constexpr double rate = 0.03, beta1 = 0.9, beta2 = 0.999;
        double beta1_power = 1, beta2_power = 1;
        for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
            std::array<double, b2 + 1> gradient{};
            for (const auto& sample : data) {
                std::array<double, hidden> activation{};
                const double z = logit(sample.input, activation);
                const double probability = 1.0 / (1.0 + std::exp(-z));
                const double delta = probability - sample.target;
                gradient[b2] += delta;
                for (std::size_t h = 0; h < hidden; ++h) {
                    gradient[w2 + h] += delta * activation[h];
                    const double hidden_delta = delta * weights[w2 + h] *
                                                (1.0 - activation[h] * activation[h]);
                    gradient[b1 + h] += hidden_delta;
                    for (std::size_t i = 0; i < input_bits; ++i)
                        gradient[h * input_bits + i] += hidden_delta * sample.input[i];
                }
            }
            beta1_power *= beta1;
            beta2_power *= beta2;
            for (std::size_t i = 0; i < weights.size(); ++i) {
                const double g = gradient[i] / data.size();
                first[i] = beta1 * first[i] + (1 - beta1) * g;
                second[i] = beta2 * second[i] + (1 - beta2) * g * g;
                const double m = first[i] / (1 - beta1_power);
                const double v = second[i] / (1 - beta2_power);
                weights[i] -= rate * m / (std::sqrt(v) + 1e-8);
            }
        }
    }
};

// Constant predictor of the training majority class: the floor any model must beat.
Metrics constant_metrics(bfl::Bit value, const std::vector<Sample>& samples) {
    std::size_t errors = 0;
    for (const auto& sample : samples) errors += sample.target != value;
    return {errors, 1.0 - double(errors) / samples.size()};
}

Metrics mlp_metrics(const MLP& model, const std::vector<Sample>& samples) {
    std::size_t errors = 0;
    for (const auto& sample : samples) errors += model.predict(sample.input) != sample.target;
    return {errors, 1.0 - double(errors) / samples.size()};
}

bfl::Graph make_direct_lookup() {
    bfl::Graph graph;
    std::array<bfl::NodeId, input_bits> inputs{};
    for (auto& input : inputs) input = graph.add_input();
    const auto zero = graph.add_constant(0), one = graph.add_constant(1);
    std::vector<bfl::NodeId> level;
    for (std::size_t i = 0; i < (1u << variable_bits); ++i)
        level.push_back(graph.add_adaptive_mux(zero, one));
    for (std::size_t bit = 0; bit < variable_bits; ++bit) {
        std::vector<bfl::NodeId> next;
        for (std::size_t i = 0; i < level.size(); i += 2)
            next.push_back(graph.add_signal_mux(level[i], level[i + 1], inputs[bit]));
        level = std::move(next);
    }
    graph.add_output(level[0]);
    return graph;
}

bfl::Dataset direct_data(const std::vector<Sample>& samples) {
    bfl::Dataset data;
    for (const auto& sample : samples)
        data.push_back({std::vector<bfl::Bit>(sample.input.begin(), sample.input.end()),
                        {sample.target}});
    return data;
}

Metrics direct_metrics(const bfl::Graph& model, const bfl::Dataset& data) {
    const auto errors = bfl::hamming_loss(model, data);
    return {errors, 1.0 - double(errors) / data.size()};
}

void print_row(const std::string& task, std::uint64_t seed, const char* model,
               const Split& split, Metrics train, Metrics validation, Metrics test,
               std::size_t candidates, std::size_t bytes, double milliseconds) {
    std::cout << task << ',' << seed << ',' << model << ',' << split.train.size() << ','
              << split.validation.size() << ',' << split.test.size() << ','
              << train.errors << ',' << train.accuracy << ','
              << validation.errors << ',' << validation.accuracy << ','
              << test.errors << ',' << test.accuracy << ',' << candidates << ','
              << bytes << ',' << milliseconds << '\n';
}

struct TrainedBFL { bfl::GreedyModel model; double milliseconds = 0; };

struct SearchVariant { const char* suffix; std::size_t pair_flips; double temperature; };
constexpr std::array<SearchVariant, 3> variants{{
    {"", 0, 0.0}, {"_pairs", 64, 0.0}, {"_anneal", 0, 1.5}}};

TrainedBFL fit_bfl(const bfl::Dataset& train_data, std::uint64_t seed,
                   std::size_t candidate, const SearchVariant& variant) {
    bfl::GreedyConfig model_config;
    model_config.input_bits = input_bits;
    model_config.output_bits = 1;
    model_config.hidden_layers = 3;
    model_config.hidden_width = 64;
    model_config.seed = static_cast<std::uint32_t>(seed * 101 + 42 + candidate * 10007);
    bfl::GreedyModel model(model_config);
    bfl::GreedyTrainConfig train_config;
    train_config.max_epochs = 100;
    train_config.candidate_limit = 128;
    train_config.patience = 15;
    train_config.seed = static_cast<std::uint32_t>(seed * 103 + 1234 + candidate * 10009);
    train_config.pair_flips = variant.pair_flips;
    train_config.initial_temperature = variant.temperature;
    const auto start = std::chrono::steady_clock::now();
    model.train(train_data, train_config);
    const double milliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return {std::move(model), milliseconds};
}
} // namespace

int main() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "task,seed,model,train_samples,validation_samples,test_samples,train_errors,train_accuracy,validation_errors,validation_accuracy,test_errors,test_accuracy,candidate_models,reported_model_bytes,training_ms\n";
    for (const std::string task : {"or2", "xor2", "majority3"}) {
        for (std::uint64_t seed = 1; seed <= 20; ++seed) {
            const auto split = make_split(task, seed);
            const auto train_data = direct_data(split.train);
            const auto validation_data = direct_data(split.validation);
            const auto test_data = direct_data(split.test);
            for (const auto& variant : variants) {
                std::vector<TrainedBFL> candidates;
                candidates.reserve(16);
                std::size_t chosen_index = 0;
                Metrics chosen_validation{};
                double search_ms = 0;
                constexpr std::size_t candidate_models = 16;
                for (std::size_t candidate = 0; candidate < candidate_models; ++candidate) {
                    auto trial = fit_bfl(train_data, seed, candidate, variant);
                    search_ms += trial.milliseconds;
                    const Metrics validation = greedy_metrics(trial.model, validation_data);
                    if (candidate == 0 || validation.errors < chosen_validation.errors ||
                        (validation.errors == chosen_validation.errors &&
                         greedy_metrics(trial.model, train_data).errors <
                         greedy_metrics(candidates[chosen_index].model, train_data).errors)) {
                        chosen_index = candidate;
                        chosen_validation = validation;
                    }
                    candidates.push_back(std::move(trial));
                }
                const auto& first = candidates.front();
                const auto& chosen = candidates[chosen_index];
                const std::string single = std::string("bfl_greedy_single") + variant.suffix;
                const std::string selected = std::string("bfl_greedy_selected") + variant.suffix;
                print_row(task, seed, single.c_str(), split,
                          greedy_metrics(first.model, train_data),
                          greedy_metrics(first.model, validation_data),
                          greedy_metrics(first.model, test_data), 1,
                          first.model.model_storage_bytes(), first.milliseconds);
                print_row(task, seed, selected.c_str(), split,
                          greedy_metrics(chosen.model, train_data), chosen_validation,
                          greedy_metrics(chosen.model, test_data), candidate_models,
                          chosen.model.model_storage_bytes(), search_ms);
            }
            {
                std::size_t ones = 0;
                for (const auto& sample : split.train) ones += sample.target;
                const bfl::Bit majority = ones * 2 >= split.train.size();
                print_row(task, seed, "majority_class", split,
                          constant_metrics(majority, split.train),
                          constant_metrics(majority, split.validation),
                          constant_metrics(majority, split.test), 1, 1, 0.0);
            }

            auto direct = make_direct_lookup();
            auto start = std::chrono::steady_clock::now();
            bfl::train(direct, train_data, {3});
            const double direct_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            print_row(task, seed, "direct_lookup_3ep", split,
                      direct_metrics(direct, train_data),
                      direct_metrics(direct, validation_data),
                      direct_metrics(direct, test_data), 1,
                      direct.memory_bytes(), direct_ms);

            MLP ann(seed * 107 + 7);
            start = std::chrono::steady_clock::now();
            ann.fit(split.train);
            const double ann_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            print_row(task, seed, "mlp_16", split, mlp_metrics(ann, split.train),
                      mlp_metrics(ann, split.validation), mlp_metrics(ann, split.test), 1,
                      ann.weights.size() * sizeof(double), ann_ms);
        }
    }
}
