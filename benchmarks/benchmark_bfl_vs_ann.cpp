// Copyright 2026 Facundo Gomez Prates
// SPDX-License-Identifier: Apache-2.0
// Matched, disjoint-domain benchmark for packed BFL and a one-hidden-layer MLP.
#include "bfl/greedy.hpp"
#include "bfl/routed_binary.hpp"

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
#include <vector>
#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace {
using Clock = std::chrono::steady_clock;
constexpr std::size_t inputs = 10, train_size = 600, hidden = 16;
constexpr std::size_t ann_epochs = 500, inference_repeats = 5, calls_per_repeat = 200000;
using Input = std::array<bfl::Bit, inputs>;
struct Sample { Input x; bfl::Bit y; };
struct Split { std::vector<Sample> train, test; std::mt19937 rng_after_split; };
volatile std::uint64_t checksum = 0;

Split split_for(unsigned seed) {
    std::vector<unsigned> domain(1u << inputs);
    std::iota(domain.begin(), domain.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(domain.begin(), domain.end(), rng);
    Split result;
    for (std::size_t i = 0; i < domain.size(); ++i) {
        Input x{};
        for (std::size_t bit = 0; bit < inputs; ++bit) x[bit] = (domain[i] >> bit) & 1u;
        const int score = -1 + x[0] + x[1] + (x[0] & x[1]) +
            (x[2] & x[3]) + (x[4] & x[5]) - x[6] - x[7] - (x[8] & x[9]);
        const Sample sample{x, static_cast<bfl::Bit>(score >= 1)};
        (i < train_size ? result.train : result.test).push_back(sample);
    }
    result.rng_after_split = rng;
    return result;
}

bfl::Dataset bfl_data(const std::vector<Sample>& samples) {
    bfl::Dataset result;
    result.reserve(samples.size());
    for (const auto& sample : samples)
        result.push_back({std::vector<bfl::Bit>(sample.x.begin(), sample.x.end()), {sample.y}});
    return result;
}

struct ANN {
    static constexpr std::size_t b1 = hidden * inputs;
    static constexpr std::size_t w2 = b1 + hidden;
    static constexpr std::size_t b2 = w2 + hidden;
    static constexpr std::size_t parameters = b2 + 1;
    std::array<double, parameters> weights{}, first{}, second{};

    explicit ANN(unsigned seed) {
        std::mt19937 rng(seed);
        std::normal_distribution<double> init(0, std::sqrt(2.0 / (inputs + hidden)));
        for (std::size_t i = 0; i < b1; ++i) weights[i] = init(rng);
        for (std::size_t i = w2; i < b2; ++i) weights[i] = init(rng);
    }
    double logit(const Input& x, std::array<double, hidden>& activation) const noexcept {
        double result = weights[b2];
        for (std::size_t h = 0; h < hidden; ++h) {
            double z = weights[b1 + h];
            for (std::size_t i = 0; i < inputs; ++i) z += weights[h * inputs + i] * x[i];
            activation[h] = std::tanh(z);
            result += weights[w2 + h] * activation[h];
        }
        return result;
    }
    bfl::Bit predict(const Input& x) const noexcept {
        std::array<double, hidden> activation{};
        return logit(x, activation) >= 0;
    }
    void fit(const std::vector<Sample>& data) {
        constexpr double rate = 0.03, beta1 = 0.9, beta2 = 0.999;
        double beta1_power = 1, beta2_power = 1;
        for (std::size_t epoch = 0; epoch < ann_epochs; ++epoch) {
            std::array<double, parameters> gradient{};
            for (const auto& sample : data) {
                std::array<double, hidden> activation{};
                const double z = logit(sample.x, activation);
                const double delta = 1.0 / (1.0 + std::exp(-z)) - sample.y;
                gradient[b2] += delta;
                for (std::size_t h = 0; h < hidden; ++h) {
                    gradient[w2 + h] += delta * activation[h];
                    const double back = delta * weights[w2 + h] *
                                        (1.0 - activation[h] * activation[h]);
                    gradient[b1 + h] += back;
                    for (std::size_t i = 0; i < inputs; ++i)
                        gradient[h * inputs + i] += back * sample.x[i];
                }
            }
            beta1_power *= beta1;
            beta2_power *= beta2;
            for (std::size_t i = 0; i < parameters; ++i) {
                const double g = gradient[i] / data.size();
                first[i] = beta1 * first[i] + (1 - beta1) * g;
                second[i] = beta2 * second[i] + (1 - beta2) * g * g;
                weights[i] -= rate * (first[i] / (1 - beta1_power)) /
                    (std::sqrt(second[i] / (1 - beta2_power)) + 1e-8);
            }
        }
    }
};

template<class Model> double accuracy(const Model& model, const std::vector<Sample>& data) {
    std::size_t correct = 0;
    for (const auto& sample : data) correct += model.predict(sample.x) == sample.y;
    return double(correct) / data.size();
}

struct Counts { double instructions = NAN, cycles = NAN, branches = NAN,
                       branch_misses = NAN, cache_misses = NAN; };

#ifdef __linux__
class Counters {
public:
    Counters() {
        const std::array<std::uint64_t, 5> events{PERF_COUNT_HW_INSTRUCTIONS,
            PERF_COUNT_HW_CPU_CYCLES, PERF_COUNT_HW_BRANCH_INSTRUCTIONS,
            PERF_COUNT_HW_BRANCH_MISSES, PERF_COUNT_HW_CACHE_MISSES};
        for (std::size_t i = 0; i < events.size(); ++i) {
            perf_event_attr attr{};
            attr.type = PERF_TYPE_HARDWARE;
            attr.size = sizeof(attr);
            attr.config = events[i];
            attr.disabled = 1;
            attr.exclude_kernel = 1;
            attr.exclude_hv = 1;
            fd_[i] = static_cast<int>(syscall(SYS_perf_event_open, &attr, 0, -1, -1, 0));
        }
    }
    ~Counters() { for (int fd : fd_) if (fd >= 0) close(fd); }
    void start() { for (int fd : fd_) if (fd >= 0) {
        ioctl(fd, PERF_EVENT_IOC_RESET, 0); ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
    } }
    Counts stop(std::size_t calls) {
        std::array<double, 5> result{};
        for (std::size_t i = 0; i < fd_.size(); ++i) {
            std::uint64_t value = 0;
            if (fd_[i] >= 0) {
                ioctl(fd_[i], PERF_EVENT_IOC_DISABLE, 0);
                if (read(fd_[i], &value, sizeof(value)) == sizeof(value)) {
                    result[i] = double(value) / calls;
                    continue;
                }
            }
            result[i] = NAN;
        }
        return {result[0], result[1], result[2], result[3], result[4]};
    }
private:
    std::array<int, 5> fd_{{-1, -1, -1, -1, -1}};
};
#else
class Counters {
public:
    void start() {}
    Counts stop(std::size_t) { return {}; }
};
#endif

struct Timing { double mean = 0, median = 0, p95 = 0; Counts counts; };

template<class Predict>
Timing inference_timing(const Split& split, Predict predict) {
    for (std::size_t i = 0; i < 10000; ++i)
        checksum += predict(split.test[i % split.test.size()].x);
    Counters counters;
    std::array<double, inference_repeats> times{};
    Counts measured;
    for (std::size_t repeat = 0; repeat < inference_repeats; ++repeat) {
        counters.start();
        const auto start = Clock::now();
        for (std::size_t i = 0; i < calls_per_repeat; ++i)
            checksum += predict(split.test[i % split.test.size()].x);
        const auto stop = Clock::now();
        const auto count = counters.stop(calls_per_repeat);
        if (repeat == 0) measured = count;
        times[repeat] = std::chrono::duration<double, std::nano>(stop - start).count() /
                        calls_per_repeat;
    }
    const double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    std::sort(times.begin(), times.end());
    return {mean, times[times.size() / 2], times.back(), measured};
}

void print_row(const char* model, unsigned seed, double train, double test,
               std::size_t model_bytes, std::size_t parameter_bytes,
               std::size_t train_memory, std::size_t infer_memory,
               double train_ms, std::size_t epochs, Timing timing,
               const bfl::GreedyTrainReport* report,
               const bfl::RoutedBinaryTrainReport* routed = nullptr) {
    std::cout << model << ',' << seed << ',' << train << ',' << test << ','
              << train - test << ',' << model_bytes << ',' << parameter_bytes << ','
              << train_memory << ',' << infer_memory << ',' << train_ms << ','
              << (train_ms * 1e6 / (train_size * epochs)) << ',' << train_ms / epochs << ','
              << timing.mean << ',' << timing.median << ',' << timing.p95 << ','
              << timing.counts.instructions << ',' << timing.counts.cycles << ','
              << timing.counts.branches << ',' << timing.counts.branch_misses << ','
              << timing.counts.cache_misses << ',';
    if (routed) {
        std::cout << routed->initial_errors << ',' << routed->final_errors << ','
                  << routed->forget_count << ',' << routed->effective_changed_bits << ','
                  << routed->candidate_evaluations << ',' << routed->popcount_operations << ','
                  << routed->accepted_updates << ',' << routed->rejected_candidates << ','
                  << routed->effective_changed_bits << ','
                  << (routed->accepted_updates ? double(routed->candidate_evaluations) /
                      routed->accepted_updates : NAN) << ',' << routed->err_before_sum << ','
                  << routed->err_after_sum << ',' << routed->err_improves << ','
                  << routed->err_ties_forget_improves << ',' <<
                     routed->err_forget_ties_change_improves;
    } else if (report) {
        std::cout << report->initial_error_bits << ',' << report->final_error_bits <<
            ",NaN,NaN," << report->tested_flips << ",NaN," << report->accepted_flips <<
            ",NaN,NaN," << (report->accepted_flips ? double(report->tested_flips) /
                report->accepted_flips : NAN) << ",NaN,NaN,NaN,NaN,NaN";
    } else std::cout << "NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN";
    std::cout << '\n';
}
} // namespace

int main(int argc, char** argv) {
    const bool one_seed = argc == 3 && std::string(argv[1]) == "--seed";
    const unsigned seeds = argc == 1 ? 20u :
        static_cast<unsigned>(std::stoul(argv[one_seed ? 2 : 1]));
    if ((!one_seed && (argc > 2 || seeds < 20 || seeds > 1000)) ||
        (one_seed && !seeds))
        throw std::invalid_argument("Usage: benchmark_bfl_vs_ann [seeds >= 20] | --seed N");
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "model,seed,train_accuracy,test_accuracy,generalization_gap,model_bytes,parameter_state_bytes,temporary_training_bytes,temporary_inference_bytes,training_ms,training_ns_per_sample,training_ms_per_epoch,ns_per_inference,median_ns_per_inference,p95_batch_ns_per_inference,instructions_per_inference,cycles_per_inference,branches_per_inference,branch_misses_per_inference,cache_misses_per_inference,err_before,err_after,forget_count,change_cost,candidate_evaluations,popcount_operations,accepted_updates,rejected_candidates,effective_changed_bits,average_candidate_evaluations_per_update,err_before_sum,err_after_sum,err_improves,err_ties_forget_improves,err_forget_ties_change_improves\n";
    for (unsigned seed = one_seed ? seeds : 1; seed <= seeds; ++seed) {
        const auto split = split_for(seed);
        const auto training = bfl_data(split.train);
        const auto testing = bfl_data(split.test);
        {
            bfl::GreedyModel model({inputs, 1, 2, hidden, seed * 101 + 42});
            bfl::GreedyTrainConfig config;
            config.update_rule = bfl::GreedyTrainConfig::UpdateRule::SingleBit;
            config.max_epochs = 30;
            config.candidate_limit = 64;
            config.patience = 5;
            config.seed = seed * 103 + 1234;
            const auto start = Clock::now();
            const auto report = model.train(training, config);
            const double train_ms = std::chrono::duration<double, std::milli>(
                Clock::now() - start).count();
            std::vector<bfl::Bit> scratch(model.node_count());
            const auto timing = inference_timing(split, [&](const Input& x) {
                bfl::Bit output = 0;
                model.predict_into(x.data(), &output, scratch.data());
                return output;
            });
            print_row("bfl_old", seed,
                      model.evaluate(training).bit_accuracy,
                      model.evaluate(testing).bit_accuracy,
                      model.model_storage_bytes() + 60, model.model_storage_bytes(),
                      report.training_working_bytes,
                      scratch.size() * sizeof(bfl::Bit), train_ms,
                      std::max<std::size_t>(1, report.epochs_ran), timing, &report);
        }
        std::mt19937 rng = split.rng_after_split;
        std::uint64_t initial_state = 0;
        for (std::size_t bit = 0; bit < bfl::RoutedBinaryModel::feature_bits; ++bit)
            if (rng() & 1u) initial_state |= std::uint64_t{1} << bit;
        bfl::RoutedBinaryModel routed(initial_state);
        std::vector<bfl::RoutedBinarySample> routed_train;
        routed_train.reserve(split.train.size());
        for (const auto& sample : split.train)
            routed_train.push_back({sample.x, sample.y});
        const auto routed_start = Clock::now();
        const auto routed_report = routed.train(routed_train, rng, 20);
        const double routed_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - routed_start).count();
        const auto routed_timing = inference_timing(split, [&](const Input& x) {
            return routed.predict(x);
        });
        print_row("bfl_binary_global", seed, accuracy(routed, split.train),
                  accuracy(routed, split.test), bfl::RoutedBinaryModel::model_storage_bytes(),
                  bfl::RoutedBinaryModel::model_storage_bytes(),
                  routed_report.training_working_bytes, 0, routed_ms,
                  routed_report.epochs, routed_timing, nullptr, &routed_report);
        ANN model(seed * 101 + 42);
        const auto start = Clock::now();
        model.fit(split.train);
        const double train_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - start).count();
        const auto timing = inference_timing(split, [&](const Input& x) {
            return model.predict(x);
        });
        print_row("ann_16", seed, accuracy(model, split.train), accuracy(model, split.test),
                  ANN::parameters * sizeof(double), ANN::parameters * sizeof(double),
                  (3 * ANN::parameters + hidden) * sizeof(double),
                  hidden * sizeof(double), train_ms, ann_epochs, timing, nullptr);
    }
    return checksum == UINT64_MAX ? 1 : 0;
}
