
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

using namespace std;

constexpr int INPUTS = 10;
constexpr int FEATURES = 16;
constexpr int ROUTE_BITS = 5;          // 32 routes
constexpr int ROUTES = 1 << ROUTE_BITS;
constexpr int MAX_FLIPS = 3;
constexpr int TRAIN_N = 600;
constexpr int EPOCHS = 20;

struct Sample {
    array<int, INPUTS> x{};
    int y{};
};

array<int, FEATURES> features(const array<int, INPUTS>& x)
{
    array<int, FEATURES> f{};
    f[0] = 1;

    for (int i = 0; i < INPUTS; ++i)
        f[i + 1] = x[i];

    f[11] = x[0] & x[1];
    f[12] = x[2] & x[3];
    f[13] = x[4] & x[5];
    f[14] = x[6] & x[7];
    f[15] = x[8] & x[9];

    return f;
}

uint64_t activeMask(const Sample& s)
{
    auto f = features(s.x);
    uint64_t m = 0;
    for (int i = 0; i < FEATURES; ++i)
        if (f[i])
            m |= (1ULL << i);
    return m;
}

int target(const array<int, INPUTS>& x)
{
    auto f = features(x);

    int score =
        -f[0]
        + f[1]
        + f[2]
        + f[11]
        + f[12]
        + f[13]
        - f[7]
        - f[8]
        - f[15];

    return score >= 1;
}

int routeOf(const Sample& s)
{
    int r = 0;
    for (int i = 0; i < ROUTE_BITS; ++i)
        r |= (s.x[i] << i);
    return r;
}

vector<uint64_t> makeCandidates()
{
    vector<uint64_t> c;
    c.push_back(0); // no-op is a legal candidate

    for (int i = 0; i < FEATURES; ++i)
        c.push_back(1ULL << i);

    if (MAX_FLIPS >= 2) {
        for (int i = 0; i < FEATURES; ++i)
            for (int j = i + 1; j < FEATURES; ++j)
                c.push_back((1ULL << i) | (1ULL << j));
    }

    if (MAX_FLIPS >= 3) {
        for (int i = 0; i < FEATURES; ++i)
            for (int j = i + 1; j < FEATURES; ++j)
                for (int k = j + 1; k < FEATURES; ++k)
                    c.push_back(
                        (1ULL << i) |
                        (1ULL << j) |
                        (1ULL << k)
                    );
    }

    return c;
}

class BinaryGreedyBFL {
    array<uint64_t, ROUTES> S{};
    bool useAndU;
    vector<uint64_t> candidates;

public:
    long long accepted_updates = 0;
    long long proposed_nonzero = 0;
    long long effective_flips = 0;
    long long forget_events = 0;

    explicit BinaryGreedyBFL(bool andU, uint64_t initial_state)
        : useAndU(andU), candidates(makeCandidates())
    {
        const uint64_t MASK = (1ULL << FEATURES) - 1ULL;
        for (int r = 0; r < ROUTES; ++r)
            S[r] = (initial_state ^
                    (0x9E3779B97F4A7C15ULL * (r + 1))) & MASK;
    }

    int predictWithState(const Sample& sample, uint64_t state) const
    {
        auto f = features(sample.x);
        int score = 0;

        for (int i = 0; i < FEATURES; ++i) {
            if (!f[i]) continue;
            const int w = (state & (1ULL << i)) ? +1 : -1;
            score += w;
        }

        return score >= 0;
    }

    int predict(const Sample& sample) const
    {
        return predictWithState(sample, S[routeOf(sample)]);
    }

    // Binary definitions:
    // E = Y XOR P
    // F = NOT(E_before) AND E_after
    //
    // The tuple is compared lexicographically:
    // (popcount(E_after), popcount(F), popcount(effective_CHANGE))
    tuple<int,int,int> scoreCandidate(
        int route,
        uint64_t effective_change,
        const vector<Sample>& memory
    ) const
    {
        uint64_t before_state = S[route];
        uint64_t after_state = before_state ^ effective_change;

        int err = 0;
        int forget = 0;

        for (const auto& s : memory) {
            if (routeOf(s) != route)
                continue;

            int e_before = s.y ^ predictWithState(s, before_state);
            int e_after  = s.y ^ predictWithState(s, after_state);

            err += e_after;
            forget += ((!e_before) & e_after);
        }

        int change_cost = __builtin_popcountll(effective_change);

        return {err, forget, change_cost};
    }

    void trainOn(
        const Sample& current,
        const vector<Sample>& memory
    )
    {
        const int route = routeOf(current);
        const uint64_t U = activeMask(current);

        auto baseline = scoreCandidate(route, 0, memory);

        tuple<int,int,int> best = baseline;
        uint64_t best_effective = 0;
        uint64_t best_raw = 0;

        for (uint64_t C : candidates) {
            uint64_t effective =
                useAndU ? (C & U) : C;

            auto sc = scoreCandidate(route, effective, memory);

            if (sc < best) {
                best = sc;
                best_effective = effective;
                best_raw = C;
            }
        }

        if (best_effective != 0) {
            ++accepted_updates;
            if (best_raw != 0) ++proposed_nonzero;

            int old_err, old_forget, old_change;
            tie(old_err, old_forget, old_change) = baseline;

            int new_err, new_forget, new_change;
            tie(new_err, new_forget, new_change) = best;

            forget_events += new_forget;
            effective_flips += __builtin_popcountll(best_effective);

            S[route] ^= best_effective;
        }
    }
};

template<typename Model>
double accuracy(const Model& model, const vector<Sample>& data)
{
    int correct = 0;
    for (const auto& s : data)
        correct += (model.predict(s) == s.y);

    return 100.0 * correct / static_cast<double>(data.size());
}

vector<Sample> fullDomain()
{
    vector<Sample> data;
    data.reserve(1 << INPUTS);

    for (int v = 0; v < (1 << INPUTS); ++v) {
        Sample s;
        for (int i = 0; i < INPUTS; ++i)
            s.x[i] = (v >> i) & 1;
        s.y = target(s.x);
        data.push_back(s);
    }

    return data;
}

int main(int argc, char** argv)
{
    uint32_t seed = 12345;
    if (argc > 1)
        seed = static_cast<uint32_t>(stoul(argv[1]));

    mt19937 rng(seed);

    auto all = fullDomain();
    shuffle(all.begin(), all.end(), rng);

    vector<Sample> train(all.begin(), all.begin() + TRAIN_N);
    vector<Sample> test(all.begin() + TRAIN_N, all.end());

    uint64_t initial_state = 0;
    for (int i = 0; i < FEATURES; ++i)
        if (rng() & 1)
            initial_state |= (1ULL << i);

    BinaryGreedyBFL withU(true, initial_state);
    BinaryGreedyBFL withoutU(false, initial_state);

    vector<Sample> memory;
    memory.reserve(train.size());

    // Build memory once. The greedy objective sees the entire training
    // set during each epoch; all operations in its objective are binary.
    memory = train;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        shuffle(train.begin(), train.end(), rng);

        for (const auto& s : train) {
            withU.trainOn(s, memory);
            withoutU.trainOn(s, memory);
        }
    }

    cout << "seed=" << seed << "\n";
    cout << "train_n=" << train.size() << "\n";
    cout << "test_n=" << test.size() << "\n";
    cout << "routes=" << ROUTES << "\n\n";

    cout << "WITH_AND_U\n";
    cout << "train=" << accuracy(withU, memory) << "\n";
    cout << "test=" << accuracy(withU, test) << "\n";
    cout << "updates=" << withU.accepted_updates << "\n";
    cout << "effective_flips=" << withU.effective_flips << "\n";
    cout << "forget_events=" << withU.forget_events << "\n\n";

    cout << "WITHOUT_AND_U\n";
    cout << "train=" << accuracy(withoutU, memory) << "\n";
    cout << "test=" << accuracy(withoutU, test) << "\n";
    cout << "updates=" << withoutU.accepted_updates << "\n";
    cout << "effective_flips=" << withoutU.effective_flips << "\n";
    cout << "forget_events=" << withoutU.forget_events << "\n";
}
