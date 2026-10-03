# Greedy MUX model

`bfl::GreedyModel` is a C++17 implementation of the layered BFL model used in the generalization benchmark. It lives in the main `bfl` library and is declared in [`include/bfl/greedy.hpp`](../include/bfl/greedy.hpp).

Each MUX reads three signals from the preceding layer: data `A`, data `B`, and dynamic selector `D`. It stores one learned bit `M` and computes:

```text
S = D XOR M
OUT = S ? B : A
```

The fixed topology is generated from the model seed. Each MUX occupies one 64-bit word containing three 21-bit source indices and its learned bit. Input constants can be supplied as fixed columns, as in the generalization benchmark.

Training traces the selector and selected data branch backward from wrong output bits. It tests flips of the reached memory bits and retains a flip only when total training Hamming error decreases. The test uses the complete training dataset and the current state after earlier accepted flips. Candidate order is optionally shuffled. `GreedyTrainConfig` controls epochs, candidate limit, patience, and seed. This rule differs from the direct update used by `bfl::Graph`.

`GreedyTrainConfig::UpdateRule::MaskedChangeUsed` implements the proposed joint update. For each incorrect output of a training example, `CHANGE` is that output's `OUT XOR TARGET` bit. `USED` is the mask of MUXes in its active dependency path: each MUX, its selected data source, and its selector source. The output error is projected onto that path, since output bits and selector bits have different dimensions. The candidate is `S_c = S XOR F`, where `F` contains all reached MUX state bits. Duplicate masks are collapsed within each candidate generation. Each candidate is scored on the **entire** training dataset:

```text
J_D(S) = error_weight * total_error_bits(S) + used_weight * total_used_bits(S)
K = [J_D(S_c) < J_D(S)]
S' = S XOR (K AND F)
```

`total_used_bits` sums the number of distinct active MUXes over examples, taking the union of paths from all outputs for each example. Without `NOT(USED)`, the defaults use `error_weight = 10` and `used_weight = 1`, giving the proposed 0.1 ratio. With `config.penalize_unused = true`, the default error weight is 17, selected by validation on the benchmark's three tasks. An explicit `config.error_weight` overrides either default. Set it to 1 for equal per-bit weights or set `used_weight = 0` for an error-only control. Ties revert. Candidate masks are regenerated at each epoch and after every accepted update, so each proposal uses the current `CHANGE` and `USED`. `candidate_limit` bounds tested proposals per epoch. The original one-bit rule remains the default for compatibility.

Set `config.penalize_unused = true` to replace the usage term with `popcount(NOT(USED))`. The complement is bounded to the model's MUX state bits for every example, so `total_unused_bits = data.size() * model.node_count() - total_used_bits`. The update mask stays `F = CHANGE AND USED`. Because the number of MUXes is fixed, minimizing `errors + total_unused_bits` is equivalent to minimizing `errors - total_used_bits`; one additional used MUX exactly offsets one additional error bit, yielding a rejected tie. It tests the opposite structural preference from the default usage penalty.

The corresponding target objective is:

```text
S* = argmin_S sum_D [popcount(CHANGE_S) + lambda * popcount(NOT(USED_S))]
```

`NOT(USED)` is restricted to valid MUX state bits. For `lambda = p / q`, `config.set_unused_lambda(p, q)` configures the equivalent integer score `q * total_error_bits + p * total_unused_bits`, avoiding floats. The validation-selected value is `lambda = 1/17`; call `config.set_unused_lambda(1, 17)` or set `config.update_rule` to `MaskedChangeUsed` and `config.penalize_unused = true` to use that default. The earlier equal-weight experiment uses `config.set_unused_lambda(1, 1)`. `GreedyModel::train` accepts only improving proposals and does **not** guarantee the global `argmin` over all selector states. Candidate and epoch limits can stop it before even a local minimum.

```cpp
#include "bfl/greedy.hpp"

bfl::GreedyModel model({2, 1, 3, 32, 44});
bfl::Dataset data = {
    {{0, 0}, {0}}, {{0, 1}, {1}},
    {{1, 0}, {1}}, {{1, 1}, {0}}
};
bfl::GreedyTrainConfig config;
config.max_epochs = 100;
config.update_rule = bfl::GreedyTrainConfig::UpdateRule::MaskedChangeUsed;
const auto report = model.train(data, config);
const auto prediction = model.predict({1, 0});
model.save("model.bflg");
```

`predict_batch`, `evaluate`, `reset`, `export_state`, and `import_state` are also available. `evaluate` reports error, accuracy, and `used_bits`. `save` writes a versioned little-endian `.bflg` file containing topology and learned state. `load` reads that format and imports the earlier little-endian `.bfl` formats v1 and v2. The implementation evaluates up to 64 examples per 64-bit word. It does not use architecture-specific SIMD or dependency-cone caching; these optimizations affect speed, not the specified acceptance rule. The [benchmark](../results/README.md#c-generalization-comparison) selects a topology on validation data and evaluates the test set afterward.

## Routed binary-global model

`bfl::RoutedBinaryModel` in [`include/bfl/routed_binary.hpp`](../include/bfl/routed_binary.hpp) implements the standalone benchmark's greedy directly in the library. It takes ten binary inputs, forms a fixed 16-bit feature mask (constant one, ten inputs, and five adjacent pairwise ANDs), and routes on the first five inputs to one of 32 learned 16-bit states. Prediction assigns each active feature weight +1 or -1 according to its state bit and returns whether their integer sum is nonnegative.

For each presented training sample, candidate `CHANGE` masks contain zero, one, two, or three feature bits. `USED` is the active feature mask for that sample. With the default `mask_with_used=true`, the effective change is `CHANGE AND USED`, and the route state is updated by `S' = S XOR (CHANGE AND USED)`. The optional `false` setting reproduces the standalone experiment's no-AND-U ablation; it is not the official update rule. Each candidate is scored against all training examples in the same route by `(ERR_after, FORGET, popcount(CHANGE AND USED))`. Examples in other routes have fixed predictions and therefore add the same constant to every candidate's global ERR and zero to FORGET. The no-op candidate is included, so a state update requires a lexicographic improvement.

The current benchmark's `bfl_binary_global` row uses this routed model. Its `bfl_old` row retains the earlier single-bit MUX model, so differences between those rows include architecture and feature design as well as greedy search. Reproduce both the standalone AND-U control and the matched ANN comparison with `bash scripts/reproduce_bfl_vs_ann.sh results 50`; see [results/README.md](../results/README.md#binary-global-vs-ann).
