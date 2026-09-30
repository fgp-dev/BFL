# Greedy MUX model

`bfl::GreedyModel` is a C++17 implementation of the layered BFL model used in the generalization benchmark. It lives in the main `bfl` library and is declared in [`include/bfl/greedy.hpp`](../include/bfl/greedy.hpp).

Each MUX reads three signals from the preceding layer: data `A`, data `B`, and dynamic selector `D`. It stores one learned bit `M` and computes:

```text
S = D XOR M
OUT = S ? B : A
```

The fixed topology is generated from the model seed. Each MUX occupies one 64-bit word containing three 21-bit source indices and its learned bit. Input constants can be supplied as fixed columns, as in the generalization benchmark.

Training traces the selector and selected data branch backward from wrong output bits. It tests flips of the reached memory bits and retains a flip only when total training Hamming error decreases. The test uses the complete training dataset and the current state after earlier accepted flips. Candidate order is optionally shuffled. `GreedyTrainConfig` controls epochs, candidate limit, patience, and seed. This rule differs from the direct update used by `bfl::Graph`.

```cpp
#include "bfl/greedy.hpp"

bfl::GreedyModel model({2, 1, 3, 32, 44});
bfl::Dataset data = {
    {{0, 0}, {0}}, {{0, 1}, {1}},
    {{1, 0}, {1}}, {{1, 1}, {0}}
};
bfl::GreedyTrainConfig config;
config.max_epochs = 100;
const auto report = model.train(data, config);
const auto prediction = model.predict({1, 0});
model.save("model.bflg");
```

`predict_batch`, `evaluate`, `reset`, `export_state`, and `import_state` are also available. `save` writes a versioned little-endian `.bflg` file containing topology and learned state. `load` reads that format and imports the earlier little-endian `.bfl` formats v1 and v2. The implementation evaluates up to 64 examples per 64-bit word. It does not use architecture-specific SIMD or dependency-cone caching; these optimizations affect speed, not the specified acceptance rule. The [benchmark](../results/README.md#c-generalization-comparison) selects a topology on validation data and evaluates the test set afterward.
