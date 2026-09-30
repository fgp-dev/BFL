# Bit Flip Learning (BFL)

This C++17 library provides two explicit BFL learning models. `bfl::GreedyModel` is a layered MUX network whose selector is a graph signal XOR a learned inversion bit. It traces active paths, tests candidate bit flips, and keeps a flip only when global training error falls. `bfl::Graph` is the direct-selector, route-key experiment. Both models and their different learning rules are described in [BFL_Theory.tex](BFL_Theory.tex).

The [matched held-out comparison](results/README.md#c-generalization-comparison) runs both C++ models and a small ANN on identical data splits. On the OR task, `GreedyModel` with validation-based topology selection reached 76.2% test accuracy and the ANN reached 100%; this is a new experiment, not a reproduction of the historical ~80% figure.

## State and route

For each supervised output comparison, `CHANGE = OUT XOR TARGET`. `USED` is traced backward from that output through every selected data branch and the selector predecessor of each signal MUX. For an adaptive MUX, the addressed selector state changes by exactly:

```text
S' = S XOR (CHANGE AND USED)
```

No scalar loss controls acceptance. The reported Hamming loss is a measurement only. Signal MUXes have no mutable selector state and produce no effective change event.

After a comparison, the library records `A_i = CHANGE_i AND USED_i` for each adaptive MUX. On the next evaluation, MUX `j` uses `RouteKey{count, prefix}`, where `count = sum_{i<j} A_i` and `prefix` contains the first `k` MUX event bits that precede `j`, in graph order. The default `k=0` reproduces count routing. Construct `Graph(k)` with `0 <= k <= 64` to retain part of the event positions; for example, `Graph(8)` retains up to eight bits. A MUX's own event enters only later routes. At the same MUX, equal keys address the same selector state. New route states inherit that MUX's initial state and are materialized only when changed.

The first evaluation uses all-zero events. Training proceeds in dataset order, then output order, with a fresh forward pass per comparison. Inference keeps the final event vector from training and does not change it. `Graph::set_routing_events` permits restoring a valid routing context explicitly. The library has no on-disk model serialization yet; a `Graph` copy preserves its in-memory state and event vector.

## Build and run

```sh
g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic -Iinclude src/bfl.cpp src/greedy.cpp tests/test_bfl.cpp -o bfl_tests
./bfl_tests
g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic -Iinclude src/bfl.cpp src/greedy.cpp tests/test_greedy.cpp -o bfl_greedy_tests
./bfl_greedy_tests
g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic -Iinclude src/bfl.cpp src/greedy.cpp examples/minimal.cpp -o bfl_example
./bfl_example
bash scripts/reproduce.sh results/local
bash scripts/reproduce_generalization.sh results/local-generalization
```

CMake is supported when available: configure, build, then run `ctest`. The benchmark compares BFL with a two-bit majority baseline on parity and majority targets. For each input width and seed, it shuffles the finite binary domain, fits on the first half, and evaluates on the next quarter. It reports training and held-out loss and accuracy, training and inference time, model bytes, node and state counts, distinct route counts, and mean training examples per route. The batch API is a scalar loop over examples, so its time includes the same evaluation semantics. See [results/README.md](results/README.md) for measured results and prior-run provenance.

`Graph::memory_bytes()` estimates object and allocated vector/map storage, including route states and event bits; it excludes allocator metadata, temporary evaluation buffers, datasets, and executable code. The online rule can oscillate or hurt held-out accuracy. Sharing a route is a structural property, not a demonstrated generalization benefit.

`GreedyModel` is documented in [docs/greedy.md](docs/greedy.md). Its MUX topology and learned state occupy one packed 64-bit word per node. The generalization benchmark uses 16 independently initialized topologies and selects one using validation data; the test set is evaluated afterward. The source tree and build use C++ only.
