# Bit Flip Learning (BFL)

This C++17 library provides three BFL learning models. `bfl::GreedyModel` is a layered MUX network whose selector is a graph signal XOR a learned inversion bit. It supports both single-bit greedy flips and joint `CHANGE AND USED` path flips, accepted by a global error-plus-usage score. `bfl::Graph` is the direct-selector, route-key experiment. `bfl::RoutedBinaryModel` learns binary feature states on input-defined routes. The [greedy model documentation](docs/greedy.md) defines their training rules.

The [matched held-out comparison](results/README.md#changeused-greedy-comparison) runs the greedy rules with `USED`, `NOT(USED)`, and no usage term, plus the direct-state graph and ANNs with 4 and 16 hidden units on identical data splits. A [validation-only search](results/README.md#selecting-the-error-weight-x) over 132 error weights selected `X=17` for `X * error + NOT(USED)`. That variant reached 68.9% pooled test accuracy across OR, XOR, and majority; the earlier single-bit greedy reached 70.1%, and both ANNs reached 100%.

## State and route

For each supervised output comparison, `CHANGE = OUT XOR TARGET`. `USED` is traced backward from that output through every selected data branch and the selector predecessor of each signal MUX. For an adaptive MUX, the addressed selector state changes by exactly:

```text
S' = S XOR (CHANGE AND USED)
```

No scalar loss controls acceptance. The reported Hamming loss is a measurement only. Signal MUXes have no mutable selector state and produce no effective change event.

After a comparison, the library records `A_i = CHANGE_i AND USED_i` for each adaptive MUX. On the next evaluation, MUX `j` uses `RouteKey{count, prefix}`, where `count = sum_{i<j} A_i` and `prefix` contains the first `k` MUX event bits that precede `j`, in graph order. The default `k=0` reproduces count routing. Construct `Graph(k)` with `0 <= k <= 64` to retain part of the event positions; for example, `Graph(8)` retains up to eight bits. A MUX's own event enters only later routes. At the same MUX, equal keys address the same selector state. New route states inherit that MUX's initial state and are materialized only when changed.

The first evaluation uses all-zero events. Training proceeds in dataset order, then output order, with a fresh forward pass per comparison. Inference keeps the final event vector from training and does not change it. `Graph::set_routing_events` permits restoring a valid routing context explicitly. The library has no on-disk model serialization yet; a `Graph` copy preserves its in-memory state and event vector.

## Build and run

Configure, build, and run the tests with CMake:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The text generation and BFLCODE experiments live in the sibling `BFL-extras/` directory. They are separate from the BFL library and its GitHub source tree.

For a direct compiler build, use:

```sh
g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic -Iinclude src/bfl.cpp src/greedy.cpp tests/test_bfl.cpp -o bfl_tests
./bfl_tests
g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic -Iinclude src/bfl.cpp src/greedy.cpp tests/test_greedy.cpp -o bfl_greedy_tests
./bfl_greedy_tests
g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic -Iinclude src/bfl.cpp src/greedy.cpp examples/minimal.cpp -o bfl_example
./bfl_example
bash scripts/reproduce.sh results/local
bash scripts/reproduce_generalization.sh results/local-generalization
bash scripts/reproduce_bfl_vs_ann.sh results 50
```

The benchmark compares BFL with a two-bit majority baseline on parity and majority targets. For each input width and seed, it shuffles the finite binary domain, fits on the first half, and evaluates on the next quarter. It reports training and held-out loss and accuracy, training and inference time, model bytes, node and state counts, distinct route counts, and mean training examples per route. The batch API is a scalar loop over examples, so its time includes the same evaluation semantics. See [results/README.md](results/README.md) for measured results and prior-run provenance.

`Graph::memory_bytes()` estimates object and allocated vector/map storage, including route states and event bits; it excludes allocator metadata, temporary evaluation buffers, datasets, and executable code. The online rule can oscillate or hurt held-out accuracy. Sharing a route is a structural property, not a demonstrated generalization benefit.

`GreedyModel` is documented in [docs/greedy.md](docs/greedy.md). Its MUX topology and learned state occupy one packed 64-bit word per node. The generalization benchmark uses 16 independently initialized topologies and selects one using validation data; the test set is evaluated afterward. The source tree and build use C++ only.

The separate [binary-global benchmark](results/README.md#binary-global-vs-ann) uses `RoutedBinaryModel`, integrated from the [standalone AND-U experiment](benchmarks/standalone_routed_binary_andu.cpp), as the primary BFL binary-global model. It compares that model with the original single-bit MUX greedy and a 16-unit ANN on 50 disjoint 10-bit splits. Its CSV includes accuracy, storage, temporary memory, training and inference timing, and greedy decision counters. The old MUX model and its API remain available.

## Publishing

Run `bash scripts/publish.sh --prepare` to build and test the current tree. Review `git status`, commit the intended sources and results, then run `bash scripts/publish.sh --publish` to push the clean `main` branch to the existing `fgp-dev/BFL` repository.
