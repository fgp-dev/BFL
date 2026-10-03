# Benchmark runs

The partial-prefix experiment is in [`run_2026-09-30-prefix/benchmark.csv`](run_2026-09-30-prefix/benchmark.csv). Reproduce it with `bash scripts/reproduce.sh results/local`. It compares `k=0,4,8,16` prefix bits, three seeds, widths 4, 6, and 8, and one or three online epochs. `k=0` is the count-only control. The file includes timings, model size, train and held-out loss, exact accuracy, state updates, route counts, and the number of adaptive selector addresses reached by USED traces.

Arithmetic means over seeds 1, 2, and 3 after **one epoch**:

| Input bits | Prefix bits | Train bit accuracy | Test bit accuracy | Distinct routes | Used adaptive addresses |
|---:|---:|---:|---:|---:|---:|
| 4 | 0 | .875 | .375 | 2.00 | 16 |
| 4 | 4 | .875 | .375 | 4.67 | 16 |
| 4 | 8 | .875 | .375 | 8.67 | 16 |
| 4 | 16 | .854 | .375 | 16.67 | 16 |
| 6 | 0 | .818 | .510 | 2.00 | 64 |
| 6 | 4 | .818 | .510 | 3.67 | 64 |
| 6 | 8 | .818 | .510 | 7.67 | 64 |
| 6 | 16 | .818 | .510 | 15.67 | 64 |
| 8 | 0 | .861 | .448 | 2.00 | 256 |
| 8 | 4 | .861 | .448 | 3.67 | 256 |
| 8 | 8 | .861 | .448 | 6.33 | 256 |
| 8 | 16 | .861 | .448 | 14.33 | 256 |

After three epochs, all BFL variants reached 1.000 training bit accuracy at these widths. Their test bit accuracies remained .375, .510, and .448 respectively. Distinct routes increased with `k`, but the number of USED adaptive addresses did not change at any width or epoch setting. The lookup graph gives each input its own adaptive leaf, so this experiment demonstrates finer route labeling without testing much competition between examples for the same trained state. It does not establish a generalization benefit from the prefix.

`distinct_routes` counts distinct route keys observed at MUXes during supervised comparisons. `used_adaptive_addresses` counts distinct `(adaptive MUX, route key)` pairs reached by a USED trace. `adaptive_addresses` in the CSV includes all evaluated adaptive MUXes, including inactive lookup leaves; it grows with `k` although the used address count does not. `mean_examples_per_route` counts each presented training example once for each route present across its output comparisons, then divides by distinct routes. Inference uses the final routing event vector from training, so held-out predictions use one shared context.

The earlier [`run_2026-09-30-count/benchmark.csv`](run_2026-09-30-count/benchmark.csv) records the count-only implementation before the route-key type and its memory estimate changed. The `run_2026-09-30`, `run_2026-09-30-packed`, and `run_2026-09-30-scalar` directories record a different inversion-mask/proposal implementation and are not directly comparable to this direct-state experiment.

## C++ generalization comparison

The separate [`run_2026-09-30-generalization/benchmark.csv`](run_2026-09-30-generalization/benchmark.csv) compares the C++ [`GreedyModel`](../include/bfl/greedy.hpp), the direct-state lookup graph, and a trained one-hidden-layer ANN (MLP). `GreedyModel` ports the learning behavior of the earlier BFL implementation into this library; no C source or C build dependency is included. Reproduce the experiment with `bash scripts/reproduce_generalization.sh results/local-generalization`; summarize with `python3 scripts/summarize_generalization.py results/local-generalization/benchmark.csv`.

Each task has eight variable binary inputs plus a constant-one column. The target is OR or XOR of the first two bits, or majority of the first three bits; the other bits vary but do not determine the target. For each of 20 seeds, the 256 distinct inputs are shuffled and split into 128 training, 64 validation, and 64 test inputs. All models use the same split. The test set is evaluated only after training and, for the selected greedy model, after choosing a topology using validation accuracy. All models receive the same input columns.

`GreedyModel` uses three layers of 64 MUXes, up to 100 greedy epochs, 128 candidates per epoch, and patience 15. The `single` row uses the first random topology. The `selected` row trains 16 topologies and chooses the one with lowest validation error, breaking ties by training error; its reported time includes all 16 training runs. The MLP has 16 tanh hidden units and is trained for 1500 full-batch Adam epochs with one initialization. The direct-state control uses one adaptive lookup leaf per eight-bit input and three online epochs. Its topology memorizes the observed inputs, so its poor test score does not isolate the effect of the direct update law.

Mean accuracies over the 20 matched splits:

| Task | Model | Train | Validation | Test |
|---|---|---:|---:|---:|
| OR of two bits | GreedyModel, one topology | .797 | .685 | .668 |
| OR of two bits | GreedyModel, selected of 16 | .880 | .825 | **.762** |
| OR of two bits | direct-state lookup, 3 epochs | 1.000 | .245 | .245 |
| OR of two bits | MLP, 16 hidden units | 1.000 | 1.000 | **1.000** |
| Majority of three bits | GreedyModel, one topology | .773 | .670 | .637 |
| Majority of three bits | GreedyModel, selected of 16 | .834 | .793 | **.733** |
| Majority of three bits | direct-state lookup, 3 epochs | 1.000 | .477 | .503 |
| Majority of three bits | MLP, 16 hidden units | 1.000 | 1.000 | **1.000** |
| XOR of two bits | GreedyModel, one topology | .679 | .557 | .546 |
| XOR of two bits | GreedyModel, selected of 16 | .746 | .697 | **.608** |
| XOR of two bits | direct-state lookup, 3 epochs | 1.000 | .498 | .488 |
| XOR of two bits | MLP, 16 hidden units | 1.000 | 1.000 | **1.000** |

The new OR result is close to, but does not reproduce, the previously reported “around 80% versus 100%”: it is **76.2% versus 100%** under this declared protocol, with BFL test scores from 60.9% to 84.4% across seeds. No dataset or script for the historical figure was present in the supplied archive. On these tasks, validation-based topology selection improves `GreedyModel` over one arbitrary topology, but its held-out accuracy varies by task. The models use different learning rules and graph families; the matched split supports this specific comparison, not a general ranking of algorithms. The CSV also contains per-seed errors, model storage, and training times. `reported_model_bytes` uses each implementation's own storage estimate (packed MUX words, C++ graph allocation estimate, or MLP weights), so those byte values are not a matched total-memory measure.

## Search extensions: pair flips and annealing

[`run_2026-10-02-search/benchmark.csv`](run_2026-10-02-search/benchmark.csv) repeats the generalization protocol above with two optional `GreedyTrainConfig` extensions and one new reference row. Both extensions are off by default, and with defaults the model is bit-identical to the previous rule (checked on the same machine; see `environment.txt`).

- `pair_flips = N`: when an epoch accepts no single flip, try up to `N` random pairs of candidate bits and keep a pair only if the training error falls (`_pairs`, N = 64).
- `initial_temperature = T`: simulated annealing; a flip that does not lower the error is kept with probability `exp(-increase / T)`, T is multiplied by `cooling` each epoch, and the best state seen is restored at the end (`_anneal`, T = 1.5).
- `majority_class`: constant predictor of the training majority class. On the OR task the target is 1 for about three quarters of inputs, so this floor is high (test .729 here).

Mean test accuracy over 20 matched splits, validation-selected of 16 topologies (absolute values are not comparable to the 2026-09-30 table; see the note in `environment.txt`):

| Task | majority_class | greedy | + pair flips | + annealing | MLP |
|---|---:|---:|---:|---:|---:|
| OR of two bits | .729 | .786 | .842 | .830 | 1.000 |
| Majority of three bits | .476 | .767 | .794 | .770 | 1.000 |
| XOR of two bits | .476 | .637 | .667 | .663 | 1.000 |

Paired per-seed gain over the baseline greedy (mean, 95% normal interval over 20 seeds): pair flips +.056 [+.020, +.092] on OR, +.030 [-.014, +.074] on XOR, +.027 [-.006, +.060] on majority; annealing +.045 [-.005, +.095] on OR, +.027 on XOR, +.002 on majority (all intervals include zero except OR with pair flips). Pair flips cost about 3.5 times the training time (16.4 ms against 4.6 ms for the 16-start search). The honest reading is a small, consistent gain from pair flips that is clearly above noise only on OR, and no demonstrated gain from annealing at this temperature. Neither closes the gap to the MLP. On OR the baseline greedy model is only about six points above the constant predictor.

## CHANGE/USED greedy comparison

[`run_2026-10-02-not-used-equal/benchmark.csv`](run_2026-10-02-not-used-equal/benchmark.csv) compares the joint `F = CHANGE AND USED` proposal with four global integer scores: `10 * error_bits`, `10 * error_bits + used_bits`, `10 * error_bits + unused_bits`, and `error_bits + unused_bits`. Here `unused_bits = number_of_examples * number_of_MUXes - used_bits`; `NOT(USED)` is restricted to valid MUX state bits. The original single-bit greedy remains a separate control. Reproduce all 20 matched splits with `bash scripts/reproduce_generalization.sh results/local-generalization`, then run `python3 scripts/summarize_generalization.py results/local-generalization/benchmark.csv`. The run also includes ANNs with 4 and 16 tanh hidden units. All models get the same train, validation, and test examples; the BFL selected rows choose among 16 topologies using validation data, while each ANN uses one initialization.

Mean **test accuracy** over 20 seeds (the full CSV also includes train and validation accuracy, per-seed errors, training time, and USED):

| Task | Original greedy | Joint mask, error only | Joint mask, + USED | Joint mask, 10·error + NOT(USED) | Joint mask, error + NOT(USED) | ANN 4 / 16 |
|---|---:|---:|---:|---:|---:|---:|
| OR of two bits | .762 | .749 | .732 | .752 | .723 | 1.000 |
| XOR of two bits | .608 | .598 | .601 | .602 | .580 | 1.000 |
| Majority of three bits | .733 | .695 | .689 | .701 | .690 | 1.000 |

With error weight 10, `NOT(USED)` improves test accuracy over `USED` by .020 on OR, .001 on XOR, and .012 on majority. Removing the factor of 10 lowers test accuracy by .029, .022, and .011 respectively. The equal-weight objective is equivalent, up to a constant, to `error_bits - used_bits`, so it can accept more error bits when the gain in active MUXes is larger. Mean test USED remains near 14–15 active MUXes per example for every BFL variant. These experiments do not establish a general generalization benefit from the usage terms. Flipping a complete dependency path is also a materially different proposal from flipping one selector at a time; the comparison with the original greedy does not isolate the penalty alone. The training objective is evaluated only on training examples, and the test set is never used for fitting or selection.

### Selecting the error weight X

[`run_2026-10-02-weight-search/validation_sweep.csv`](run_2026-10-02-weight-search/validation_sweep.csv) evaluates `J = X * error_bits + unused_bits` for every integer `X` from 1 through 128, plus 256, 512, 1024, and 25000. For each weight, task, and seed, 16 MUX topologies are trained on the same training split and one is chosen by validation error. [`weight_ranking.csv`](run_2026-10-02-weight-search/weight_ranking.csv) pools the 60 validation results per weight (3840 validation predictions). **X=17** has the fewest validation errors: 977, or 74.557% accuracy. X=8 and X=9 each have 978 errors. Ties are resolved in favor of the smaller X. The search does not inspect test targets.

After selecting X=17, [`benchmark.csv`](run_2026-10-02-weight-search/benchmark.csv) evaluates it on the held-out test splits. Pooled test accuracy over 60 task/seed combinations is 68.88%, versus 68.49% for X=10 and 66.43% for X=1. The original single-bit greedy reaches 70.10%, and both ANNs reach 100%. For X=17, test accuracy is 75.2% on OR, 61.3% on XOR, and 70.2% on majority. The one-error validation lead over X=8 and X=9 is small, so 17 is the best **within this search protocol**, not a universal optimum. Reproduce the search with `bash scripts/search_error_weight.sh results/local-weight-search`. When `penalize_unused` is enabled without an explicit `error_weight`, the library now uses 17; an explicit weight still overrides it.

## Binary global vs ANN

Run `bash scripts/reproduce_bfl_vs_ann.sh results 50` to compile with `-O3 -march=native`, run all three test binaries, write [`bfl_vs_ann.csv`](bfl_vs_ann.csv), and regenerate [`summary.txt`](summary.txt). The compiler, CPU, flags, and run date are in [`environment_bfl_vs_ann.txt`](environment_bfl_vs_ann.txt). This run used 50 seeds, each with all 1,024 distinct 10-bit inputs shuffled and split into 600 training and 424 test inputs. No input appears in both sets. The target is the standalone benchmark's binary threshold over a constant, ten input bits, and five pairwise AND features. All models receive the same raw inputs, labels, and split.

`bfl_old` is the existing one-bit `GreedyModel` with two hidden MUX layers of width 16, at most 30 epochs and 64 candidates per epoch. The new `bfl_binary_global` row now uses the standalone benchmark's `RoutedBinaryModel`: 32 routes keyed by the first five input bits, 16 signed binary features, and candidate masks of zero to three bits. It evaluates `(ERR, FORGET, CHANGE_COST)` over the current route's entire training memory and applies `S' = S XOR (CHANGE AND USED)` for 20 epochs. The ANN has 16 tanh hidden units and is trained with sigmoid/BCE and full-batch Adam for 500 epochs. The two BFL rows have different model architectures, so their accuracy difference cannot isolate the greedy rule alone. The routed model has hand-designed features that mirror the target function; the ANN receives only the ten raw bits. No validation-based topology selection is used.

Means across 50 matched seeds (population standard deviations and min/median/max for each metric are in `summary.txt`):

| Model | Train accuracy | Test accuracy | Gap | Persistent model | Training temporary | Inference temporary | Training | Inference |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BFL old MUX | 62.38% | 59.92% | 2.46 pp | 324 B | 2,224 B | 33 B | 0.124 ms | 90.55 ns |
| BFL routed binary global | 96.01% | 90.31% | 5.71 pp | 256 B | 27,649 B | 0 B | 194.42 ms | 9.70 ns |
| ANN 16 | 100.00% | 99.48% | 0.52 pp | 1,544 B | 4,760 B | 128 B | 142.46 ms | 385.66 ns |

For routed binary-global relative to the ANN, matched-seed means show 83.42% less persistent model storage and 39.75× faster inference, with 9.17 percentage points lower test accuracy. Its 194.42 ms mean training time is 1.36× the ANN's 142.46 ms, and its temporary training memory is 5.81× larger. Across seeds, routed BFL test accuracy ranges from 85.61% to 94.34% (SD 1.94 points), so the 90% threshold is a mean result, not a per-seed guarantee.

The library-backed reproduction of the [standalone experiment](../benchmarks/standalone_routed_binary_andu.cpp) is in [`routed_binary_andu_12345.txt`](routed_binary_andu_12345.txt): with `AND U`, train accuracy is 96.6667%, test accuracy 90.5660%, 48 updates, 86 effective flipped bits, and 26 forget events. The experimental no-AND-U control gives 90.3302% test accuracy on that seed. Both match the standalone experiment. Across 50 seeds, routed binary-global evaluated 8,364,000 proposals per seed, accepted a mean 49.54 updates, performed a mean 329.38 million popcount calls, changed 95.10 effective bits, and recorded 25.74 forget events. ERR chose a better proposal 81.88 times per seed; FORGET resolved an ERR tie 4.90 times; CHANGE_COST resolved no remaining ties. Candidate evaluation includes the no-op proposal at each training step.

Inference used 10,000 warmup calls and five measured batches of 200,000 calls per model and seed, with a checksum to retain the predictions. The CSV gives the mean and median batch time per inference and the highest of the five batch times (its p95 estimate); it is not a per-call latency percentile. Timing excludes dataset generation, I/O, and model construction. The routed model's 256 bytes are its 32 64-bit route states; fixed feature/routing code and object metadata are excluded. Old MUX bytes include 264 packed node bytes plus the 60-byte `.bflg` header. ANN bytes count 1,544 double-precision weights and biases. Temporary byte counts exclude datasets, allocator overhead, and executable code. The routed training estimate includes route-grouped samples, shuffle order, candidate masks, and a before-error vector. The ANN estimate includes Adam moments, gradient, and activations.

Linux `perf_event_open` was attempted for instructions, cycles, branches, branch misses, and cache misses. This host has `perf_event_paranoid=4`, so those CSV fields are `NaN`; instruction and cycle ratios are unavailable. This experiment tests one target built from the routed model's own fixed features. Its high accuracy therefore measures this feature and routing design on a favorable task, not a general advantage over an ANN. The ANN has higher held-out accuracy here, while routed BFL has a smaller stored state and cheaper inference. Training time and temporary memory favor the ANN in this implementation.
