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
