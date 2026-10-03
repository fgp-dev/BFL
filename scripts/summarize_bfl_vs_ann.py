#!/usr/bin/env python3
"""Summarize matched BFL/ANN runs without external dependencies."""
import csv
import math
import statistics
import sys
from collections import defaultdict


def numbers(rows, key):
    return [float(row[key]) for row in rows if row[key] and math.isfinite(float(row[key]))]


def describe(values):
    if not values:
        return "unavailable"
    return (f"mean={statistics.mean(values):.6f}, median={statistics.median(values):.6f}, "
            f"min={min(values):.6f}, max={max(values):.6f}, "
            f"sd={statistics.pstdev(values):.6f}")


def main(path):
    with open(path, newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    groups = defaultdict(list)
    for row in rows:
        groups[row["model"]].append(row)
    seeds = {model: {int(row["seed"]) for row in entries}
             for model, entries in groups.items()}
    if len(groups) != 3 or len(set(map(frozenset, seeds.values()))) != 1:
        raise ValueError("Expected three models on identical seed sets")
    if min(map(len, seeds.values())) < 20:
        raise ValueError("At least 20 matched seeds are required")
    fields = ["train_accuracy", "test_accuracy", "generalization_gap", "model_bytes",
              "parameter_state_bytes", "temporary_training_bytes",
              "temporary_inference_bytes", "training_ms", "training_ns_per_sample",
              "training_ms_per_epoch", "ns_per_inference", "median_ns_per_inference",
              "p95_batch_ns_per_inference", "instructions_per_inference",
              "cycles_per_inference", "branches_per_inference",
              "branch_misses_per_inference", "cache_misses_per_inference",
              "forget_count", "change_cost", "candidate_evaluations",
              "popcount_operations", "accepted_updates", "rejected_candidates",
              "effective_changed_bits", "average_candidate_evaluations_per_update",
              "err_improves", "err_ties_forget_improves",
              "err_forget_ties_change_improves"]
    lines = [f"Task: binary threshold over 16 fixed features; {len(next(iter(seeds.values())))} matched seeds.",
             "Train: 600 unique inputs; test: 424 unique inputs; 10 binary inputs total.",
             "SD is population standard deviation across seeds. NaN means unavailable.",
             "Inference mean/median/p95 are across five batches of 200,000 calls per seed.", ""]
    for model in ("bfl_old", "bfl_binary_global", "ann_16"):
        lines.append(model)
        for field in fields:
            lines.append(f"  {field}: {describe(numbers(groups[model], field))}")
        lines.append("")
    ann = groups["ann_16"]
    for model in ("bfl_old", "bfl_binary_global"):
        bfl = groups[model]
        lines.append(f"{model} relative to ann_16 (ratios of per-seed matched values)")
        for field, label, ratio in [
            ("model_bytes", "memory reduction", lambda a, b: 1 - b / a),
            ("ns_per_inference", "inference speedup", lambda a, b: a / b),
            ("test_accuracy", "test accuracy difference (BFL - ANN)", lambda a, b: b - a),
            ("instructions_per_inference", "instruction reduction", lambda a, b: 1 - b / a),
        ]:
            matched = []
            for a, b in zip(ann, bfl):
                av, bv = float(a[field]), float(b[field])
                if math.isfinite(av) and math.isfinite(bv) and av and bv:
                    matched.append(ratio(av, bv))
            lines.append(f"  {label}: {describe(matched)}")
        lines.append("")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
