#!/usr/bin/env python3
# Copyright 2026 Facundo Gomez Prates
# SPDX-License-Identifier: Apache-2.0
"""Print mean accuracies from the held-out C++ BFL comparison."""

import csv
import sys
from collections import defaultdict
from statistics import mean


def main(path):
    groups = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            groups[(row["task"], row["model"])].append(row)

    print("| Task | Model | Train accuracy | Validation accuracy | Test accuracy | Mean training ms |")
    print("|---|---|---:|---:|---:|---:|")
    for key in sorted(groups):
        rows = groups[key]
        if len(rows) != 20 or len({row["seed"] for row in rows}) != 20:
            raise ValueError(f"Expected 20 unique seeds for {key}")
        train = mean(float(row["train_accuracy"]) for row in rows)
        validation = mean(float(row["validation_accuracy"]) for row in rows)
        test = mean(float(row["test_accuracy"]) for row in rows)
        milliseconds = mean(float(row["training_ms"]) for row in rows)
        print(f"| {key[0]} | {key[1]} | {train:.3f} | {validation:.3f} | {test:.3f} | {milliseconds:.1f} |")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: summarize_generalization.py benchmark.csv")
    main(sys.argv[1])
