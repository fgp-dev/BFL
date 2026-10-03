#!/usr/bin/env python3
# Copyright 2026 Facundo Gomez Prates
# SPDX-License-Identifier: Apache-2.0
"""Select one global error weight using validation results only."""

import csv
import sys
from collections import defaultdict
from pathlib import Path


def main(source_path, output_dir):
    totals = defaultdict(lambda: [0, 0, set()])
    with open(source_path, newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            weight = int(row["weight"])
            key = (row["task"], int(row["seed"]))
            errors, samples, seen = totals[weight]
            if key in seen:
                raise ValueError(f"Duplicate task and seed for weight {weight}: {key}")
            seen.add(key)
            totals[weight][0] = errors + int(row["validation_errors"])
            totals[weight][1] = samples + int(row["validation_samples"])

    expected = {(task, seed) for task in ("or2", "xor2", "majority3")
                for seed in range(1, 21)}
    if not totals or any(values[2] != expected for values in totals.values()):
        raise ValueError("Every weight needs the same 20 splits for all three tasks")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "weight_ranking.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(("weight", "validation_errors", "validation_samples", "validation_accuracy"))
        for weight in sorted(totals):
            errors, samples, _ = totals[weight]
            writer.writerow((weight, errors, samples, f"{1 - errors / samples:.6f}"))

    selected = min(totals, key=lambda weight: (totals[weight][0], weight))
    (output_dir / "selected_weight.txt").write_text(f"{selected}\n", encoding="utf-8")
    errors, samples, _ = totals[selected]
    print(f"Selected X={selected}: validation accuracy {1 - errors / samples:.6f} "
          f"({samples - errors}/{samples})")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: select_error_weight.py validation_sweep.csv output_dir")
    main(sys.argv[1], sys.argv[2])
