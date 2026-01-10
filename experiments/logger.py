"""Simple CSV logger for experiment metrics."""

import csv
from pathlib import Path


def append_metrics_csv(path: str, row: dict):
    p = Path(path)
    header = sorted(row.keys())
    write_header = not p.exists()
    with p.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
