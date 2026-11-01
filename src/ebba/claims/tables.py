"""Transcribed tables from the paper (Tables 3-7) and simple row views.

Every number here was copied from the extracted PDF text and is checked by
``tests/test_claims.py`` for row counts and value ranges.  The transcription is
the single source of truth for the arithmetic audits in ``audit.py``; if a
number looks wrong, the verdict must change, never the table.
"""

from __future__ import annotations

# Table 3: poisoned-image detection, PDR / F1.
# Method order: FTD, TeCo, SCALE, EBBA+.
TABLE3 = {
    ("Cifar10", "BadNets"): {"FTD": (0.97, 0.94), "TeCo": (0.94, 0.91), "SCALE": (0.97, 0.93), "EBBA+": (0.96, 0.96)},
    ("Cifar10", "Blend"):   {"FTD": (0.95, 0.89), "TeCo": (0.93, 0.95), "SCALE": (0.89, 0.87), "EBBA+": (0.95, 0.94)},
    ("Cifar10", "WaNet"):   {"FTD": (0.52, 0.54), "TeCo": (0.93, 0.92), "SCALE": (0.92, 0.90), "EBBA+": (0.95, 0.94)},
    ("Cifar10", "FIBA"):    {"FTD": (0.58, 0.52), "TeCo": (0.96, 0.94), "SCALE": (0.88, 0.87), "EBBA+": (0.95, 0.95)},
    ("Cifar10", "DUBA"):    {"FTD": (0.53, 0.53), "TeCo": (0.95, 0.94), "SCALE": (0.82, 0.79), "EBBA+": (0.96, 0.95)},
    ("GTSRB", "BadNets"):   {"FTD": (0.96, 0.93), "TeCo": (0.92, 0.88), "SCALE": (0.98, 0.96), "EBBA+": (0.98, 0.97)},
    ("GTSRB", "Blend"):     {"FTD": (0.95, 0.96), "TeCo": (0.93, 0.91), "SCALE": (0.89, 0.88), "EBBA+": (0.97, 0.96)},
    ("GTSRB", "WaNet"):     {"FTD": (0.54, 0.58), "TeCo": (0.97, 0.95), "SCALE": (0.91, 0.89), "EBBA+": (0.96, 0.95)},
    ("GTSRB", "FIBA"):      {"FTD": (0.50, 0.58), "TeCo": (0.93, 0.91), "SCALE": (0.89, 0.87), "EBBA+": (0.95, 0.95)},
    ("GTSRB", "DUBA"):      {"FTD": (0.54, 0.58), "TeCo": (0.90, 0.93), "SCALE": (0.89, 0.86), "EBBA+": (0.96, 0.95)},
    ("ImageNet", "BadNets"): {"FTD": (0.90, 0.93), "TeCo": (0.91, 0.92), "SCALE": (0.98, 0.96), "EBBA+": (0.98, 0.95)},
    ("ImageNet", "Blend"):   {"FTD": (0.95, 0.93), "TeCo": (0.93, 0.92), "SCALE": (0.88, 0.86), "EBBA+": (0.98, 0.95)},
    ("ImageNet", "WaNet"):   {"FTD": (0.64, 0.60), "TeCo": (0.94, 0.92), "SCALE": (0.87, 0.86), "EBBA+": (0.96, 0.94)},
    ("ImageNet", "FIBA"):    {"FTD": (0.57, 0.52), "TeCo": (0.94, 0.92), "SCALE": (0.93, 0.92), "EBBA+": (0.96, 0.95)},
    ("ImageNet", "DUBA"):    {"FTD": (0.52, 0.55), "TeCo": (0.93, 0.90), "SCALE": (0.89, 0.91), "EBBA+": (0.96, 0.95)},
}

TABLE3_METHODS = ("FTD", "TeCo", "SCALE", "EBBA+")

# Table 4: backdoor removal, BA (%) / ASR (%).
# Order: No Defense, Fine-Pruning*, NAD*, ANP*, RNP*, MEDIC*, ZIP, EBBA+.
TABLE4 = {
    ("Cifar10", "BadNets"): {"No Defense": (91.22, 99.36), "Fine-Pruning": (86.79, 15.76), "NAD": (88.96, 4.72), "ANP": (87.08, 3.42), "RNP": (89.72, 1.46), "MEDIC": (88.09, 3.76), "ZIP": (88.02, 5.53), "EBBA+": (89.92, 2.92)},
    ("Cifar10", "Blend"):   {"No Defense": (91.35, 99.97), "Fine-Pruning": (85.42, 17.92), "NAD": (87.36, 5.92), "ANP": (89.59, 1.59), "RNP": (90.06, 3.72), "MEDIC": (86.27, 6.79), "ZIP": (83.67, 7.75), "EBBA+": (89.72, 5.79)},
    ("Cifar10", "WaNet"):   {"No Defense": (91.25, 99.78), "Fine-Pruning": (86.92, 79.63), "NAD": (86.37, 39.36), "ANP": (87.62, 19.92), "RNP": (89.30, 12.72), "MEDIC": (87.91, 9.66), "ZIP": (86.25, 3.96), "EBBA+": (89.96, 4.36)},
    ("Cifar10", "FIBA"):    {"No Defense": (91.08, 99.26), "Fine-Pruning": (86.59, 59.07), "NAD": (86.71, 39.79), "ANP": (88.96, 16.29), "RNP": (89.75, 9.91), "MEDIC": (87.09, 8.78), "ZIP": (85.08, 2.85), "EBBA+": (90.02, 7.88)},
    ("Cifar10", "DUBA"):    {"No Defense": (91.55, 99.98), "Fine-Pruning": (85.97, 82.96), "NAD": (89.32, 62.58), "ANP": (81.39, 25.97), "RNP": (90.95, 11.80), "MEDIC": (85.46, 6.27), "ZIP": (85.05, 4.09), "EBBA+": (89.52, 5.24)},
    ("GTSRB", "BadNets"):   {"No Defense": (99.14, 99.62), "Fine-Pruning": (94.72, 7.09), "NAD": (92.69, 2.92), "ANP": (95.16, 2.39), "RNP": (97.82, 0.72), "MEDIC": (94.58, 1.99), "ZIP": (96.92, 6.19), "EBBA+": (99.06, 1.79)},
    ("GTSRB", "Blend"):     {"No Defense": (99.15, 99.72), "Fine-Pruning": (96.93, 42.39), "NAD": (92.61, 10.29), "ANP": (97.57, 5.92), "RNP": (97.02, 2.36), "MEDIC": (94.64, 8.42), "ZIP": (97.01, 8.53), "EBBA+": (98.92, 2.72)},
    ("GTSRB", "WaNet"):     {"No Defense": (99.07, 99.81), "Fine-Pruning": (92.42, 21.97), "NAD": (96.02, 8.39), "ANP": (96.47, 1.92), "RNP": (97.09, 2.09), "MEDIC": (95.79, 10.36), "ZIP": (97.50, 3.29), "EBBA+": (98.79, 1.37)},
    ("GTSRB", "FIBA"):      {"No Defense": (99.22, 98.91), "Fine-Pruning": (93.55, 90.72), "NAD": (95.98, 14.69), "ANP": (96.52, 4.93), "RNP": (96.46, 1.93), "MEDIC": (96.21, 9.57), "ZIP": (96.68, 1.92), "EBBA+": (98.36, 1.47)},
    ("GTSRB", "DUBA"):      {"No Defense": (99.21, 99.92), "Fine-Pruning": (97.83, 94.31), "NAD": (94.19, 27.08), "ANP": (95.14, 15.12), "RNP": (93.69, 7.84), "MEDIC": (95.92, 14.82), "ZIP": (96.84, 3.33), "EBBA+": (98.94, 3.72)},
    ("ImageNet", "BadNets"): {"No Defense": (88.56, 99.22), "Fine-Pruning": (79.62, 2.39), "NAD": (85.72, 9.36), "ANP": (85.21, 11.63), "RNP": (85.92, 6.83), "MEDIC": (86.05, 2.28), "ZIP": (87.09, 7.55), "EBBA+": (88.19, 1.51)},
    ("ImageNet", "Blend"):   {"No Defense": (88.39, 98.62), "Fine-Pruning": (72.97, 7.92), "NAD": (85.93, 11.92), "ANP": (84.27, 6.34), "RNP": (86.92, 5.04), "MEDIC": (84.65, 8.09), "ZIP": (86.08, 8.35), "EBBA+": (88.36, 1.32)},
    ("ImageNet", "WaNet"):   {"No Defense": (88.62, 99.27), "Fine-Pruning": (62.91, 84.27), "NAD": (85.62, 14.96), "ANP": (85.49, 12.35), "RNP": (86.56, 9.26), "MEDIC": (86.15, 10.97), "ZIP": (86.90, 3.02), "EBBA+": (87.94, 4.76)},
    ("ImageNet", "FIBA"):    {"No Defense": (89.07, 98.59), "Fine-Pruning": (82.17, 79.62), "NAD": (84.92, 11.15), "ANP": (86.39, 6.11), "RNP": (87.07, 2.30), "MEDIC": (86.82, 3.09), "ZIP": (87.15, 2.92), "EBBA+": (88.09, 1.92)},
    ("ImageNet", "DUBA"):    {"No Defense": (88.92, 99.36), "Fine-Pruning": (81.91, 92.47), "NAD": (86.39, 35.72), "ANP": (85.51, 13.94), "RNP": (85.35, 10.09), "MEDIC": (84.99, 6.90), "ZIP": (86.69, 2.06), "EBBA+": (87.91, 2.96)},
}

TABLE4_METHODS = ("No Defense", "Fine-Pruning", "NAD", "ANP", "RNP", "MEDIC", "ZIP", "EBBA+")

# Table 5 (p.9) and Table 7 (p.24) print identical numbers.
TABLE5 = {"HC": 0.972, "Birch": 0.968, "Mean Shift": 0.965, "DBSCAN": 0.975}
TABLE7 = {"HC": 0.972, "Birch": 0.968, "Mean Shift": 0.965, "DBSCAN": 0.975}

# Table 6: dataset sizes printed in the paper.
TABLE6 = {
    "Cifar10": {"train": 50000, "test": 10000, "labels": 10, "image": "32x32x3"},
    "Gtsrb": {"train": 39209, "test": 12603, "labels": 43, "image": "64x64x3"},
    "ImageNet": {"train": 48000, "test": 12000, "labels": 100, "image": "224x224x3"},
}

# Standard public test sizes, used only to check Table 6.
STANDARD_TEST_SIZES = {"Cifar10": 10000, "Gtsrb": 12630, "ImageNet_subset": 12000}

NUM_CLASSES_BY_DATASET = {"Cifar10": 10, "GTSRB": 43, "ImageNet": 100}


def table3_rows() -> list[dict]:
    rows = []
    for (dataset, attack), values in TABLE3.items():
        row = {"dataset": dataset, "attack": attack}
        for method in TABLE3_METHODS:
            row[f"{method}_pdr"], row[f"{method}_f1"] = values[method]
        rows.append(row)
    return rows


def table4_rows() -> list[dict]:
    rows = []
    for (dataset, attack), values in TABLE4.items():
        row = {"dataset": dataset, "attack": attack}
        for method in TABLE4_METHODS:
            row[f"{method}_ba"], row[f"{method}_asr"] = values[method]
        rows.append(row)
    return rows
