"""Derive the response-probability threshold from the training cohort only (Youden index of the
ensemble probabilities) and compare it with the frozen value in icrnet/config.py.

    python scripts/derive_threshold.py --predictions predictions_fusion.csv
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icrnet import config  # noqa: E402
from icrnet.metrics import youden_threshold  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True)
    args = ap.parse_args()

    train = pd.read_csv(args.predictions).query("cohort == 'training'")
    thr = youden_threshold(train["label"], train["prob"])
    print(f"training-cohort Youden threshold: {thr:.6f} (n = {len(train)})")
    print(f"frozen threshold in config:       {config.THRESHOLD:.6f}")


if __name__ == "__main__":
    main()
