"""Discrimination, fixed-threshold classification and Brier scores of ICR-Net and the unimodal models
in each of the four cohorts (training, internal test, external test, prospective test), with paired DeLong tests of
each unimodal model against ICR-Net (Table 2). The test cohorts are evaluated once with the fixed model and threshold.

ICR-Net uses the frozen threshold; each unimodal model uses the same rule (training-cohort Youden index)
applied to its own probabilities. With --check, the results are compared with reference/table2_reported.json.

    python scripts/evaluate.py --fusion predictions_fusion.csv --ct predictions_ct.csv --wsi predictions_wsi.csv --out table2.csv --check
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icrnet import config  # noqa: E402
from icrnet.data import COHORTS  # noqa: E402
from icrnet.metrics import auc_delong, brier, delong_paired_test, threshold_metrics, youden_threshold  # noqa: E402

REFERENCE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference", "table2_reported.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for m in ("fusion", "ct", "wsi"):
        ap.add_argument(f"--{m}", required=True, help=f"predictions CSV from scripts/predict.py --modality {m}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true", help="compare with the values reported in the manuscript")
    args = ap.parse_args()

    preds = {m: pd.read_csv(getattr(args, m), dtype={"patient_id": str}).set_index("patient_id") for m in ("fusion", "ct", "wsi")}
    thresholds = {"fusion": config.THRESHOLD}
    for m in ("ct", "wsi"):
        tr = preds[m].query("cohort == 'training'")
        thresholds[m] = youden_threshold(tr["label"], tr["prob"])

    rows = []
    for cohort in COHORTS:
        fus = preds["fusion"].query("cohort == @cohort")
        for m in ("ct", "wsi", "fusion"):
            p = preds[m].loc[fus.index, "prob"]
            auc, lo, hi = auc_delong(fus["label"], p)
            row = {"cohort": cohort, "model": m, "n": len(fus), "threshold": thresholds[m],
                   "auc": auc, "auc_lo": lo, "auc_hi": hi, "brier": brier(fus["label"], p),
                   **threshold_metrics(fus["label"], p, thresholds[m])}
            if m != "fusion":
                row["delong_p_vs_fusion"] = delong_paired_test(fus["label"], p, fus["prob"])
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False)
    print(table.round(3).to_string(index=False))

    if args.check:
        with open(REFERENCE, encoding="utf-8") as f:
            ref = json.load(f)
        mismatches = 0
        for row in rows:
            want = ref[row["cohort"]][row["model"]]
            for key in ("auc", "brier"):
                if round(row[key], 3) != want[key]:
                    mismatches += 1
                    print(f"[mismatch] {row['cohort']} {row['model']} {key}: reproduced {row[key]:.3f}, reported {want[key]:.3f}")
            for key in ("tp", "fn", "tn", "fp"):
                if row[key] != want[key]:
                    mismatches += 1
                    print(f"[mismatch] {row['cohort']} {row['model']} {key}: reproduced {row[key]}, reported {want[key]}")
        print("all reported Table 2 values reproduced" if mismatches == 0 else f"{mismatches} values differ from Table 2")


if __name__ == "__main__":
    main()
