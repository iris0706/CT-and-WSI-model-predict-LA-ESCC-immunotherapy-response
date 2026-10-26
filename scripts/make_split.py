"""Split the Center 1 retrospective development cohort into training (80%) and internal test (20%)
by stratified random sampling on the responder:non-responder ratio.

Rows whose cohort is 'development' are reassigned to 'training' or 'internal_test'; all other rows
(external_test, prospective_test) are left unchanged.

    python scripts/make_split.py --manifest manifest_unsplit.csv --out manifest.csv
"""
import argparse
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icrnet import config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest, dtype={"patient_id": str})
    dev = df.index[df["cohort"] == "development"]
    _, val = train_test_split(dev, test_size=config.INTERNAL_TEST_FRACTION,
                              stratify=df.loc[dev, "label"], random_state=config.SPLIT_RANDOM_STATE)
    df.loc[dev, "cohort"] = "training"
    df.loc[val, "cohort"] = "internal_test"
    df.to_csv(args.out, index=False)
    print(df.groupby("cohort")["label"].agg(n="size", responders="sum"))


if __name__ == "__main__":
    main()
