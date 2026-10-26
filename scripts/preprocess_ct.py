"""Crop the tumor region from pretreatment contrast-enhanced CT and apply the soft-tissue window.

Input CSV columns: patient_id, ct_path (DICOM series directory or NIfTI file), mask_path (ITK-SNAP label).
Output: <out_dir>/<patient_id>.npy, uint8 array (D, H, W) scaled to 0-255.

    python scripts/preprocess_ct.py --input ct_list.csv --out-dir data/ct_npy
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icrnet import config  # noqa: E402


def read_ct(path):
    if os.path.isdir(path):
        reader = sitk.ImageSeriesReader()
        names = reader.GetGDCMSeriesFileNames(path)
        if not names:
            raise FileNotFoundError(f"no DICOM series in {path}")
        reader.SetFileNames(names)
        return reader.Execute()
    return sitk.ReadImage(path)


def align_mask(ct, mask, tol=1e-3):
    """Resample the mask onto the CT grid (nearest neighbour) when size or spacing differ."""
    same = ct.GetSize() == mask.GetSize() and all(abs(a - b) <= tol for a, b in zip(ct.GetSpacing(), mask.GetSpacing()))
    if same:
        return mask
    return sitk.Resample(mask, ct, sitk.Transform(), sitk.sitkNearestNeighbor, 0.0, mask.GetPixelID())


def crop_to_mask(ct_array, mask_array, margin=config.CT_MARGIN):
    idx = np.nonzero(mask_array > 0)
    if len(idx[0]) == 0:
        raise ValueError("empty tumor mask")
    lo = [max(0, int(i.min()) - margin) for i in idx]
    hi = [min(s, int(i.max()) + 1 + margin) for i, s in zip(idx, ct_array.shape)]
    return ct_array[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]


def window(ct_array, width=config.CT_WINDOW_WIDTH, level=config.CT_WINDOW_LEVEL):
    scaled = (ct_array.astype(np.float32) - (level - width / 2)) / width
    return (np.clip(scaled, 0, 1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = pd.read_csv(args.input, dtype={"patient_id": str})
    failed = 0
    for row in rows.itertuples():
        try:
            ct = read_ct(row.ct_path)
            mask = align_mask(ct, sitk.ReadImage(row.mask_path))
            vol = window(crop_to_mask(sitk.GetArrayFromImage(ct), sitk.GetArrayFromImage(mask)))
            np.save(os.path.join(args.out_dir, f"{row.patient_id}.npy"), vol)
            print(f"[ok] {row.patient_id} {vol.shape}")
        except Exception as exc:  # report and continue with the remaining patients
            failed += 1
            print(f"[failed] {row.patient_id}: {exc}")
    print(f"done: {len(rows) - failed} written, {failed} failed")


if __name__ == "__main__":
    main()
