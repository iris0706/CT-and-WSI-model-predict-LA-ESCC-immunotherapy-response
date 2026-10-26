"""Tile H&E biopsy whole-slide images and encode every tissue patch with a frozen patch encoder.

Default pipeline (Methods, Image preprocessing): non-overlapping 224 x 224-pixel patches at 0.5 um per pixel
(20x equivalent; the slide resolution is read from the scanner metadata), predominantly blank patches excluded,
each patch encoded into a 1,536-dimensional embedding by the pathology foundation model UNI (UNI2-h release),
giving an (n, 1536) matrix per slide. The encoder is frozen; only the model on top of the embeddings is trained.

Earlier pipeline (for weights trained with it): --encoder resnet50_l3 --mpp 0.25 --patch 256
(ImageNet ResNet-50 truncated after layer3, 1,024-dimensional features).

Input CSV columns: patient_id, slide_path [, slide_mpp  (um per pixel at level 0, overrides the slide metadata)].
Output: <out_dir>/<patient_id>.npy (float32, n x dim) and <patient_id>_coords.csv (level-0 x, y);
existing outputs are skipped unless --overwrite is given, so an interrupted run can be resumed.

    huggingface-cli login            # once, after accepting the UNI2-h terms on the model page
    python scripts/extract_wsi_features.py --input wsi_list.csv --out-dir data/wsi_features --device cuda
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icrnet import config  # noqa: E402
from icrnet.wsi import ENCODERS, PatchEncoder, iter_patches, slide_mpp  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--encoder", default=config.WSI_ENCODER, choices=sorted(ENCODERS))
    ap.add_argument("--mpp", type=float, default=config.WSI_TARGET_MPP, help="target resolution in um per pixel")
    ap.add_argument("--patch", type=int, default=config.WSI_PATCH_SIZE, help="patch size in pixels at the target resolution")
    ap.add_argument("--slide-mpp", type=float, default=None, help="um per pixel of every slide when the metadata lack it")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-fp16", action="store_true", help="full precision (fp16 is used on CUDA by default)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true", help="random encoder weights (testing only)")
    args = ap.parse_args()

    import openslide
    os.makedirs(args.out_dir, exist_ok=True)
    fp16 = args.device.startswith("cuda") and not args.no_fp16
    encoder = PatchEncoder(args.encoder, pretrained=not args.no_pretrained, fp16=fp16).to(args.device)
    print(f"encoder {args.encoder} ({encoder.dim}-d) | {args.mpp} um/px, {args.patch} px | device {args.device} | fp16 {fp16}")
    rows = pd.read_csv(args.input, dtype={"patient_id": str})
    failed = []
    for i, row in enumerate(rows.itertuples(), 1):
        out = os.path.join(args.out_dir, f"{row.patient_id}.npy")
        if os.path.exists(out) and not args.overwrite:
            continue
        t0 = time.time()
        try:
            slide = openslide.open_slide(row.slide_path)      # OpenSlide formats; plain images fall back to ImageSlide
            mpp0 = args.slide_mpp or (float(row.slide_mpp) if "slide_mpp" in rows.columns and pd.notna(row.slide_mpp) else None)
            mpp0, src = (mpp0, "input") if mpp0 else slide_mpp(slide)
            feats, coords, buf = [], [], []
            for x, y, img in iter_patches(slide, args.mpp, args.patch, mpp0):
                coords.append((x, y)); buf.append(img)
                if len(buf) == args.batch_size:
                    feats.append(encoder.embed(buf)); buf = []
            if buf:
                feats.append(encoder.embed(buf))
            slide.close()
            if not feats:
                raise RuntimeError("no tissue patches")
            feats = np.concatenate(feats)
            np.save(out, feats)
            pd.DataFrame(coords, columns=["x", "y"]).to_csv(os.path.join(args.out_dir, f"{row.patient_id}_coords.csv"), index=False)
            print(f"[{i}/{len(rows)}] {row.patient_id}: {feats.shape[0]} patches ({mpp0:.3f} um/px from {src}) in {time.time() - t0:.0f}s")
        except Exception as e:                                   # noqa: BLE001
            failed.append({"patient_id": row.patient_id, "slide_path": row.slide_path, "error": repr(e)})
            print(f"[{i}/{len(rows)}] {row.patient_id}: FAILED {e!r}")
    if failed:
        pd.DataFrame(failed).to_csv(os.path.join(args.out_dir, "_failed.csv"), index=False)
        print(f"{len(failed)} slides failed; see _failed.csv")


if __name__ == "__main__":
    main()
