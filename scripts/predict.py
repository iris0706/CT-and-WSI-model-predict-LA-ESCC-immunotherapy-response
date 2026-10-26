"""Apply a frozen model to every patient in the manifest.

The predicted probability of radiographic response is the softmax output of the model, with the CT volume
resampled to 5 slices; patients with probability >= the frozen threshold are predicted responders.
The patch-embedding dimension (1,536 for UNI2-h, 1,024 for the earlier ResNet-50 pipeline) is read from the
checkpoint; the feature files named in the manifest must come from the same encoder.

    python scripts/predict.py --manifest manifest.csv --modality fusion --checkpoint runs/fusion/fusion.pth --out predictions_fusion.csv
"""
import argparse
import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icrnet import config  # noqa: E402
from icrnet.data import PatientDataset, collate, read_manifest  # noqa: E402
from icrnet.models import checkpoint_wsi_dim, model_from_checkpoint  # noqa: E402


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ids, probs = [], []
    for ct, bags, _, pid in loader:
        logits, _ = model(ct.to(device), [b.to(device) for b in bags])
        probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
        ids.extend(pid)
    return pd.Series(probs, index=ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--modality", choices=("fusion", "ct", "wsi"), required=True)
    ap.add_argument("--checkpoint", required=True, help="weights written by scripts/train.py or the original implementation")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=config.THRESHOLD)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    df = read_manifest(args.manifest)
    state = torch.load(args.checkpoint, map_location=args.device)
    wsi_dim = checkpoint_wsi_dim(state)
    model = model_from_checkpoint(args.modality, state, map_location=args.device).to(args.device)
    print(f"checkpoint {args.checkpoint}: patch embeddings {wsi_dim or 'not used'}" + (f" ({state['wsi_encoder']})" if "wsi_encoder" in state else ""))
    loader = DataLoader(PatientDataset(df, wsi_dim), batch_size=config.INFERENCE_BATCH_SIZE, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate)

    out = df[["patient_id", "center", "cohort", "label"]].set_index("patient_id").join(predict(model, loader, args.device).rename("prob"))
    out["predicted_responder"] = (out["prob"] >= args.threshold).astype(int)
    out.reset_index().to_csv(args.out, index=False)
    print(out.groupby("cohort")["predicted_responder"].agg(n="size", predicted_responders="sum"))


if __name__ == "__main__":
    main()
