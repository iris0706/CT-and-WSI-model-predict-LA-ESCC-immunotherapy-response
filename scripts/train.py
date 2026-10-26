"""Train ICR-Net or a unimodal model episodically on class-balanced tasks from the training cohort.

Each task draws 5 responders and 5 non-responders as a support set and another 5 of each as a query set.
The model takes 5 SGD steps on the support set (learning rate 1e-3, momentum 0.9, weight decay 1e-3,
unweighted cross-entropy); the query set only monitors performance. An epoch is 50 tasks; the seed is 42.
Only rows with cohort == training are used; the internal, external and prospective test cohorts are never seen
during training or threshold selection. The WSI patch encoder is frozen (its embeddings are precomputed by
scripts/extract_wsi_features.py); the WSI branch trained here is the fully connected layer on top of the embeddings,
sized from the feature files.

The original implementation wrapped these steps in a meta-learning outer loop whose update had no effect
(the outer loss was rebuilt from detached values, and the saved "initial" weights shared memory with the
model), so the procedure below is what the published weights actually received. The number of epochs used
for the published weights was not recorded; the checkpoint is overwritten at the end of every epoch.

    python scripts/train.py --manifest manifest.csv --modality fusion --epochs 1 --out-dir runs/fusion
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icrnet import config  # noqa: E402
from icrnet.data import feature_dim, load_training_pools, read_manifest, to_three_channels  # noqa: E402
from icrnet.models import build_model  # noqa: E402


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_task(pools, rng):
    """Disjoint support and query sets of N_SHOT and N_QUERY patients per class."""
    support, query = [], []
    for label, pool in pools.items():
        idx = rng.permutation(len(pool))[:config.N_SHOT + config.N_QUERY]
        support += [(pool[i], label) for i in idx[:config.N_SHOT]]
        query += [(pool[i], label) for i in idx[config.N_SHOT:]]
    return support, query


def to_batch(items, device):
    ct = to_three_channels(torch.stack([ct for (ct, _), _ in items])).to(device)
    bags = [wsi.to(device) for (_, wsi), _ in items]
    y = torch.tensor([label for _, label in items], dtype=torch.long, device=device)
    return ct, bags, y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--modality", choices=("fusion", "ct", "wsi"), required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tasks-per-epoch", type=int, default=config.TASKS_PER_EPOCH)
    ap.add_argument("--encoder", default=config.WSI_ENCODER, help="name of the patch encoder, recorded in the checkpoint")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-pretrained", action="store_true", help="random initialization (testing only)")
    args = ap.parse_args()

    seed_everything(config.SEED)
    rng = np.random.default_rng(config.SEED)
    os.makedirs(args.out_dir, exist_ok=True)
    df = read_manifest(args.manifest)
    train = df[df["cohort"] == "training"]
    assert len(train) and set(train["cohort"]) == {"training"}, "training uses the training cohort only"
    wsi_dim = feature_dim(train["wsi_features"].iloc[0])
    pools = load_training_pools(train, wsi_dim)
    print(f"training cohort: {len(pools[1])} responders, {len(pools[0])} non-responders | patch embeddings {wsi_dim}-d ({args.encoder})")

    model = build_model(args.modality, pretrained=not args.no_pretrained, wsi_dim=wsi_dim).to(args.device)
    model.train()   # as in the original implementation, the model stays in training mode throughout
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM,
                                weight_decay=config.WEIGHT_DECAY)

    ckpt = os.path.join(args.out_dir, f"{args.modality}.pth")
    log = []
    for epoch in range(1, args.epochs + 1):
        query_loss, query_acc = [], []
        for _ in range(args.tasks_per_epoch):
            support, query = sample_task(pools, rng)
            ct, bags, y = to_batch(support, args.device)
            for _ in range(config.INNER_STEPS):
                logits, _ = model(ct, bags)
                loss = criterion(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            ct, bags, y = to_batch(query, args.device)
            with torch.no_grad():
                logits, _ = model(ct, bags)
                query_loss.append(criterion(logits, y).item())
                query_acc.append((logits.argmax(dim=1) == y).float().mean().item())
        log.append({"epoch": epoch, "query_loss": float(np.mean(query_loss)), "query_accuracy": float(np.mean(query_acc))})
        print(f"epoch {epoch:3d}  query loss {log[-1]['query_loss']:.4f}  query accuracy {log[-1]['query_accuracy']:.4f}")
        torch.save({"model_state_dict": model.state_dict(), "modality": args.modality, "seed": config.SEED, "epoch": epoch,
                    "wsi_dim": wsi_dim, "wsi_encoder": args.encoder}, ckpt)

    with open(os.path.join(args.out_dir, f"{args.modality}_log.json"), "w") as f:
        json.dump({"wsi_dim": wsi_dim, "wsi_encoder": args.encoder, "epochs": log}, f, indent=2)
    print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
