#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ancestry probe on latent representations.")
    p.add_argument("--run_dir", required=True, help="Training run directory.")
    p.add_argument("--metadata_tsv", required=True, help="Metadata TSV/ANNO path.")
    p.add_argument("--target_col", default="Political Entity", help="Metadata target column.")
    p.add_argument("--seed", type=int, default=42, help="Split seed.")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    return p.parse_args()


def _is_missing(value: str) -> bool:
    v = value.strip()
    if not v:
        return True
    return v.lower() in {"na", "nan", "none", "null", "..", "n/a"}


def _load_sample_ids_from_csv(global_latents_csv: Path) -> np.ndarray:
    sample_ids = []
    with global_latents_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "sample_id" not in reader.fieldnames:
            raise ValueError(f"{global_latents_csv} is missing a sample_id column")
        for row in reader:
            sample_ids.append(row["sample_id"])
    return np.asarray(sample_ids, dtype=object)


def _load_target_map(metadata_tsv: Path, target_col: str) -> Dict[str, str]:
    with metadata_tsv.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or len(reader.fieldnames) == 0:
            raise ValueError(f"No header found in metadata file: {metadata_tsv}")
        id_col = reader.fieldnames[0]
        if target_col not in reader.fieldnames:
            raise ValueError(f"Target column '{target_col}' not found in metadata header")

        out: Dict[str, str] = {}
        for row in reader:
            sid = (row.get(id_col) or "").strip()
            target = (row.get(target_col) or "").strip()
            if not sid or _is_missing(target):
                continue
            out[sid] = target
    return out


def _macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    eps = 1e-12
    f1_vals = []
    for c in range(n_classes):
        tp = float(np.sum((y_true == c) & (y_pred == c)))
        fp = float(np.sum((y_true != c) & (y_pred == c)))
        fn = float(np.sum((y_true == c) & (y_pred != c)))
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        f1_vals.append(f1)
    return float(np.mean(f1_vals))


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float((y_true == y_pred).mean())


def _stratified_split(
    y: np.ndarray,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    test_idx = []

    classes = np.unique(y)
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = idx.size
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        if n_train < 1:
            n_train = 1
        if n_val < 1:
            n_val = 1
        while n_train + n_val >= n and (n_train > 1 or n_val > 1):
            if n_val > 1:
                n_val -= 1
            else:
                n_train -= 1
        n_test = n - n_train - n_val
        if n_test < 1:
            # Skip tiny classes that cannot support train/val/test.
            continue
        train_idx.append(idx[:n_train])
        val_idx.append(idx[n_train : n_train + n_val])
        test_idx.append(idx[n_train + n_val :])

    if not train_idx or not val_idx or not test_idx:
        raise ValueError("Unable to build non-empty stratified splits")

    train = np.concatenate(train_idx)
    val = np.concatenate(val_idx)
    test = np.concatenate(test_idx)
    return train, val, test


def _evaluate_with_sklearn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> Tuple[float, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1200,
        random_state=seed,
    )
    clf.fit(x_train_scaled, y_train)
    y_pred = clf.predict(x_test_scaled)
    return _macro_f1_score(y_test, y_pred, n_classes=len(np.unique(y_train))), _accuracy(y_test, y_pred)


def _evaluate_with_torch(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> Tuple[float, float]:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x_train_std = (x_train - mean) / std
    x_test_std = (x_test - mean) / std

    x_train_t = torch.from_numpy(x_train_std.astype(np.float32))
    y_train_t = torch.from_numpy(y_train.astype(np.int64))
    x_test_t = torch.from_numpy(x_test_std.astype(np.float32))

    n_features = x_train_t.shape[1]
    n_classes = int(np.max(y_train)) + 1
    model = torch.nn.Linear(n_features, n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-2, weight_decay=1e-3)

    for _ in range(300):
        logits = model(x_train_t)
        loss = F.cross_entropy(logits, y_train_t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        y_pred = model(x_test_t).argmax(dim=1).cpu().numpy()
    return _macro_f1_score(y_test, y_pred, n_classes=n_classes), _accuracy(y_test, y_pred)


def _evaluate_representation(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
) -> Dict[str, float]:
    # Probe uses fixed train split; val exists for protocol symmetry and reporting.
    n_val = int(val_idx.size)
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]

    try:
        macro_f1, acc = _evaluate_with_sklearn(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            seed=seed,
        )
        backend = "sklearn"
    except Exception:
        macro_f1, acc = _evaluate_with_torch(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            seed=seed,
        )
        backend = "torch_fallback"

    return {
        "macro_f1": float(macro_f1),
        "accuracy": float(acc),
        "backend": backend,
        "n_train": int(train_idx.size),
        "n_val": n_val,
        "n_test": int(test_idx.size),
        "n_classes": int(np.unique(y).size),
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    metadata_tsv = Path(args.metadata_tsv)

    global_latents_csv = run_dir / "global_latents.csv"
    global_latents_npy = run_dir / "global_latents.npy"
    global_latent_tokens_npy = run_dir / "global_latent_tokens.npy"
    global_latents_concat_npy = run_dir / "global_latents_concat.npy"

    for path in (
        global_latents_csv,
        global_latents_npy,
        global_latent_tokens_npy,
        global_latents_concat_npy,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required probe input missing: {path}")
    if not metadata_tsv.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_tsv}")

    sample_ids = _load_sample_ids_from_csv(global_latents_csv)
    x_mean = np.load(global_latents_npy)
    x_tokens = np.load(global_latent_tokens_npy)
    x_concat = np.load(global_latents_concat_npy)

    if not (len(sample_ids) == x_mean.shape[0] == x_tokens.shape[0] == x_concat.shape[0]):
        raise ValueError("Sample count mismatch across latent artifacts")

    target_map = _load_target_map(metadata_tsv=metadata_tsv, target_col=args.target_col)
    matched_idx = []
    matched_labels = []
    for i, sid in enumerate(sample_ids.tolist()):
        label = target_map.get(str(sid))
        if label is None:
            continue
        matched_idx.append(i)
        matched_labels.append(label)

    if len(matched_idx) == 0:
        raise ValueError("No overlap between run sample IDs and metadata labels")

    matched_idx_arr = np.asarray(matched_idx, dtype=np.int64)
    labels_arr = np.asarray(matched_labels, dtype=object)

    uniq, counts = np.unique(labels_arr, return_counts=True)
    keep = uniq[counts >= 3]
    keep_set = set(keep.tolist())
    keep_mask = np.asarray([lbl in keep_set for lbl in labels_arr], dtype=bool)
    matched_idx_arr = matched_idx_arr[keep_mask]
    labels_arr = labels_arr[keep_mask]

    if np.unique(labels_arr).size < 2:
        raise ValueError("Need at least 2 classes with >=3 samples each for probe evaluation")

    classes = sorted(np.unique(labels_arr).tolist())
    class_to_id = {c: i for i, c in enumerate(classes)}
    y = np.asarray([class_to_id[c] for c in labels_arr], dtype=np.int64)

    x_mean = x_mean[matched_idx_arr]
    x_tokens = x_tokens[matched_idx_arr]
    x_concat = x_concat[matched_idx_arr]

    train_idx, val_idx, test_idx = _stratified_split(
        y=y,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    metrics: Dict[str, Dict[str, float]] = {}
    metrics["mean"] = _evaluate_representation(x_mean, y, train_idx, val_idx, test_idx, seed=args.seed)
    metrics["concat"] = _evaluate_representation(x_concat, y, train_idx, val_idx, test_idx, seed=args.seed)
    for tok_i in range(x_tokens.shape[1]):
        metrics[f"token_{tok_i}"] = _evaluate_representation(
            x_tokens[:, tok_i, :],
            y,
            train_idx,
            val_idx,
            test_idx,
            seed=args.seed,
        )

    summary = {
        "target_col": args.target_col,
        "seed": int(args.seed),
        "n_samples_used": int(y.size),
        "n_classes": int(len(classes)),
        "classes": classes,
        "split_sizes": {
            "train": int(train_idx.size),
            "val": int(val_idx.size),
            "test": int(test_idx.size),
        },
        "metrics": metrics,
    }

    probe_summary_path = run_dir / "probe_summary.json"
    with probe_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    probe_tsv_path = run_dir / "probe_metrics.tsv"
    with probe_tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "representation",
                "macro_f1",
                "accuracy",
                "backend",
                "n_train",
                "n_val",
                "n_test",
                "n_classes",
            ]
        )
        for rep_name, vals in metrics.items():
            writer.writerow(
                [
                    rep_name,
                    vals.get("macro_f1"),
                    vals.get("accuracy"),
                    vals.get("backend"),
                    vals.get("n_train"),
                    vals.get("n_val"),
                    vals.get("n_test"),
                    vals.get("n_classes"),
                ]
            )

    print(f"Wrote {probe_summary_path}")
    print(f"Wrote {probe_tsv_path}")


if __name__ == "__main__":
    main()
