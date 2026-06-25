#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
DEFAULT_ASSIGNMENTS = {
    "latent3_matched": ANALYSIS_DIR
    / "csoge_ho_dropgt90_obs005_country000_origgroup020_globalog_latent3_17439177"
    / "optics_clusters_matched_6d20d_config"
    / "optics_cluster_assignments.tsv",
    "latent6": ANALYSIS_DIR
    / "csoge_ho_dropgt90_obs005_country005_origgroup020_17125764"
    / "optics_clusters_default_sweep"
    / "optics_cluster_assignments.tsv",
    "latent20": ANALYSIS_DIR
    / "csoge_ho_dropgt90_obs005_country000_origgroup020_globalog_latent20_17630186"
    / "optics_clusters_default_sweep"
    / "optics_cluster_assignments.tsv",
}

STATUS_ORDER = [
    "mutual_cluster_match",
    "both_noise",
    "cluster_mismatch",
    "one_noise",
]
STATUS_COLORS = {
    "mutual_cluster_match": "#2a9d8f",
    "both_noise": "#8d99ae",
    "cluster_mismatch": "#e76f51",
    "one_noise": "#f4a261",
}
STATUS_LABELS = {
    "mutual_cluster_match": "Mutual cluster match",
    "both_noise": "Both noise",
    "cluster_mismatch": "Cluster mismatch",
    "one_noise": "One run noise",
}


def _parse_assignment(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Assignment must be LABEL=PATH, got: {value}")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Empty assignment label: {value}")
    return label, Path(path).expanduser()


def _load_assignment(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    required = {"sample_id", "cluster_id", "is_optics_noise", "z1", "z2", "z3"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        duplicated = frame.loc[frame["sample_id"].duplicated(), "sample_id"].head(5).tolist()
        raise ValueError(f"{path} has duplicate sample IDs, examples: {duplicated}")
    out = frame.copy()
    out["sample_id"] = out["sample_id"].astype(str)
    out["cluster_id"] = pd.to_numeric(out["cluster_id"], errors="raise").astype(int)
    if out["is_optics_noise"].dtype != bool:
        out["is_optics_noise"] = out["is_optics_noise"].astype(str).str.lower().isin({"1", "true", "yes"})
    for col in ("z1", "z2", "z3"):
        out[col] = pd.to_numeric(out[col], errors="raise")
    return out


def _mutual_best_pairs(cluster_a: pd.Series, cluster_b: pd.Series) -> set[tuple[int, int]]:
    counts = pd.crosstab(cluster_a, cluster_b)
    if counts.empty:
        return set()
    row_best: Dict[int, int] = {}
    for cluster_id, row in counts.iterrows():
        if int(cluster_id) == -1:
            continue
        non_noise = row[row.index != -1]
        if not non_noise.empty and int(non_noise.max()) > 0:
            row_best[int(cluster_id)] = int(non_noise.idxmax())
    col_best: Dict[int, int] = {}
    for cluster_id, col in counts.items():
        if int(cluster_id) == -1:
            continue
        non_noise = col[col.index != -1]
        if not non_noise.empty and int(non_noise.max()) > 0:
            col_best[int(cluster_id)] = int(non_noise.idxmax())
    return {
        (cluster_a_id, cluster_b_id)
        for cluster_a_id, cluster_b_id in row_best.items()
        if col_best.get(cluster_b_id) == cluster_a_id
    }


def classify_pair(label_a: str, frame_a: pd.DataFrame, label_b: str, frame_b: pd.DataFrame) -> pd.DataFrame:
    keep_meta = [
        "sample_id",
        "country",
        "original_group_id",
        "date_mean_bp",
        "observed_fraction",
        "z1",
        "z2",
        "z3",
        "cluster_id",
        "is_optics_noise",
        "reachability",
        "core_distance",
    ]
    keep_a = [col for col in keep_meta if col in frame_a.columns]
    keep_b = [col for col in keep_meta if col in frame_b.columns]
    merged = frame_a[keep_a].merge(
        frame_b[keep_b],
        on="sample_id",
        how="inner",
        suffixes=(f"_{label_a}", f"_{label_b}"),
        validate="one_to_one",
    )
    cluster_a = f"cluster_id_{label_a}"
    cluster_b = f"cluster_id_{label_b}"
    noise_a = f"is_optics_noise_{label_a}"
    noise_b = f"is_optics_noise_{label_b}"
    mutual_pairs = _mutual_best_pairs(merged[cluster_a], merged[cluster_b])

    def status(row: pd.Series) -> str:
        a_noise = bool(row[noise_a]) or int(row[cluster_a]) == -1
        b_noise = bool(row[noise_b]) or int(row[cluster_b]) == -1
        if a_noise and b_noise:
            return "both_noise"
        if a_noise or b_noise:
            return "one_noise"
        if (int(row[cluster_a]), int(row[cluster_b])) in mutual_pairs:
            return "mutual_cluster_match"
        return "cluster_mismatch"

    merged["match_status"] = merged.apply(status, axis=1)
    merged["match_status_label"] = merged["match_status"].map(STATUS_LABELS)
    return merged


def _html_plot(frame: pd.DataFrame, label_a: str, label_b: str, coord_label: str, output_path: Path) -> None:
    xcol = f"z1_{coord_label}"
    ycol = f"z2_{coord_label}"
    zcol = f"z3_{coord_label}"
    country_col = f"country_{coord_label}"
    original_col = f"original_group_id_{coord_label}"
    cluster_a = f"cluster_id_{label_a}"
    cluster_b = f"cluster_id_{label_b}"

    traces = []
    for status in STATUS_ORDER:
        sub = frame[frame["match_status"] == status].copy()
        if sub.empty:
            continue
        custom_cols = [cluster_a, cluster_b]
        for col in (country_col, original_col, f"observed_fraction_{coord_label}"):
            if col in sub.columns:
                custom_cols.append(col)
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": f"{STATUS_LABELS[status]} ({len(sub)})",
                "x": sub[xcol].astype(float).tolist(),
                "y": sub[ycol].astype(float).tolist(),
                "z": sub[zcol].astype(float).tolist(),
                "text": sub["sample_id"].astype(str).tolist(),
                "customdata": sub[custom_cols].fillna("").astype(str).values.tolist(),
                "marker": {
                    "size": 3.4 if status in {"cluster_mismatch", "one_noise"} else 2.8,
                    "color": STATUS_COLORS[status],
                    "opacity": 0.82 if status != "both_noise" else 0.55,
                    "line": {"width": 0.8, "color": "black"},
                },
                "hovertemplate": (
                    "<b>%{text}</b><br>"
                    f"{label_a} cluster=%{{customdata[0]}}<br>"
                    f"{label_b} cluster=%{{customdata[1]}}<br>"
                    "country=%{customdata[2]}<br>"
                    "original_group_id=%{customdata[3]}<br>"
                    "observed_fraction=%{customdata[4]}<extra></extra>"
                ),
            }
        )

    payload = {
        "traces": traces,
        "layout": {
            "title": f"{label_a} vs {label_b}: match status in {coord_label} latent coordinates",
            "scene": {
                "xaxis": {"title": f"{coord_label} z1"},
                "yaxis": {"title": f"{coord_label} z2"},
                "zaxis": {"title": f"{coord_label} z3"},
            },
            "legend": {"itemsizing": "constant"},
            "margin": {"l": 0, "r": 0, "b": 0, "t": 60},
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <title>OPTICS Match Comparison</title>
  <style>
    body { margin: 0; background: #f7f3ea; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    #plot { width: 100vw; height: 100vh; }
    .note { position: absolute; top: 10px; left: 12px; z-index: 10; background: rgba(255,255,255,0.86); padding: 8px 10px; border: 1px solid #d8cbb8; border-radius: 8px; font-size: 12px; max-width: 480px; }
  </style>
</head>
<body>
  <div class="note">
    Match = sample belongs to a mutually best-overlap non-noise cluster pair. Both-noise is shown separately.
    Legend entries can be clicked to toggle categories.
  </div>
  <div id="plot"></div>
  <script>
    const payload = """
        + json.dumps(payload)
        + """;
    Plotly.newPlot("plot", payload.traces, payload.layout, {responsive: true});
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )


def render_pair(label_a: str, frame_a: pd.DataFrame, label_b: str, frame_b: pd.DataFrame, output_dir: Path) -> dict:
    pair_dir = output_dir / f"{label_a}_vs_{label_b}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    classified = classify_pair(label_a, frame_a, label_b, frame_b)
    classified.to_csv(pair_dir / "match_status.tsv", sep="\t", index=False)
    summary = classified["match_status"].value_counts().reindex(STATUS_ORDER, fill_value=0).to_dict()
    for coord_label in (label_a, label_b):
        _html_plot(
            classified,
            label_a,
            label_b,
            coord_label,
            pair_dir / f"{label_a}_vs_{label_b}_in_{coord_label}_coords.html",
        )
    return {
        "pair": f"{label_a}_vs_{label_b}",
        "n_samples": int(len(classified)),
        "status_counts": {key: int(value) for key, value in summary.items()},
        "output_dir": str(pair_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render 3D OPTICS comparison plots with matched/non-matched samples colored differently.",
    )
    parser.add_argument(
        "--assignment",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Assignment TSV. May be repeated. Defaults to matched 3D, 6D, and 20D OPTICS runs.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ANALYSIS_DIR / "optics_latent_dim_match_renderings_3dmatched_6d_20d",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    specs = dict(_parse_assignment(value) for value in args.assignment) if args.assignment else DEFAULT_ASSIGNMENTS
    frames = {label: _load_assignment(path) for label, path in specs.items()}
    summaries = [
        render_pair(label_a, frames[label_a], label_b, frames[label_b], args.output_dir)
        for label_a, label_b in combinations(frames.keys(), 2)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "assignments": {label: str(path) for label, path in specs.items()},
        "pair_summaries": summaries,
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
