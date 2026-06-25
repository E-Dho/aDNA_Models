from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OpticsConfig:
    min_samples: int = 15
    min_cluster_size: float | int = 0.005
    xi: float = 0.03
    metric: str = "minkowski"


@dataclass
class OpticsRun:
    config: OpticsConfig
    labels: np.ndarray
    reachability: np.ndarray
    core_distances: np.ndarray
    ordering: np.ndarray
    diagnostics: Dict[str, float | int | str | bool]


def _latent_columns(frame: pd.DataFrame, component_cols: Optional[Sequence[str]]) -> list[str]:
    if component_cols:
        missing = [col for col in component_cols if col not in frame.columns]
        if missing:
            raise ValueError(f"Requested latent columns missing from latents CSV: {missing}")
        return list(component_cols)
    zcols = [col for col in frame.columns if col.startswith("z")]
    if not zcols:
        raise ValueError("No latent columns found; expected z* columns or --component_cols")
    return zcols


def load_latents_with_stats(
    latents_csv: Path,
    sample_stats_tsv: Path,
    *,
    component_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, list[str]]:
    latents = pd.read_csv(latents_csv)
    stats = pd.read_csv(sample_stats_tsv, sep="\t")
    if "sample_id" not in latents.columns or "sample_id" not in stats.columns:
        raise ValueError("latents_csv and sample_stats_tsv must both contain sample_id")
    latent_cols = _latent_columns(latents, component_cols)
    frame = latents[["sample_id", *latent_cols]].merge(
        stats,
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    for col in latent_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=latent_cols).reset_index(drop=True)
    if len(frame) < 2:
        raise ValueError("At least two finite latent rows are required for clustering")
    return frame, frame[latent_cols].to_numpy(dtype=np.float64), latent_cols


def default_sweep_configs(metric: str = "minkowski") -> list[OpticsConfig]:
    return [
        OpticsConfig(min_samples=min_samples, min_cluster_size=min_cluster_size, xi=xi, metric=metric)
        for min_samples in (8, 15)
        for min_cluster_size in (0.003, 0.0075)
        for xi in (0.03, 0.05)
    ]


def _cluster_diagnostics(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    random_state: int = 42,
) -> Dict[str, float | int | bool]:
    from sklearn.metrics import silhouette_score

    labels = np.asarray(labels, dtype=np.int64)
    non_noise = labels >= 0
    cluster_labels = sorted(set(labels[non_noise].tolist()))
    cluster_sizes = pd.Series(labels[non_noise]).value_counts() if non_noise.any() else pd.Series(dtype=int)
    n_clusters = int(len(cluster_labels))
    n_noise = int((labels < 0).sum())
    noise_fraction = float(n_noise / max(len(labels), 1))
    largest_cluster_fraction = float(cluster_sizes.max() / max(non_noise.sum(), 1)) if n_clusters else 0.0
    silhouette = float("nan")
    if n_clusters >= 2 and int(non_noise.sum()) > n_clusters:
        sample_size = min(int(non_noise.sum()), 2000)
        silhouette = float(
            silhouette_score(
                values[non_noise],
                labels[non_noise],
                metric="euclidean",
                sample_size=sample_size if sample_size < int(non_noise.sum()) else None,
                random_state=random_state,
            )
        )
    degenerate = bool(n_clusters < 2 or noise_fraction >= 0.98 or largest_cluster_fraction >= 0.995)
    structure_score = (
        (0.0 if not np.isfinite(silhouette) else silhouette)
        + min(n_clusters, 40) / 100.0
        - abs(noise_fraction - 0.20) / 4.0
        - max(largest_cluster_fraction - 0.85, 0.0)
        - (2.0 if degenerate else 0.0)
    )
    return {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_fraction": noise_fraction,
        "n_non_noise": int(non_noise.sum()),
        "largest_cluster_size": int(cluster_sizes.max()) if n_clusters else 0,
        "largest_cluster_fraction": largest_cluster_fraction,
        "silhouette_non_noise": silhouette,
        "degenerate": degenerate,
        "selection_score": float(structure_score),
    }


def run_optics(values: np.ndarray, config: OpticsConfig) -> OpticsRun:
    from sklearn.cluster import OPTICS

    optics = OPTICS(
        min_samples=int(config.min_samples),
        min_cluster_size=config.min_cluster_size,
        xi=float(config.xi),
        metric=config.metric,
        cluster_method="xi",
        n_jobs=-1,
    )
    labels = optics.fit_predict(values)
    diagnostics = _cluster_diagnostics(values, labels)
    diagnostics.update(asdict(config))
    return OpticsRun(
        config=config,
        labels=np.asarray(labels, dtype=np.int64),
        reachability=np.asarray(optics.reachability_, dtype=np.float64),
        core_distances=np.asarray(optics.core_distances_, dtype=np.float64),
        ordering=np.asarray(optics.ordering_, dtype=np.int64),
        diagnostics=diagnostics,
    )


def run_optics_sweep(values: np.ndarray, configs: Sequence[OpticsConfig]) -> Tuple[OpticsRun, list[OpticsRun]]:
    if not configs:
        raise ValueError("At least one OPTICS configuration is required")
    runs = [run_optics(values, config) for config in configs]
    chosen = max(
        runs,
        key=lambda run: (
            not bool(run.diagnostics["degenerate"]),
            float(run.diagnostics["selection_score"]),
            int(run.diagnostics["n_clusters"]),
        ),
    )
    return chosen, runs


def assignments_frame(
    frame: pd.DataFrame,
    latent_cols: Sequence[str],
    run: OpticsRun,
) -> pd.DataFrame:
    ordering_rank = np.empty(len(run.ordering), dtype=np.int64)
    ordering_rank[run.ordering] = np.arange(len(run.ordering), dtype=np.int64)
    keep = [
        "sample_id",
        *latent_cols,
        "country",
        "original_group_id",
        "date_mean_bp",
        "observed_fraction",
    ]
    keep = [col for col in keep if col in frame.columns]
    out = frame[keep].copy()
    out["cluster_id"] = run.labels
    out["is_optics_noise"] = run.labels < 0
    out["reachability"] = run.reachability
    out["core_distance"] = run.core_distances
    out["ordering_rank"] = ordering_rank
    return out


def _composition(values: pd.Series, top_k: int = 5) -> str:
    counts = values.fillna("").astype(str)
    counts = counts[counts != ""].value_counts().head(top_k)
    return ";".join(f"{label}:{int(count)}" for label, count in counts.items())


def summarize_clusters(assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster_id, sub in assignments[~assignments["is_optics_noise"]].groupby("cluster_id"):
        dates = pd.to_numeric(sub.get("date_mean_bp"), errors="coerce")
        coverage = pd.to_numeric(sub.get("observed_fraction"), errors="coerce")
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "n_samples": int(len(sub)),
                "country_top5": _composition(sub["country"]) if "country" in sub else "",
                "original_group_top5": _composition(sub["original_group_id"]) if "original_group_id" in sub else "",
                "date_mean_bp_median": float(dates.median()) if dates.notna().any() else float("nan"),
                "date_mean_bp_iqr": float(dates.quantile(0.75) - dates.quantile(0.25)) if dates.notna().any() else float("nan"),
                "observed_fraction_mean": float(coverage.mean()) if coverage.notna().any() else float("nan"),
                "observed_fraction_median": float(coverage.median()) if coverage.notna().any() else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("n_samples", ascending=False) if rows else pd.DataFrame()


def sweep_summary(runs: Sequence[OpticsRun], chosen: OpticsRun) -> pd.DataFrame:
    rows = []
    chosen_key = asdict(chosen.config)
    for run in runs:
        row = dict(run.diagnostics)
        row["chosen"] = asdict(run.config) == chosen_key
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["chosen", "selection_score"], ascending=[False, False])


def write_optics_outputs(
    *,
    output_dir: Path,
    assignments: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    sweep: pd.DataFrame,
    chosen: OpticsRun,
    input_summary: Dict[str, object],
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(output_dir / "optics_cluster_assignments.tsv", sep="\t", index=False)
    assignments[assignments["is_optics_noise"]].to_csv(
        output_dir / "optics_noise_candidates.tsv",
        sep="\t",
        index=False,
    )
    cluster_summary.to_csv(output_dir / "optics_cluster_summary.tsv", sep="\t", index=False)
    sweep.to_csv(output_dir / "optics_sweep_summary.tsv", sep="\t", index=False)
    summary = {
        **input_summary,
        "chosen_config": asdict(chosen.config),
        "chosen_diagnostics": chosen.diagnostics,
        "n_assignments": int(len(assignments)),
        "n_noise_candidates": int(assignments["is_optics_noise"].sum()),
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)
    return summary


def maybe_write_reachability_plot(run: OpticsRun, output_dir: Path) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    reachability = run.reachability[run.ordering]
    finite = reachability[np.isfinite(reachability)]
    cap = float(np.quantile(finite, 0.995)) if finite.size else 1.0
    values = np.minimum(reachability, cap)
    path = output_dir / "optics_reachability.png"
    plt.figure(figsize=(12, 5))
    plt.plot(np.arange(len(values)), values, linewidth=0.8, color="#264653")
    plt.xlabel("OPTICS ordering rank")
    plt.ylabel("Reachability distance")
    plt.title("OPTICS reachability for chosen clustering")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return str(path)


def maybe_write_optics_3d_html(
    assignments: pd.DataFrame,
    latent_cols: Sequence[str],
    output_dir: Path,
) -> list[str]:
    if len(latent_cols) < 3:
        return []
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    xcol, ycol, zcol = latent_cols[:3]
    frame = assignments.copy()
    frame["cluster_label"] = frame["cluster_id"].map(
        lambda value: "OPTICS noise" if int(value) < 0 else f"cluster {int(value)}"
    )
    hover_data = {
        col: True
        for col in ["country", "original_group_id", "date_mean_bp", "observed_fraction", "reachability"]
        if col in frame.columns
    }
    fig = px.scatter_3d(
        frame,
        x=xcol,
        y=ycol,
        z=zcol,
        color="cluster_label",
        hover_name="sample_id",
        hover_data=hover_data,
        opacity=0.72,
        title="OPTICS clusters in learned latent space",
    )
    fig.update_traces(marker={"size": 2})
    fig.update_layout(scene=dict(xaxis_title=xcol, yaxis_title=ycol, zaxis_title=zcol))
    colored_path = output_dir / "optics_clusters_3d.html"
    fig.write_html(colored_path, include_plotlyjs="cdn")

    toggle = go.Figure()
    toggle.add_trace(
        go.Scatter3d(
            x=frame[xcol],
            y=frame[ycol],
            z=frame[zcol],
            mode="markers",
            name="All samples (background)",
            marker=dict(
                size=2.4,
                color="rgba(255,255,255,0.78)",
                opacity=0.78,
                line=dict(width=1, color="rgba(0,0,0,0.62)"),
            ),
            text=frame["sample_id"],
            customdata=frame[["cluster_id", "country", "original_group_id"]].to_numpy(),
            hovertemplate="<b>%{text}</b><br>cluster=%{customdata[0]}<br>country=%{customdata[1]}<br>original_group_id=%{customdata[2]}<extra></extra>",
        )
    )
    label_counts = frame["cluster_label"].value_counts()
    labels = label_counts.index.tolist()
    palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24 + px.colors.qualitative.Alphabet
    trace_indices = []
    for idx, label in enumerate(labels):
        sub = frame[frame["cluster_label"] == label]
        trace_indices.append(len(toggle.data))
        is_noise = label == "OPTICS noise"
        toggle.add_trace(
            go.Scatter3d(
                x=sub[xcol],
                y=sub[ycol],
                z=sub[zcol],
                mode="markers",
                name=f"{label} ({len(sub)})",
                visible=False,
                marker=dict(
                    size=5 if is_noise else 4,
                    color="#e63946" if is_noise else palette[idx % len(palette)],
                    opacity=0.90,
                    symbol="diamond" if is_noise else "circle",
                    line=dict(width=1, color="black"),
                ),
                text=sub["sample_id"],
                customdata=sub[["cluster_id", "country", "original_group_id", "reachability"]].to_numpy(),
                hovertemplate="<b>%{text}</b><br>cluster=%{customdata[0]}<br>country=%{customdata[1]}<br>original_group_id=%{customdata[2]}<br>reachability=%{customdata[3]:.4f}<extra></extra>",
            )
        )

    n_traces = len(toggle.data)

    def visible_for(label: Optional[str] = None, all_labels: bool = False) -> list[bool]:
        visible = [False] * n_traces
        visible[0] = True
        if all_labels:
            for trace_idx in trace_indices:
                visible[trace_idx] = True
        elif label is not None:
            visible[trace_indices[labels.index(label)]] = True
        return visible

    buttons = [
        dict(label="Background only", method="update", args=[{"visible": visible_for()}]),
        dict(label=f"Show all clusters + noise ({len(labels)})", method="update", args=[{"visible": visible_for(all_labels=True)}]),
    ]
    for label in labels:
        buttons.append(
            dict(
                label=f"{label} ({int(label_counts[label])})",
                method="update",
                args=[{"visible": visible_for(label)}],
            )
        )
    toggle.update_layout(
        title="OPTICS cluster toggles in learned latent space",
        scene=dict(xaxis_title=xcol, yaxis_title=ycol, zaxis_title=zcol),
        updatemenus=[dict(type="dropdown", x=0.01, y=1.10, xanchor="left", yanchor="top", buttons=buttons)],
        annotations=[dict(text="Dropdown toggles each OPTICS cluster or the noise set.", x=0.01, y=1.15, xref="paper", yref="paper", showarrow=False)],
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, b=0, t=78),
    )
    toggle_path = output_dir / "optics_clusters_3d_toggle.html"
    toggle.write_html(toggle_path, include_plotlyjs="cdn")

    noise_path = output_dir / "optics_noise_3d.html"
    noise_fig = go.Figure(toggle.data[:1])
    noise = frame[frame["is_optics_noise"]]
    noise_fig.add_trace(
        go.Scatter3d(
            x=noise[xcol],
            y=noise[ycol],
            z=noise[zcol],
            mode="markers",
            name=f"OPTICS noise ({len(noise)})",
            marker=dict(size=5, color="#e63946", opacity=0.92, symbol="diamond", line=dict(width=1, color="white")),
            text=noise["sample_id"],
            customdata=noise[["country", "original_group_id", "reachability"]].to_numpy(),
            hovertemplate="<b>%{text}</b><br>country=%{customdata[0]}<br>original_group_id=%{customdata[1]}<br>reachability=%{customdata[2]:.4f}<extra></extra>",
        )
    )
    noise_fig.update_layout(
        title="OPTICS noise candidates in learned latent space",
        scene=dict(xaxis_title=xcol, yaxis_title=ycol, zaxis_title=zcol),
    )
    noise_fig.write_html(noise_path, include_plotlyjs="cdn")
    return [str(colored_path), str(toggle_path), str(noise_path)]


def sorted_cluster_members(
    assignments: pd.DataFrame,
    *,
    group_col: str = "country",
    include_noise: bool = False,
) -> pd.DataFrame:
    frame = assignments.copy()
    if not include_noise:
        frame = frame[~frame["is_optics_noise"]].copy()
    if group_col not in frame.columns:
        raise ValueError(f"group_col '{group_col}' not found in assignments")
    if frame.empty:
        return frame
    counts = (
        frame.groupby(["cluster_id", group_col], dropna=False)
        .size()
        .rename("cluster_group_count")
        .reset_index()
    )
    frame = frame.merge(counts, on=["cluster_id", group_col], how="left")
    sort_cols = ["cluster_id", "cluster_group_count", group_col, "original_group_id", "sample_id"]
    sort_cols = [col for col in sort_cols if col in frame.columns]
    ascending = [True, False, True, True, True][: len(sort_cols)]
    return frame.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def write_cluster_member_tables(
    assignments: pd.DataFrame,
    output_dir: Path,
    *,
    group_col: str = "country",
    include_noise: bool = False,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    members_dir = output_dir / "cluster_members"
    members_dir.mkdir(parents=True, exist_ok=True)
    members = sorted_cluster_members(assignments, group_col=group_col, include_noise=include_noise)
    combined_path = output_dir / "optics_cluster_members_sorted.tsv"
    members.to_csv(combined_path, sep="\t", index=False)

    rows = []
    for cluster_id, sub in members.groupby("cluster_id", sort=True):
        cluster_path = members_dir / f"cluster_{int(cluster_id):03d}_members.tsv"
        sub.to_csv(cluster_path, sep="\t", index=False)
        group_counts = sub[group_col].fillna("").astype(str).value_counts()
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_size": int(len(sub)),
                "top_group_id": group_counts.index[0] if len(group_counts) else "",
                "top_group_count": int(group_counts.iloc[0]) if len(group_counts) else 0,
                "n_group_ids": int(group_counts.shape[0]),
                "members_tsv": str(cluster_path),
            }
        )
    overview = pd.DataFrame(rows).sort_values("cluster_size", ascending=False)
    overview_path = output_dir / "optics_cluster_member_overview.tsv"
    overview.to_csv(overview_path, sep="\t", index=False)
    return combined_path, overview_path, members_dir


def write_optics_cluster_browser_html(
    assignments: pd.DataFrame,
    latent_cols: Sequence[str],
    output_path: Path,
    *,
    group_col: str = "country",
    include_noise: bool = False,
) -> Path:
    if len(latent_cols) < 3:
        raise ValueError("At least three latent columns are required for the 3D browser")
    if group_col not in assignments.columns:
        raise ValueError(f"group_col '{group_col}' not found in assignments")

    xcol, ycol, zcol = latent_cols[:3]
    frame = assignments.copy()
    if not include_noise:
        frame = frame[~frame["is_optics_noise"]].copy()
    frame = frame.sort_values("cluster_id").reset_index(drop=True)

    clusters = []
    data_traces = []
    trace_index_by_cluster: dict[int, int] = {}
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#393b79", "#637939",
        "#8c6d31", "#843c39", "#7b4173", "#3182bd", "#e6550d", "#31a354",
        "#756bb1", "#636363", "#9ecae1", "#fdae6b", "#a1d99b", "#bcbddc",
    ]
    for idx, (cluster_id, sub) in enumerate(frame.groupby("cluster_id", sort=True)):
        group_counts = sub[group_col].fillna("").astype(str).value_counts()
        cluster_id_int = int(cluster_id)
        color = palette[idx % len(palette)]
        trace_index_by_cluster[cluster_id_int] = len(data_traces) + 1
        data_traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": f"cluster {cluster_id_int} ({len(sub)})",
                "x": sub[xcol].astype(float).tolist(),
                "y": sub[ycol].astype(float).tolist(),
                "z": sub[zcol].astype(float).tolist(),
                "text": sub["sample_id"].astype(str).tolist(),
                "customdata": sub[[group_col, "original_group_id", "date_mean_bp", "observed_fraction"]].astype(str).values.tolist(),
                "hovertemplate": (
                    "<b>%{text}</b><br>"
                    f"{group_col}=%{{customdata[0]}}<br>"
                    "original_group_id=%{customdata[1]}<br>"
                    "date_mean_bp=%{customdata[2]}<br>"
                    "observed_fraction=%{customdata[3]}<extra></extra>"
                ),
                "marker": {"size": 4, "color": color, "opacity": 0.9, "line": {"width": 1, "color": "black"}},
                "visible": False,
            }
        )
        members = sorted_cluster_members(sub, group_col=group_col, include_noise=True)
        member_cols = [
            "sample_id",
            group_col,
            "original_group_id",
            "date_mean_bp",
            "observed_fraction",
            "reachability",
            "core_distance",
        ]
        member_cols = [col for col in member_cols if col in members.columns]
        clusters.append(
            {
                "cluster_id": cluster_id_int,
                "cluster_size": int(len(sub)),
                "top_group_id": group_counts.index[0] if len(group_counts) else "",
                "top_group_count": int(group_counts.iloc[0]) if len(group_counts) else 0,
                "n_group_ids": int(len(group_counts)),
                "group_counts": [{"group_id": str(k), "n": int(v)} for k, v in group_counts.items()],
                "members": members[member_cols].fillna("").astype(str).to_dict(orient="records"),
            }
        )

    background = {
        "type": "scatter3d",
        "mode": "markers",
        "name": "All clustered samples (background)",
        "x": frame[xcol].astype(float).tolist(),
        "y": frame[ycol].astype(float).tolist(),
        "z": frame[zcol].astype(float).tolist(),
        "text": frame["sample_id"].astype(str).tolist(),
        "marker": {
            "size": 2.4,
            "color": "rgba(255,255,255,0.78)",
            "opacity": 0.78,
            "line": {"width": 1, "color": "rgba(0,0,0,0.62)"},
        },
        "hovertemplate": "<b>%{text}</b><extra></extra>",
    }
    traces = [background, *data_traces]

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <title>OPTICS Cluster Browser</title>
  <style>
    body {{ margin:0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:#f6f2ea; color:#211f1b; }}
    .layout {{ display:grid; grid-template-columns: minmax(620px, 1fr) 430px; height:100vh; }}
    #plot {{ width:100%; height:100vh; }}
    aside {{ border-left:1px solid #d8cbb8; padding:16px; overflow:auto; background:#fffaf2; }}
    h1 {{ font-size:18px; margin:0 0 8px; }}
    .muted {{ color:#6d6258; font-size:12px; line-height:1.4; }}
    .controls {{ display:flex; gap:8px; flex-wrap:wrap; margin:12px 0; }}
    button {{ border:1px solid #b9aa95; background:#fff; padding:6px 10px; border-radius:8px; cursor:pointer; }}
    .cluster-row {{ display:flex; align-items:center; gap:8px; padding:5px 0; border-bottom:1px solid #eee3d4; font-size:13px; }}
    .chip {{ display:inline-block; min-width:11px; min-height:11px; border-radius:999px; border:1px solid #222; }}
    .cluster-list {{ max-height:32vh; overflow:auto; border:1px solid #e5d8c7; border-radius:10px; padding:6px 10px; background:#fff; }}
    .panel {{ margin-top:14px; }}
    details {{ border:1px solid #e5d8c7; background:white; border-radius:10px; padding:8px; margin:10px 0; }}
    summary {{ cursor:pointer; font-weight:700; }}
    table {{ width:100%; border-collapse:collapse; font-size:12px; margin-top:8px; }}
    th, td {{ text-align:left; border-bottom:1px solid #eee; padding:4px 5px; vertical-align:top; }}
    th {{ position:sticky; top:0; background:#fff; }}
    .member-table {{ max-height:280px; overflow:auto; }}
    .counts {{ font-size:12px; color:#4d453f; margin-top:6px; }}
  </style>
</head>
<body>
<div class="layout">
  <div id="plot"></div>
  <aside>
    <h1>OPTICS Cluster Browser</h1>
    <div class="muted">
      Select multiple clusters to show them together. Member tables are sorted by <b>{escape(group_col)}</b>
      frequency inside each cluster, highest-count group first.
    </div>
    <div class="controls">
      <button onclick="selectAll()">Show all</button>
      <button onclick="clearAll()">Clear</button>
      <button onclick="selectLargest(10)">Top 10 largest</button>
    </div>
    <div class="cluster-list" id="clusterControls"></div>
    <div class="panel" id="selectedPanel"><div class="muted">No clusters selected.</div></div>
  </aside>
</div>
<script>
const traces = {json.dumps(traces)};
const clusters = {json.dumps(clusters)};
const traceIndexByCluster = {json.dumps(trace_index_by_cluster)};
const layout = {{
  title: "OPTICS clusters with multi-select member browser",
  scene: {{xaxis: {{title: "{xcol}"}}, yaxis: {{title: "{ycol}"}}, zaxis: {{title: "{zcol}"}}}},
  margin: {{l:0, r:0, b:0, t:42}},
  legend: {{itemsizing:"constant"}}
}};
Plotly.newPlot("plot", traces, layout, {{responsive:true}});

function clusterColor(clusterId) {{
  const idx = traceIndexByCluster[clusterId];
  return traces[idx].marker.color;
}}

function renderControls() {{
  const root = document.getElementById("clusterControls");
  root.innerHTML = clusters.map(c => `
    <label class="cluster-row">
      <input type="checkbox" class="clusterBox" value="${{c.cluster_id}}" onchange="refreshSelection()">
      <span class="chip" style="background:${{clusterColor(c.cluster_id)}}"></span>
      <span><b>cluster ${{c.cluster_id}}</b> · n=${{c.cluster_size}} · top {group_col}: ${{c.top_group_id}} (${{c.top_group_count}})</span>
    </label>
  `).join("");
}}

function selectedIds() {{
  return Array.from(document.querySelectorAll(".clusterBox:checked")).map(x => Number(x.value));
}}

function setVisible(ids) {{
  const update = Array(traces.length).fill(false);
  update[0] = true;
  ids.forEach(id => {{ update[traceIndexByCluster[id]] = true; }});
  Plotly.restyle("plot", {{visible: update}});
}}

function memberTable(members) {{
  if (!members.length) return "<div class='muted'>No members.</div>";
  const cols = Object.keys(members[0]);
  return `<div class="member-table"><table><thead><tr>${{cols.map(c => `<th>${{c}}</th>`).join("")}}</tr></thead>
    <tbody>${{members.map(row => `<tr>${{cols.map(c => `<td>${{row[c] ?? ""}}</td>`).join("")}}</tr>`).join("")}}</tbody></table></div>`;
}}

function renderPanel(ids) {{
  const panel = document.getElementById("selectedPanel");
  if (!ids.length) {{
    panel.innerHTML = "<div class='muted'>No clusters selected.</div>";
    return;
  }}
  const byId = new Map(clusters.map(c => [c.cluster_id, c]));
  panel.innerHTML = ids.map(id => {{
    const c = byId.get(id);
    const counts = c.group_counts.slice(0, 8).map(x => `${{x.group_id}}:${{x.n}}`).join("; ");
    return `<details open>
      <summary><span class="chip" style="background:${{clusterColor(id)}}"></span> cluster ${{id}} · size ${{c.cluster_size}}</summary>
      <div class="counts"><b>Top {group_col}s:</b> ${{counts}}</div>
      ${{memberTable(c.members)}}
    </details>`;
  }}).join("");
}}

function refreshSelection() {{
  const ids = selectedIds();
  setVisible(ids);
  renderPanel(ids);
}}

function clearAll() {{
  document.querySelectorAll(".clusterBox").forEach(x => x.checked = false);
  refreshSelection();
}}

function selectAll() {{
  document.querySelectorAll(".clusterBox").forEach(x => x.checked = true);
  refreshSelection();
}}

function selectLargest(n) {{
  const keep = new Set([...clusters].sort((a,b) => b.cluster_size - a.cluster_size).slice(0,n).map(c => c.cluster_id));
  document.querySelectorAll(".clusterBox").forEach(x => x.checked = keep.has(Number(x.value)));
  refreshSelection();
}}

renderControls();
refreshSelection();
</script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path
