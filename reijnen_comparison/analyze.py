"""Instance-level comparison of dispatching rules (FIFO/SPT/MOR/MWR) vs the
DRL policy (Song et al. reproduction, greedy + sampling) on the Hurink
edata/rdata/vdata benchmark.

Unlike Reijnen's paper, which only reports per-dataset averages, this script
keeps every instance separate so a size-dependent trend (DRL doing very well
on some instance sizes and dragging the average up) is visible instead of
hidden.

Expected folder layout (missing folders/files are skipped, not required):
  reijnen_comparison/{FIFO,SPT,MOR,MWR}/{edata,rdata,vdata}.csv
      columns: instance_name, makespan, runtime_seconds
  reijnen_comparison/{edata,rdata,vdata}_{G,S}/makespan_*.xlsx
      columns: file_name, <checkpoint name>          (G = greedy, S = sampling)
  reijnen_comparison/{edata,rdata,vdata}_{G,S}/time_*.xlsx
      columns: file_name, <checkpoint name>
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
DATA_DIR = BASE.parent / "data_test"
PLOTS_DIR = BASE / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

DATASETS = ["edata", "rdata", "vdata"]
DR_METHODS = ["FIFO", "SPT", "MOR", "MWR"]
DRL_MODES = {"G": "DRL-Greedy", "S": "DRL-Sampling"}

METHOD_COLORS = {
    "FIFO": "#4C72B0",
    "SPT": "#DD8452",
    "MOR": "#55A868",
    "MWR": "#8172B2",
    "DRL-Greedy": "#C44E52",
    "DRL-Sampling": "#937860",
}
DATASET_COLORS = {"edata": "#4C72B0", "rdata": "#DD8452", "vdata": "#55A868"}


def load_instance_sizes() -> dict:
    """(dataset, instance filename) -> (n_jobs, n_machines), read from the .fjs header."""
    sizes = {}
    for dataset in DATASETS:
        folder = DATA_DIR / dataset
        if not folder.exists():
            continue
        for f in folder.glob("*.fjs"):
            with open(f) as fh:
                n_jobs, n_machines = fh.readline().split()[:2]
            sizes[(dataset, f.name)] = (int(n_jobs), int(n_machines))
    return sizes


def load_dr_results() -> pd.DataFrame:
    rows = []
    for method in DR_METHODS:
        for dataset in DATASETS:
            path = BASE / method / f"{dataset}.csv"
            if not path.exists():
                print(f"[skip] {path} not found")
                continue
            df = pd.read_csv(path)
            df["dataset"] = dataset
            df["method"] = method
            rows.append(df.rename(columns={"instance_name": "instance"}))
    if not rows:
        return pd.DataFrame(columns=["dataset", "instance", "method", "makespan", "runtime_seconds"])
    return pd.concat(rows, ignore_index=True)[["dataset", "instance", "method", "makespan", "runtime_seconds"]]


def load_drl_results() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for suffix, label in DRL_MODES.items():
            folder = BASE / f"{dataset}_{suffix}"
            if not folder.exists():
                print(f"[skip] {folder} not found")
                continue
            makespan_files = sorted(folder.glob("makespan_*.xlsx"))
            time_files = sorted(folder.glob("time_*.xlsx"))
            if not makespan_files:
                print(f"[skip] no makespan_*.xlsx in {folder}")
                continue

            mk_df = pd.read_excel(makespan_files[-1])
            checkpoint_col = mk_df.columns[1]
            mk_df = mk_df.rename(columns={"file_name": "instance", checkpoint_col: "makespan"})
            mk_df = mk_df[["instance", "makespan"]]

            if time_files:
                time_df = pd.read_excel(time_files[-1])
                time_col = time_df.columns[1]
                time_df = time_df.rename(columns={"file_name": "instance", time_col: "runtime_seconds"})
                mk_df = mk_df.merge(time_df[["instance", "runtime_seconds"]], on="instance", how="left")
            else:
                mk_df["runtime_seconds"] = np.nan

            mk_df["dataset"] = dataset
            mk_df["method"] = label
            rows.append(mk_df)
    if not rows:
        return pd.DataFrame(columns=["dataset", "instance", "method", "makespan", "runtime_seconds"])
    return pd.concat(rows, ignore_index=True)[["dataset", "instance", "method", "makespan", "runtime_seconds"]]


def build_combined() -> pd.DataFrame:
    combined = pd.concat([load_dr_results(), load_drl_results()], ignore_index=True)
    sizes = load_instance_sizes()
    combined["n_jobs"] = [sizes.get((d, i), (None, None))[0] for d, i in zip(combined["dataset"], combined["instance"])]
    combined["n_machines"] = [sizes.get((d, i), (None, None))[1] for d, i in zip(combined["dataset"], combined["instance"])]
    combined["size"] = combined["n_jobs"] * combined["n_machines"]
    return combined


def compute_gaps(combined: pd.DataFrame) -> pd.DataFrame:
    """Per DRL result, gap (%) vs the best and vs the mean of the 4 dispatching rules
    on that exact instance. Negative gap = DRL beats the dispatching rules."""
    dr_only = combined[combined["method"].isin(DR_METHODS) & (combined["makespan"] > 0)]
    dropped = combined[combined["method"].isin(DR_METHODS) & (combined["makespan"] <= 0)]
    if not dropped.empty:
        bad = sorted(dropped[["dataset", "instance"]].drop_duplicates().apply(tuple, axis=1))
        print(f"[warn] dropping {len(dropped)} dispatching-rule rows with makespan <= 0 (failed run sentinel): {bad}")
    best_dr = dr_only.groupby(["dataset", "instance"])["makespan"].min().rename("best_dr_makespan")
    mean_dr = dr_only.groupby(["dataset", "instance"])["makespan"].mean().rename("mean_dr_makespan")

    drl_only = combined[combined["method"].isin(DRL_MODES.values())].copy()
    drl_only = drl_only.join(best_dr, on=["dataset", "instance"]).join(mean_dr, on=["dataset", "instance"])
    drl_only["gap_vs_best_dr_pct"] = (drl_only["makespan"] - drl_only["best_dr_makespan"]) / drl_only["best_dr_makespan"] * 100
    drl_only["gap_vs_mean_dr_pct"] = (drl_only["makespan"] - drl_only["mean_dr_makespan"]) / drl_only["mean_dr_makespan"] * 100
    return drl_only.sort_values(["dataset", "size", "instance"])


def plot_per_instance_gap(gaps: pd.DataFrame, dataset: str):
    sub = gaps[gaps["dataset"] == dataset]
    if sub.empty:
        return
    order = sub[["instance", "n_jobs", "n_machines"]].drop_duplicates().sort_values(["n_jobs", "n_machines", "instance"])
    order["x"] = range(len(order))
    x_map = dict(zip(order["instance"], order["x"]))

    fig, ax = plt.subplots(figsize=(max(10, len(order) * 0.22), 5))
    for mode_label in DRL_MODES.values():
        s = sub[sub["method"] == mode_label]
        if s.empty:
            continue
        ax.scatter([x_map[i] for i in s["instance"]], s["gap_vs_best_dr_pct"],
                   label=mode_label, color=METHOD_COLORS[mode_label], s=28, zorder=3)

    ax.axhline(0, color="#888888", linewidth=1, zorder=1)

    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    ax.set_ylim(ymin - 0.12 * yrange, ymax)

    # size-group boundaries + labels
    groups = order.groupby(["n_jobs", "n_machines"])["x"]
    for i, ((nj, nm), xs) in enumerate(groups):
        start, end = xs.min(), xs.max()
        if i % 2 == 0:
            ax.axvspan(start - 0.5, end + 0.5, color="#f2f2f2", zorder=0)
        ax.text((start + end) / 2, ymin - 0.02 * yrange, f"{nj}x{nm}", ha="center", va="top",
                fontsize=7, color="#555555", rotation=90)

    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_xticks([])
    ax.set_ylabel("gap vs. best dispatching rule (%)")
    ax.set_title(f"{dataset}: DRL vs. best-of-4 dispatching rule, per instance (grouped by size)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"per_instance_gap_{dataset}.png", dpi=150)
    plt.close(fig)


def plot_gap_vs_size(gaps: pd.DataFrame):
    modes_present = [m for m in DRL_MODES.values() if m in gaps["method"].unique()]
    if not modes_present:
        return
    fig, axes = plt.subplots(1, len(modes_present), figsize=(6 * len(modes_present), 5), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, mode_label in zip(axes, modes_present):
        s = gaps[gaps["method"] == mode_label]
        for dataset in DATASETS:
            ds = s[s["dataset"] == dataset]
            if ds.empty:
                continue
            jitter = (np.random.default_rng(0).random(len(ds)) - 0.5) * 3
            ax.scatter(ds["size"] + jitter, ds["gap_vs_best_dr_pct"], label=dataset,
                       color=DATASET_COLORS[dataset], s=22, alpha=0.75)
        s_valid = s.dropna(subset=["gap_vs_best_dr_pct"])
        if not s_valid.empty:
            coeffs = np.polyfit(s_valid["size"], s_valid["gap_vs_best_dr_pct"], 1)
            xs = np.linspace(s_valid["size"].min(), s_valid["size"].max(), 50)
            ax.plot(xs, np.polyval(coeffs, xs), color="#333333", linewidth=1.5, linestyle="--",
                    label="trend (all datasets)")
        ax.axhline(0, color="#888888", linewidth=1)
        ax.set_xlabel("instance size (n_jobs x n_machines)")
        ax.set_title(mode_label)
    axes[0].set_ylabel("gap vs. best dispatching rule (%)")
    axes[-1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "gap_vs_size.png", dpi=150)
    plt.close(fig)


def plot_gap_by_size_bucket(gaps: pd.DataFrame):
    if gaps.empty:
        return
    bucket = gaps.groupby(["dataset", "n_jobs", "n_machines", "method"])["gap_vs_best_dr_pct"].mean().reset_index()
    bucket["label"] = bucket["n_jobs"].astype(str) + "x" + bucket["n_machines"].astype(str)
    modes_present = [m for m in DRL_MODES.values() if m in bucket["method"].unique()]

    fig, axes = plt.subplots(len(DATASETS), 1, figsize=(10, 3.2 * len(DATASETS)), sharex=False)
    for ax, dataset in zip(axes, DATASETS):
        sub = bucket[bucket["dataset"] == dataset].sort_values(["n_jobs", "n_machines"])
        if sub.empty:
            ax.set_visible(False)
            continue
        labels = sub["label"].drop_duplicates().tolist()
        x = np.arange(len(labels))
        width = 0.8 / max(len(modes_present), 1)
        for j, mode_label in enumerate(modes_present):
            s = sub[sub["method"] == mode_label].set_index("label").reindex(labels)
            ax.bar(x + j * width, s["gap_vs_best_dr_pct"], width=width, label=mode_label,
                   color=METHOD_COLORS[mode_label])
        ax.axhline(0, color="#888888", linewidth=1)
        ax.set_xticks(x + width * (len(modes_present) - 1) / 2)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{dataset}: mean gap vs. best dispatching rule, by instance-size class")
        ax.set_ylabel("mean gap (%)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "gap_by_size_bucket.png", dpi=150)
    plt.close(fig)


def plot_gap_distribution(gaps: pd.DataFrame):
    if gaps.empty:
        return
    modes_present = [m for m in DRL_MODES.values() if m in gaps["method"].unique()]
    fig, ax = plt.subplots(figsize=(8, 5))
    positions, labels, data, colors = [], [], [], []
    pos = 0
    for dataset in DATASETS:
        for mode_label in modes_present:
            s = gaps[(gaps["dataset"] == dataset) & (gaps["method"] == mode_label)]["gap_vs_best_dr_pct"].dropna()
            if s.empty:
                continue
            data.append(s.values)
            positions.append(pos)
            labels.append(f"{dataset}\n{mode_label}")
            colors.append(METHOD_COLORS[mode_label])
            pos += 1
        pos += 0.6
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True, showmeans=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    ax.axhline(0, color="#888888", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("gap vs. best dispatching rule (%)")
    ax.set_title("Distribution of per-instance gap (below 0 = DRL beats the dispatching rules)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "gap_distribution_boxplot.png", dpi=150)
    plt.close(fig)


def plot_win_rate(gaps: pd.DataFrame):
    if gaps.empty:
        return
    modes_present = [m for m in DRL_MODES.values() if m in gaps["method"].unique()]
    win = gaps.groupby(["dataset", "method"])["gap_vs_best_dr_pct"].apply(lambda s: (s < 0).mean() * 100).reset_index()
    win.columns = ["dataset", "method", "win_rate_pct"]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(DATASETS))
    width = 0.8 / max(len(modes_present), 1)
    for j, mode_label in enumerate(modes_present):
        s = win[win["method"] == mode_label].set_index("dataset").reindex(DATASETS)
        ax.bar(x + j * width, s["win_rate_pct"], width=width, label=mode_label, color=METHOD_COLORS[mode_label])
    ax.set_xticks(x + width * (len(modes_present) - 1) / 2)
    ax.set_xticklabels(DATASETS)
    ax.set_ylabel("instances where DRL beats best dispatching rule (%)")
    ax.set_title("Win rate of DRL vs. best-of-4 dispatching rule")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "win_rate.png", dpi=150)
    plt.close(fig)


def main():
    combined = build_combined()
    combined.to_csv(PLOTS_DIR / "all_results_long.csv", index=False)

    gaps = compute_gaps(combined)
    gaps.to_csv(PLOTS_DIR / "instance_gaps.csv", index=False)

    for dataset in DATASETS:
        plot_per_instance_gap(gaps, dataset)
    plot_gap_vs_size(gaps)
    plot_gap_by_size_bucket(gaps)
    plot_gap_distribution(gaps)
    plot_win_rate(gaps)

    print(f"\n{len(combined)} rows loaded, {len(gaps)} DRL-vs-DR comparisons computed.")
    print(f"CSVs and plots written to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
