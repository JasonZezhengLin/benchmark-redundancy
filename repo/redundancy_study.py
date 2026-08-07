"""
redundancy_study.py
===================

Empirical measurement of redundancy and frontier resolution in the current LLM
benchmark ecosystem, using the Marginal Information Statement diagnostics
(Differentiation Degradation DD, Frontier Discrimination Resolution FDR) as the
measurement instrument. All scores are real public leaderboard data (see
data/PROVENANCE.md). This script computes:

  1. Pairwise Kendall tau-b and DD = 1 - tau between every benchmark pair, for
     three real matrices: Open LLM Leaderboard v1 (6 benchmarks), v2 (6
     benchmarks), and a merged 12-benchmark matrix on shared models.
  2. FDR per benchmark from real frontier models and a binomial noise floor
     derived from the standard test-set size.
  3. The effective number of non-redundant benchmarks, via (a) the participation
     ratio of the rank-correlation matrix and (b) a greedy marginal-DD ordering.
  4. Hierarchical clustering of benchmarks by DD distance.

Outputs: CSV tables in data/derived/ and figure PDFs in figures/.
Deterministic; the only randomness is the bootstrap, which is seeded.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20270819
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
DERIVED = os.path.join(DATA, "derived")
FIGS = os.path.join(HERE, "figures")
os.makedirs(DERIVED, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

# ---------------------------------------------------------------------------
# Benchmark definitions: column name -> (short label, test-set size n)
# n is the standard public test-set size, used only for the binomial floor.
# ---------------------------------------------------------------------------
V1 = {
    "ARC": ("ARC", 1172),
    "HellaSwag": ("HellaSwag", 10042),
    "MMLU": ("MMLU", 14042),
    "TruthfulQA": ("TruthfulQA", 817),
    "Winogrande": ("Winogrande", 1267),
    "GSM8K": ("GSM8K", 1319),
}
V2 = {
    "IFEval": ("IFEval", 541),
    "BBH": ("BBH", 6511),
    "MATH Lvl 5": ("MATH", 1324),
    "GPQA": ("GPQA", 1192),
    "MUSR": ("MuSR", 756),
    "MMLU-PRO": ("MMLU-Pro", 12032),
}
# Raw (0-1 accuracy) columns available only for v2; v1 scores are direct accuracies.
V2_RAW = {
    "IFEval": "IFEval Raw", "BBH": "BBH Raw", "MATH Lvl 5": "MATH Lvl 5 Raw",
    "GPQA": "GPQA Raw", "MUSR": "MUSR Raw", "MMLU-PRO": "MMLU-PRO Raw",
}


def load_matrices():
    v1 = pd.read_parquet(os.path.join(DATA, "ollb_v1.parquet"))
    v2 = pd.read_parquet(os.path.join(DATA, "ollb_v2.parquet"))
    c1 = list(V1.keys())
    c2 = list(V2.keys())
    m1 = v1[["fullname"] + c1].dropna().drop_duplicates("fullname").reset_index(drop=True)
    raw2 = list(V2_RAW.values())
    m2 = v2[["fullname"] + c2 + raw2].dropna().drop_duplicates("fullname").reset_index(drop=True)
    merged = m1.merge(m2, on="fullname")
    return m1, m2, merged


def kendall_matrix(mat: pd.DataFrame, cols):
    """Pairwise Kendall tau-b and DD = 1 - tau over the shared model set."""
    k = len(cols)
    tau = np.ones((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            t, _ = stats.kendalltau(mat[cols[i]].values, mat[cols[j]].values,
                                    variant="b")
            tau[i, j] = tau[j, i] = t
    dd = 1.0 - tau
    labels = [cols_label(c, cols_map(cols)) for c in cols]
    return (pd.DataFrame(tau, index=labels, columns=labels),
            pd.DataFrame(dd, index=labels, columns=labels))


def cols_map(cols):
    d = {}
    d.update({k: v for k, v in V1.items()})
    d.update({k: v for k, v in V2.items()})
    return {c: d[c] for c in cols}


def cols_label(col, m):
    return m[col][0]


def effective_number_pr(tau_df: pd.DataFrame) -> float:
    """Participation ratio of the |tau| similarity matrix eigenspectrum:
    N_eff = (sum lambda)^2 / sum(lambda^2). Treats the benchmark set as a
    correlation structure and returns its effective dimensionality."""
    S = np.abs(tau_df.values)
    S = 0.5 * (S + S.T)
    w = np.linalg.eigvalsh(S)
    w = np.clip(w, 0, None)
    return float((w.sum() ** 2) / (np.square(w).sum()))


def eig_spectrum(tau_df: pd.DataFrame):
    """Eigenvalue spectrum of the |tau| similarity matrix and cumulative share
    of the trace captured by the leading k components."""
    S = np.abs(tau_df.values)
    S = 0.5 * (S + S.T)
    w = np.clip(np.linalg.eigvalsh(S), 0, None)[::-1]
    cum = np.cumsum(w) / w.sum()
    return w, cum


def covering_subset(tau_df: pd.DataFrame, thr: float):
    """Greedy minimum covering subset: smallest set S of benchmarks such that
    every benchmark b has max_{s in S} |tau(b,s)| >= thr (i.e. every benchmark
    is reproduced up to DD <= 1-thr by some member of S). Returns the ordered
    subset. This is the representative (non-redundant) core at level thr."""
    labels = list(tau_df.index)
    T = np.abs(tau_df.values.copy())
    n = len(labels)
    covered = np.zeros(n, dtype=bool)
    chosen = []
    while not covered.all():
        best, best_gain = None, -1
        for c in range(n):
            if c in chosen:
                continue
            gain = int(np.sum((T[c] >= thr) & (~covered)))
            if gain > best_gain:
                best_gain, best = gain, c
        chosen.append(best)
        covered |= (T[best] >= thr)
        covered[best] = True
    return [labels[c] for c in chosen]


def greedy_marginal_dd(tau_df: pd.DataFrame):
    """Greedy ordering that maximises marginal information at each step.
    Start from the pair with the lowest tau (most complementary). Then add the
    benchmark whose maximum tau to the selected set is smallest; its marginal
    DD is 1 - that maximum tau. Returns the ordered labels and marginal DD."""
    labels = list(tau_df.index)
    tau = tau_df.values
    n = len(labels)
    # seed with the most complementary pair
    best = (0, 1)
    lo = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            if tau[i, j] < lo:
                lo = tau[i, j]
                best = (i, j)
    selected = [best[0], best[1]]
    order = [(labels[best[0]], np.nan), (labels[best[1]], 1.0 - tau[best[0], best[1]])]
    remaining = [x for x in range(n) if x not in selected]
    while remaining:
        # pick the candidate with the smallest maximum tau to the selected set
        cand, cand_marg = None, -np.inf
        for c in remaining:
            max_tau = max(tau[c, s] for s in selected)
            marg = 1.0 - max_tau
            if marg > cand_marg:
                cand_marg, cand = marg, c
        selected.append(cand)
        remaining.remove(cand)
        order.append((labels[cand], cand_marg))
    return order


def frontier_fdr(mat: pd.DataFrame, cols, use_raw, top_k=10):
    """FDR per benchmark. Frontier = top_k models by that benchmark's score.
    se = sqrt(p(1-p)/n) with raw accuracy p. For adjacent frontier pairs,
    normalised gap = |p_hi - p_lo| / sqrt(se_hi^2 + se_lo^2). FDR is the median
    normalised adjacent gap; sep2 is the count of adjacent pairs separable at
    the 2-sigma level out of (top_k - 1). Low FDR means the frontier is not
    resolved (saturated near the ceiling or compressed near a floor)."""
    cm = cols_map(cols)
    rows = []
    for c in cols:
        n = cm[c][1]
        if use_raw and c in V2_RAW:
            p = mat[V2_RAW[c]].values
        else:
            p = mat[c].values / 100.0
        p = np.clip(p, 1e-6, 1 - 1e-6)
        order = np.argsort(-p)[:top_k]
        pk = p[order]
        se = np.sqrt(pk * (1 - pk) / n)
        gaps = np.abs(np.diff(pk))
        comb = np.sqrt(se[:-1] ** 2 + se[1:] ** 2)
        norm = gaps / comb
        rows.append({
            "benchmark": cm[c][0],
            "n_items": n,
            "top1": round(float(pk[0]), 4),
            "topk_spread": round(float(pk[0] - pk[-1]), 4),
            "FDR_median": round(float(np.median(norm)), 3),
            "FDR_min": round(float(np.min(norm)), 3),
            "sep_at_2sigma": int(np.sum(norm >= 2.0)),
            "adjacent_pairs": int(top_k - 1),
        })
    return pd.DataFrame(rows)


def bootstrap_dd_ci(mat: pd.DataFrame, ci, cj, B=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    x = mat[ci].values
    y = mat[cj].values
    n = len(x)
    vals = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        t, _ = stats.kendalltau(x[idx], y[idx], variant="b")
        vals[b] = 1.0 - t
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def run_matrix(name, mat, cols, use_raw):
    print(f"\n=== {name}  (models={len(mat)}, benchmarks={len(cols)}) ===")
    tau_df, dd_df = kendall_matrix(mat, cols)
    tau_df.round(3).to_csv(os.path.join(DERIVED, f"tau_{name}.csv"))
    dd_df.round(3).to_csv(os.path.join(DERIVED, f"dd_{name}.csv"))
    # redundant / complementary pairs
    labs = list(dd_df.index)
    pairs = []
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            pairs.append((labs[i], labs[j], round(tau_df.iloc[i, j], 3),
                          round(dd_df.iloc[i, j], 3)))
    pdf = pd.DataFrame(pairs, columns=["bench_a", "bench_b", "tau_b", "DD"])
    pdf = pdf.sort_values("DD").reset_index(drop=True)
    pdf.to_csv(os.path.join(DERIVED, f"pairs_{name}.csv"), index=False)
    print("Most redundant pair:", tuple(pdf.iloc[0][["bench_a", "bench_b", "DD"]]))
    print("Most complementary pair:", tuple(pdf.iloc[-1][["bench_a", "bench_b", "DD"]]))
    print("Mean off-diagonal DD:", round(pdf["DD"].mean(), 3))
    # FDR
    fdr = frontier_fdr(mat, cols, use_raw)
    fdr.to_csv(os.path.join(DERIVED, f"fdr_{name}.csv"), index=False)
    print(fdr.to_string(index=False))
    # effective number
    pr = effective_number_pr(tau_df)
    order = greedy_marginal_dd(tau_df)
    w, cum = eig_spectrum(tau_df)
    cover70 = covering_subset(tau_df, 0.70)
    cover50 = covering_subset(tau_df, 0.50)
    print("Eigenvalue cumulative share (leading k):", [round(float(x), 3) for x in cum[:min(4, len(cum))]])
    print(f"Representative core (max pairwise DD <= 0.30, tau>=0.70): {len(cover70)} benchmarks -> {cover70}")
    print(f"Representative core (max pairwise DD <= 0.50, tau>=0.50): {len(cover50)} benchmarks -> {cover50}")
    print(f"Effective number of non-redundant benchmarks (participation ratio): {pr:.2f} of {len(cols)}")
    marg = [round(m, 3) for _, m in order[1:]]
    print("Greedy marginal-DD sequence:", [(l, None if np.isnan(m) else round(m, 3)) for l, m in order])
    # count until marginal DD drops below thresholds
    saturating = {}
    for thr in (0.30, 0.20, 0.10):
        cnt = 2  # seed pair
        for l, m in order[2:]:
            if m >= thr:
                cnt += 1
            else:
                break
        saturating[thr] = cnt
    print("Benchmarks retained before marginal DD <", saturating)
    with open(os.path.join(DERIVED, f"effective_{name}.txt"), "w") as f:
        f.write(f"participation_ratio\t{pr:.4f}\tof\t{len(cols)}\n")
        for l, m in order:
            f.write(f"{l}\t{'' if (isinstance(m,float) and np.isnan(m)) else round(m,4)}\n")
        for thr, cnt in saturating.items():
            f.write(f"retained_at_thr_{thr}\t{cnt}\n")
        f.write("eig_cumulative\t" + "\t".join(f"{x:.4f}" for x in cum) + "\n")
        f.write(f"cover_tau0.70\t{len(cover70)}\t" + ",".join(cover70) + "\n")
        f.write(f"cover_tau0.50\t{len(cover50)}\t" + ",".join(cover50) + "\n")
    return tau_df, dd_df, pdf, fdr, pr, order


def main():
    np.random.seed(SEED)
    m1, m2, merged = load_matrices()
    print("Loaded real matrices:")
    print(f"  v1 (classic 6):   {len(m1)} models")
    print(f"  v2 (current 6):   {len(m2)} models")
    print(f"  merged (12):      {len(merged)} models scored on all twelve")

    c1, c2 = list(V1.keys()), list(V2.keys())
    run_matrix("v1", m1, c1, use_raw=False)
    run_matrix("v2", m2, c2, use_raw=True)
    tau12, dd12, pairs12, fdr12, pr12, order12 = run_matrix(
        "merged12", merged, c1 + c2, use_raw=True)

    # bootstrap CI for the extreme pairs in the merged matrix
    print("\n=== Bootstrap 95% DD intervals (merged 12, B=1000) ===")
    boot_rows = []
    extreme = pd.concat([pairs12.head(3), pairs12.tail(3)])
    inv = {v[0]: k for k, v in {**V1, **V2}.items()}
    for _, r in extreme.iterrows():
        ci, cj = inv[r["bench_a"]], inv[r["bench_b"]]
        lo, hi = bootstrap_dd_ci(merged, ci, cj)
        boot_rows.append({"bench_a": r["bench_a"], "bench_b": r["bench_b"],
                          "DD": r["DD"], "DD_lo": round(lo, 3), "DD_hi": round(hi, 3)})
        print(f"  {r['bench_a']:>10} vs {r['bench_b']:<10} DD={r['DD']:.3f} [{lo:.3f}, {hi:.3f}]")
    pd.DataFrame(boot_rows).to_csv(os.path.join(DERIVED, "bootstrap_merged12.csv"), index=False)

    make_figures(dd12, tau12, fdr12, order12, pr12)
    print("\nAll derived tables in data/derived/, figures in figures/.")


def make_figures(dd12, tau12, fdr12, order12, pr12):
    labels = list(dd12.index)
    # Figure 1: DD heatmap (merged 12)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(dd12.values, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = dd12.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < 0.6 else "black", fontsize=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Differentiation Degradation  DD = 1 - Kendall tau-b", fontsize=8)
    ax.set_title("Pairwise ranking redundancy across 12 real benchmarks", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_dd_heatmap.pdf"))
    plt.close(fig)

    # Figure 2: greedy marginal-DD curve
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    labs = [l for l, _ in order12]
    marg = [0.0 if (isinstance(m, float) and np.isnan(m)) else m for _, m in order12]
    xs = range(1, len(labs) + 1)
    ax.plot(xs, marg, "o-", color="#1f4e79")
    ax.axhline(0.20, ls="--", color="gray", lw=0.8)
    ax.axhline(0.10, ls=":", color="gray", lw=0.8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Marginal DD when added", fontsize=9)
    ax.set_xlabel("Benchmarks added in greedy complementary order", fontsize=9)
    ax.set_title(f"Marginal ranking information decays; effective non-redundant count = {pr12:.1f}",
                 fontsize=9)
    ax.text(len(labs), 0.205, "DD = 0.20", fontsize=7, color="gray", va="bottom", ha="right")
    ax.text(len(labs), 0.105, "DD = 0.10", fontsize=7, color="gray", va="bottom", ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_marginal_dd.pdf"))
    plt.close(fig)

    # Figure 3: dendrogram + FDR bars
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
    dist = squareform(dd12.values, checks=False)
    Z = hierarchy.linkage(dist, method="average")
    hierarchy.dendrogram(Z, labels=labels, ax=axes[0],
                         leaf_font_size=7, color_threshold=0.6)
    axes[0].set_title("Benchmark clustering by DD distance", fontsize=9)
    axes[0].set_ylabel("DD (average linkage)", fontsize=8)
    axes[0].tick_params(axis="x", labelrotation=90)

    f = fdr12.sort_values("FDR_median")
    axes[1].barh(f["benchmark"], f["FDR_median"], color="#c0504d")
    axes[1].axvline(1.0, ls="--", color="gray", lw=0.8)
    axes[1].set_xlabel("FDR (median normalised frontier gap)", fontsize=8)
    axes[1].set_title("Frontier resolution per benchmark", fontsize=9)
    axes[1].tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_cluster_fdr.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
