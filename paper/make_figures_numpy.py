"""Regenerate the three paper figures from the released derived tables, scipy-free.

Reads data/derived/dd_merged12.csv, fdr_merged12.csv (population column from
fdr_v1/v2), effective_merged12.txt (greedy marginal-DD order), and reproduces
figures/fig_dd_heatmap.pdf, fig_marginal_dd.pdf, fig_cluster_fdr.pdf with the
same content as the original scipy pipeline (average-linkage clustering
implemented directly).
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DERIVED = os.path.join(HERE, "data", "derived")
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)

dd12 = pd.read_csv(os.path.join(DERIVED, "dd_merged12.csv"), index_col=0)
labels = list(dd12.index)

# FDR per benchmark on each generation's full population (as in Table 4)
fdr1 = pd.read_csv(os.path.join(DERIVED, "fdr_v1.csv"))
fdr2 = pd.read_csv(os.path.join(DERIVED, "fdr_v2.csv"))
fdr12 = pd.concat([fdr1, fdr2], ignore_index=True)

# greedy order and marginal DD from effective_merged12.txt
order = []
with open(os.path.join(DERIVED, "effective_merged12.txt")) as f:
    lines = [l.rstrip("\n") for l in f]
pr12 = float(lines[0].split("\t")[1])
for l in lines[1:]:
    parts = l.split("\t")
    if parts[0].startswith(("retained_at", "eig_cumulative", "cover_tau")):
        break
    name = parts[0]
    marg = float(parts[1]) if len(parts) > 1 and parts[1] else np.nan
    order.append((name, marg))

# ---------------- Figure 1: DD heatmap ----------------
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

# ---------------- Figure 2: greedy marginal-DD curve ----------------
fig, ax = plt.subplots(figsize=(6.6, 3.8))
labs = [l for l, _ in order]
marg = [0.0 if (isinstance(m, float) and np.isnan(m)) else m for _, m in order]
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


# ---------------- average-linkage dendrogram, implemented directly ----------
def average_linkage(D):
    """Return merge list [(i, j, height, size)] using average linkage on
    condensed distance matrix D (full square numpy array)."""
    n = D.shape[0]
    active = {i: [i] for i in range(n)}
    dist = {(i, j): D[i, j] for i in range(n) for j in range(i + 1, n)}
    merges = []
    next_id = n
    while len(active) > 1:
        (a, b), h = min(dist.items(), key=lambda kv: kv[1])
        members = active[a] + active[b]
        merges.append((a, b, h, len(members)))
        del active[a], active[b]
        # distances to the new cluster: average of member pairwise distances
        newd = {}
        for c, mem in active.items():
            vals = [D[x, y] for x in members for y in mem]
            newd[c] = float(np.mean(vals))
        active[next_id] = members
        dist = {k: v for k, v in dist.items() if a not in k and b not in k}
        for c in list(active.keys()):
            if c == next_id:
                continue
            dist[(min(c, next_id), max(c, next_id))] = newd[c]
        next_id += 1
    return merges


def dendrogram_coords(merges, n):
    """Compute leaf order and segment coordinates for a simple dendrogram."""
    children = {}
    for idx, (a, b, h, s) in enumerate(merges):
        children[n + idx] = (a, b, h)

    def leaves(node):
        if node < n:
            return [node]
        a, b, _ = children[node]
        return leaves(a) + leaves(b)

    root = n + len(merges) - 1
    order = leaves(root)
    xpos = {leaf: i for i, leaf in enumerate(order)}
    segs = []

    def draw(node):
        if node < n:
            return xpos[node], 0.0
        a, b, h = children[node]
        xa, ha = draw(a)
        xb, hb = draw(b)
        segs.append(((xa, ha), (xa, h)))
        segs.append(((xb, hb), (xb, h)))
        segs.append(((xa, h), (xb, h)))
        return 0.5 * (xa + xb), h

    draw(root)
    return order, segs


merges = average_linkage(dd12.values)
leaf_order, segs = dendrogram_coords(merges, len(labels))

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
for (x1, y1), (x2, y2) in segs:
    axes[0].plot([x1, x2], [y1, y2], color="#1f4e79", lw=1.0)
axes[0].set_xticks(range(len(labels)))
axes[0].set_xticklabels([labels[i] for i in leaf_order], rotation=90, fontsize=7)
axes[0].set_title("Benchmark clustering by DD distance", fontsize=9)
axes[0].set_ylabel("DD (average linkage)", fontsize=8)

f = fdr12.sort_values("FDR_median")
axes[1].barh(f["benchmark"], f["FDR_median"], color="#c0504d")
axes[1].axvline(1.0, ls="--", color="gray", lw=0.8)
axes[1].set_xlabel("FDR (median normalised frontier gap)", fontsize=8)
axes[1].set_title("Frontier resolution per benchmark", fontsize=9)
axes[1].tick_params(axis="y", labelsize=7)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig_cluster_fdr.pdf"))
plt.close(fig)
print("figures written to", FIGS)
