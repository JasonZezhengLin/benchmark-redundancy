"""Subset robustness for the redundancy study, scipy-free.

Recomputes, with a pure-numpy Kendall tau-b:
  1. The merged-population FDR table (validates data/derived/fdr_merged12.csv).
  2. The DD matrix and N_eff on the merged 397-model population (validates
     data/derived/dd_merged12.csv and effective_merged12.txt).
  3. N_eff and DD stability on model subsets: top half versus bottom half by
     average rank percentile, and five seeded random halves, for the merged,
     v1, and v2 populations.

Outputs data/derived/subset_robustness.json and prints a summary.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

SEED = 20270819
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
DERIVED = os.path.join(DATA, "derived")

V1 = {
    "ARC": ("ARC", 1172), "HellaSwag": ("HellaSwag", 10042), "MMLU": ("MMLU", 14042),
    "TruthfulQA": ("TruthfulQA", 817), "Winogrande": ("Winogrande", 1267), "GSM8K": ("GSM8K", 1319),
}
V2 = {
    "IFEval": ("IFEval", 541), "BBH": ("BBH", 6511), "MATH Lvl 5": ("MATH", 1324),
    "GPQA": ("GPQA", 1192), "MUSR": ("MuSR", 756), "MMLU-PRO": ("MMLU-Pro", 12032),
}
V2_RAW = {
    "IFEval": "IFEval Raw", "BBH": "BBH Raw", "MATH Lvl 5": "MATH Lvl 5 Raw",
    "GPQA": "GPQA Raw", "MUSR": "MUSR Raw", "MMLU-PRO": "MMLU-PRO Raw",
}


def kendall_tau_b(x: np.ndarray, y: np.ndarray, block: int = 512) -> float:
    """Kendall tau-b via chunked pairwise sign products (handles ties)."""
    n = len(x)
    S = 0.0
    for i0 in range(0, n, block):
        xb = x[i0:i0 + block, None]
        yb = y[i0:i0 + block, None]
        sx = np.sign(xb - x[None, i0:])  # only j >= i0 to halve work
        sy = np.sign(yb - y[None, i0:])
        # zero out pairs with j <= i (within this block window)
        m = sx.shape[0]
        cols = np.arange(sx.shape[1])
        rows = np.arange(m)[:, None]
        mask = cols[None, :] > rows  # strict upper triangle in local coords
        S += float(np.sum(sx * sy * mask))
    n0 = n * (n - 1) / 2.0
    _, cx = np.unique(x, return_counts=True)
    _, cy = np.unique(y, return_counts=True)
    n1 = float(np.sum(cx * (cx - 1) / 2.0))
    n2 = float(np.sum(cy * (cy - 1) / 2.0))
    denom = np.sqrt((n0 - n1) * (n0 - n2))
    return S / denom if denom > 0 else np.nan


def tau_matrix(mat: pd.DataFrame, cols, labels):
    k = len(cols)
    tau = np.ones((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            t = kendall_tau_b(mat[cols[i]].to_numpy(float), mat[cols[j]].to_numpy(float))
            tau[i, j] = tau[j, i] = t
    return pd.DataFrame(tau, index=labels, columns=labels)


def n_eff(tau_df: pd.DataFrame) -> float:
    S = np.abs(tau_df.values)
    S = 0.5 * (S + S.T)
    w = np.clip(np.linalg.eigvalsh(S), 0, None)
    return float((w.sum() ** 2) / np.square(w).sum())


def lead_share(tau_df: pd.DataFrame) -> float:
    S = np.abs(tau_df.values)
    S = 0.5 * (S + S.T)
    w = np.clip(np.linalg.eigvalsh(S), 0, None)[::-1]
    return float(w[0] / w.sum())


def frontier_fdr(mat, cols, cm, use_raw, top_k=10):
    rows = []
    for c in cols:
        n = cm[c][1]
        p = mat[V2_RAW[c]].to_numpy(float) if (use_raw and c in V2_RAW) else mat[c].to_numpy(float) / 100.0
        p = np.clip(p, 1e-6, 1 - 1e-6)
        pk = p[np.argsort(-p)[:top_k]]
        se = np.sqrt(pk * (1 - pk) / n)
        norm = np.abs(np.diff(pk)) / np.sqrt(se[:-1] ** 2 + se[1:] ** 2)
        rows.append({"benchmark": cm[c][0], "n_items": n, "top1": round(float(pk[0]), 4),
                     "FDR_median": round(float(np.median(norm)), 3),
                     "sep_at_2sigma": int(np.sum(norm >= 2.0))})
    return pd.DataFrame(rows)


def load():
    v1 = pd.read_parquet(os.path.join(DATA, "ollb_v1.parquet"))
    v2 = pd.read_parquet(os.path.join(DATA, "ollb_v2.parquet"))
    c1, c2 = list(V1.keys()), list(V2.keys())
    m1 = v1[["fullname"] + c1].dropna().drop_duplicates("fullname").reset_index(drop=True)
    m2 = v2[["fullname"] + c2 + list(V2_RAW.values())].dropna().drop_duplicates("fullname").reset_index(drop=True)
    merged = m1.merge(m2, on="fullname")
    return m1, m2, merged


def rank_pct(col: pd.Series) -> pd.Series:
    return col.rank(pct=True)


def stats_of(tdf, nmodels, full_tau=None):
    k = len(tdf)
    d = {"models": nmodels, "n_eff": round(n_eff(tdf), 3),
         "mean_dd": round(float(1 - tdf.values[np.triu_indices(k, 1)].mean()), 3),
         "lead_share": round(lead_share(tdf), 3)}
    if full_tau is not None:
        d["max_abs_dd_dev_from_full"] = round(float(np.max(np.abs(tdf.values - full_tau.values))), 3)
    return d


def subset_report(mat, cols, labels, name, out, save):
    """Fine-grained checkpointing: keys <name>:full/top/bottom/rand<i>, plus tau
    of the full matrix cached as csv for reuse across resumed runs."""
    tau_csv = os.path.join(DERIVED, f"taufull_{name}_tmp.csv")
    if f"{name}:full" not in out:
        full_tau = tau_matrix(mat, cols, labels)
        full_tau.to_csv(tau_csv)
        out[f"{name}:full"] = stats_of(full_tau, len(mat))
        save()
        print(name, "full", out[f"{name}:full"], flush=True)
    full_tau = pd.read_csv(tau_csv, index_col=0)
    avg = pd.concat([rank_pct(mat[c]) for c in cols], axis=1).mean(axis=1)
    order = np.argsort(-avg.to_numpy())
    half = len(mat) // 2
    for sname, idx in (("top", order[:half]), ("bottom", order[half:])):
        if f"{name}:{sname}" in out:
            continue
        tdf = tau_matrix(mat.iloc[idx], cols, labels)
        out[f"{name}:{sname}"] = stats_of(tdf, len(idx), full_tau)
        save()
        print(name, sname, out[f"{name}:{sname}"], flush=True)
    for i in range(5):
        if f"{name}:rand{i}" in out:
            continue
        rng = np.random.default_rng([SEED, name.encode()[0], i])
        idx = rng.permutation(len(mat))[:half]
        tdf = tau_matrix(mat.iloc[idx], cols, labels)
        out[f"{name}:rand{i}"] = stats_of(tdf, len(idx), full_tau)
        save()
        print(name, f"rand{i}", out[f"{name}:rand{i}"], flush=True)


def main():
    ckpt = os.path.join(DERIVED, "subset_robustness.json")
    out = json.load(open(ckpt)) if os.path.exists(ckpt) else {}

    def save():
        tmp = ckpt + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=2)
        os.replace(tmp, ckpt)

    m1, m2, merged = load()
    c1, c2 = list(V1.keys()), list(V2.keys())
    l1 = [V1[c][0] for c in c1]
    l2 = [V2[c][0] for c in c2]
    cm = {**V1, **V2}

    if "fdr_merged_validated" not in out:
        # 1. validate merged FDR against the packaged derived table
        fdr = frontier_fdr(merged, c1 + c2, cm, use_raw=True)
        packaged = pd.read_csv(os.path.join(DERIVED, "fdr_merged12.csv"))
        chk = fdr.merge(packaged, on="benchmark", suffixes=("_new", "_pkg"))
        fdr_ok = bool(np.allclose(chk["FDR_median_new"], chk["FDR_median_pkg"]) and
                      (chk["sep_at_2sigma_new"] == chk["sep_at_2sigma_pkg"]).all())
        print("merged FDR matches packaged csv:", fdr_ok, flush=True)
        print(fdr.to_string(index=False), flush=True)
        out["fdr_merged_validated"] = fdr_ok
        save()

    subset_report(merged, c1 + c2, l1 + l2, "merged12", out, save)
    if "dd_merged12_max_dev_vs_packaged" not in out:
        tau12 = pd.read_csv(os.path.join(DERIVED, "taufull_merged12_tmp.csv"), index_col=0)
        dd_pkg = pd.read_csv(os.path.join(DERIVED, "dd_merged12.csv"), index_col=0)
        dev = float(np.max(np.abs((1 - tau12.values) - dd_pkg.values)))
        out["dd_merged12_max_dev_vs_packaged"] = round(dev, 4)
        print("max deviation vs packaged dd_merged12:", dev, flush=True)
        save()
    subset_report(m2, c2, l2, "v2", out, save)
    subset_report(m1, c1, l1, "v1", out, save)
    print("done", flush=True)


if __name__ == "__main__":
    main()
