#!/usr/bin/env python3
"""Plot std(log rho_t) vs within-response token position, with fitted sigma*sqrt(t) curve.

This visualization directly supports the theoretical claim that
Var(log rho_t^cum) = t * sigma^2, i.e. std grows as sqrt(t).

Example:
  python3 plot_pg_ratio_std_vs_position.py --step 80 --exp-dir checkpoints/TIR/<exp>
  python3 plot_pg_ratio_std_vs_position.py --step 80 --dump-dir path/to/importance_weights

Output: a publication-ready figure with:
  - scatter/errorbar of empirical std(log rho_t) per position bin
  - fitted sigma*sqrt(t) curve
  - optional: 95% confidence band
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

DEFAULT_EXP_DIR = Path(
    "checkpoints/TIR/"
    "ctpola_cumprod_Qwen3-4B_deepscaler-train_bs512_mbs32_n8_p16000_r8000_"
    "cll0.025_clh0.05_clc0.05_lap0.5_issequence_th5.0_maskvoidFalse_save20"
)


# ---------------------------------------------------------------------------
# Data loading (reuse logic from original script)
# ---------------------------------------------------------------------------

def _find_dump_files(dump_dir: Path, step: int) -> list[Path]:
    step_sub = dump_dir / f"step_{step:07d}"
    if step_sub.is_dir():
        files = sorted(step_sub.glob("*.npz"))
        if files:
            return files
    files = sorted(dump_dir.glob(f"step_{step:07d}_rank_*.npz"))
    if files:
        return files
    raise FileNotFoundError(
        f"No dump files for step {step} under {dump_dir}. "
        "Check --step and --dump-dir / --exp-dir."
    )


def _pad_width(blocks: list[np.ndarray], pad_val: float) -> list[np.ndarray]:
    max_w = max(b.shape[1] for b in blocks)
    out = []
    for b in blocks:
        if b.shape[1] == max_w:
            out.append(b)
            continue
        pad = np.full((b.shape[0], max_w - b.shape[1]), pad_val, dtype=b.dtype)
        out.append(np.concatenate([b, pad], axis=1))
    return out


def load_merged_step(dump_dir: Path, step: int):
    files = _find_dump_files(dump_dir, step)
    ratios, masks = [], []
    for fp in files:
        z = np.load(fp)
        ratios.append(z["pg_update_is_weights"].astype(np.float64))
        masks.append(z["response_mask"].astype(np.float64))
    ratios = _pad_width(ratios, 0.0)
    masks  = _pad_width(masks,  0.0)
    return np.vstack(ratios), np.vstack(masks)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_std_per_position(
    ratio: np.ndarray,
    mask: np.ndarray,
    max_pos: int,
    log_eps: float = 1e-30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (positions, std_per_position, count_per_position) for t in [1, max_pos]."""
    m = mask > 0
    pos = np.cumsum(mask, axis=-1)          # 1-indexed position within response

    r_safe = np.maximum(np.where(m, ratio, 1.0), log_eps)
    log_r  = np.where(m, np.log(r_safe), np.nan)

    positions = np.arange(1, max_pos + 1, dtype=np.float64)
    stds   = np.full(max_pos, np.nan)
    counts = np.zeros(max_pos, dtype=np.int64)

    for i, t in enumerate(range(1, max_pos + 1)):
        vals = log_r[pos == t]
        vals = vals[np.isfinite(vals)]
        if len(vals) >= 2:
            stds[i]   = float(np.std(vals, ddof=1))
            counts[i] = len(vals)

    return positions, stds, counts


def fit_sqrt_curve(
    positions: np.ndarray,
    stds: np.ndarray,
    counts: np.ndarray,
    min_count: int = 30,
) -> float:
    """Fit sigma in std(t) = sigma * sqrt(t) via weighted least squares.

    Weight by sqrt(count) to down-weight sparse late positions.
    Returns fitted sigma.
    """
    valid = np.isfinite(stds) & (counts >= min_count)
    t  = positions[valid]
    y  = stds[valid]
    w  = np.sqrt(counts[valid].astype(float))

    # Linear regression: y = sigma * sqrt(t), no intercept
    # => sigma = sum(w * sqrt(t) * y) / sum(w * t)
    sqrt_t = np.sqrt(t)
    sigma  = float(np.sum(w * sqrt_t * y) / np.sum(w * t))
    return sigma


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_std_vs_position(
    positions: np.ndarray,
    stds: np.ndarray,
    counts: np.ndarray,
    sigma: float,
    *,
    out_path: Path,
    max_pos: int,
    min_count: int,
    dpi: int,
    bin_size: int,
    step: int,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    # ---- bin the per-position std for cleaner display ----
    bin_edges  = np.arange(0, max_pos + bin_size, bin_size)
    bin_centers, bin_stds, bin_errs, bin_counts = [], [], [], []

    for lo in bin_edges[:-1]:
        hi   = lo + bin_size
        mask = (positions >= lo + 1) & (positions <= hi) & np.isfinite(stds) & (counts >= min_count)
        if mask.sum() == 0:
            continue
        w    = counts[mask].astype(float)
        vals = stds[mask]
        wmean = np.average(vals, weights=w)
        # weighted std of the bin mean (simple SEM proxy)
        werr  = np.sqrt(np.average((vals - wmean)**2, weights=w)) / np.sqrt(mask.sum())
        bin_centers.append((lo + hi) / 2)
        bin_stds.append(wmean)
        bin_errs.append(werr)
        bin_counts.append(int(w.sum()))

    bin_centers = np.array(bin_centers)
    bin_stds    = np.array(bin_stds)
    bin_errs    = np.array(bin_errs)

    # ---- fitted curve ----
    t_curve = np.linspace(1, max_pos, 500)
    y_curve = sigma * np.sqrt(t_curve)

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f9fa")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.7, color="#999999")

    # empirical std (binned)
    ax.errorbar(
        bin_centers, bin_stds, yerr=bin_errs,
        fmt="o", markersize=4, linewidth=1.2,
        color="#2d6a4f", ecolor="#2d6a4f", elinewidth=0.8, capsize=2,
        label=r"Empirical $\mathrm{std}(\log \rho_t^{\mathrm{cum}})$",
        zorder=3,
    )

    # fitted sigma * sqrt(t)
    ax.plot(
        t_curve, y_curve,
        color="#e76f51", linewidth=2.0, linestyle="--",
        label=rf"Fitted $\hat{{\sigma}}\sqrt{{t}}$  ($\hat{{\sigma}}={sigma:.4f}$)",
        zorder=2,
    )

    # confidence band: +/- 20% of fitted curve (visual guide only)
    ax.fill_between(
        t_curve, 0.8 * y_curve, 1.2 * y_curve,
        color="#e76f51", alpha=0.12, zorder=1,
        label=r"$\pm20\%$ band",
    )

    ax.set_xlabel("Token position within response $t$", fontsize=12)
    ax.set_ylabel(r"$\mathrm{std}(\log \rho_t^{\mathrm{cum}})$", fontsize=12)
    ax.set_title(
        rf"Variance growth of $\log \rho_t^{{\mathrm{{cum}}}}$ vs. position  (step {step})",
        fontsize=12, pad=10,
    )
    ax.set_xlim(0, max_pos * 1.02)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=10, framealpha=0.9, loc="upper left")

    # annotate sigma
    ax.text(
        0.97, 0.08,
        rf"$\mathrm{{Var}}(\log \rho_t^{{\mathrm{{cum}}}}) \approx t\sigma^2$",
        transform=ax.transAxes, fontsize=10,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot_std] Saved → {out_path.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--step",     type=int,  default=41,
                   help="global training step")
    p.add_argument("--dump-dir", type=Path, default=None,
                   help="root PG dump dir (overrides --exp-dir)")
    p.add_argument("--exp-dir",  type=Path, default=DEFAULT_EXP_DIR,
                   help="experiment checkpoint dir; reads exp-dir/importance_weights")
    p.add_argument("-o", "--output", type=Path,
                   default=Path("pg_ratio_std_vs_position.pdf"),
                   help="output figure path (.pdf or .png)")
    p.add_argument("--max-pos",  type=int,  default=5000,
                   help="only consider positions <= this (default 2000)")
    p.add_argument("--bin-size", type=int,  default=50,
                   help="bin width for aggregating per-position stds (default 50)")
    p.add_argument("--min-count",type=int,  default=30,
                   help="min token count per position to include in fit / plot (default 30)")
    p.add_argument("--dpi",      type=int,  default=300)
    args = p.parse_args()

    dump_dir = args.dump_dir or (args.exp_dir / "importance_weights")

    print(f"[plot_std] Loading step {args.step} from {dump_dir} ...")
    ratio, mask = load_merged_step(dump_dir, args.step)
    print(f"[plot_std] Loaded: ratio shape={ratio.shape}")

    print("[plot_std] Computing per-position std ...")
    positions, stds, counts = compute_std_per_position(ratio, mask, args.max_pos)

    valid = np.isfinite(stds) & (counts >= args.min_count)
    print(f"[plot_std] Valid positions for fitting: {valid.sum()} / {args.max_pos}")

    sigma = fit_sqrt_curve(positions, stds, counts, min_count=args.min_count)
    print(f"[plot_std] Fitted sigma = {sigma:.6f}")

    plot_std_vs_position(
        positions, stds, counts, sigma,
        out_path=args.output,
        max_pos=args.max_pos,
        min_count=args.min_count,
        dpi=args.dpi,
        bin_size=args.bin_size,
        step=args.step,
    )


if __name__ == "__main__":
    main()