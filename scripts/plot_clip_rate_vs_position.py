#!/usr/bin/env python3
"""Plot clip rate vs within-response token position: fixed clip vs CTPO-LA adaptive clip.

Clip is advantage-aware: a token is considered clipped only when the ratio
exceeds the bound in the direction that would actually trigger clipping:
  - A > 0: clipped when ratio > clip_high  (policy update suppressed upward)
  - A < 0: clipped when ratio < clip_low   (policy update suppressed downward)

Two strategies are compared:
  - Fixed clip:    ratio outside [0.5, 5]  (log ratio outside [-0.693, 1.609])
  - CTPO-LA clip:  log ratio outside [-eps_low * t^p, eps_high * t^p]

Example:
  python3 scripts/plot_clip_rate_vs_position.py --step 41 -o figures/clip_rate_step41.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DEFAULT_EXP_DIR = Path(
    "checkpoints/TIR/"
    "ctpola_cumprod_Qwen3-4B_deepscaler-train_bs512_mbs32_n8_p16000_r8000_"
    "cll0.025_clh0.05_clc0.05_lap0.5_issequence_th5.0_maskvoidFalse_save20"
)

# Fixed clip bounds in log space: ratio in [0.5, 5]
LOG_CLIP_LOW_FIXED  = np.log(0.5)   # -0.6931
LOG_CLIP_HIGH_FIXED = np.log(5.0)   #  1.6094


# ---------------------------------------------------------------------------
# Data loading
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
        f"No dump files for step {step} under {dump_dir}."
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
    """Load and merge all rank shards. Returns (ratio, mask, advantages)."""
    files = _find_dump_files(dump_dir, step)
    ratios, masks, advs = [], [], []
    for fp in files:
        z = np.load(fp)
        ratios.append(z["pg_update_is_weights"].astype(np.float64))
        masks.append(z["response_mask"].astype(np.float64))
        advs.append(z["advantages"].astype(np.float64))

    ratios = _pad_width(ratios, 0.0)
    masks  = _pad_width(masks,  0.0)
    advs   = _pad_width(advs,   0.0)

    ratio = np.vstack(ratios)
    mask  = np.vstack(masks)
    adv   = np.vstack(advs)

    if ratio.shape != adv.shape:
        import warnings
        warnings.warn(
            f"ratio shape {ratio.shape} != advantages shape {adv.shape}; "
            "clipping analysis may be inaccurate."
        )

    return ratio, mask, adv


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_clip_rates(
    ratio: np.ndarray,
    mask: np.ndarray,
    adv: np.ndarray,
    *,
    eps_low: float,
    eps_high: float,
    la_power: float,
    max_pos: int,
    min_count: int = 30,
    log_eps: float = 1e-30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-position clip rate for fixed and CTPO-LA strategies.

    Clip is advantage-aware:
      A > 0: clipped when log_ratio > clip_high
      A < 0: clipped when log_ratio < -clip_low

    Returns:
        positions         : np.ndarray [max_pos]  (1-indexed)
        clip_rate_fixed   : np.ndarray [max_pos]  clip rate under fixed clip
        clip_rate_la      : np.ndarray [max_pos]  clip rate under CTPO-LA
        counts            : np.ndarray [max_pos]  number of valid tokens per position
    """
    m   = mask > 0
    pos = np.cumsum(mask, axis=-1)

    r_safe = np.maximum(np.where(m, ratio, 1.0), log_eps)
    log_r  = np.where(m, np.log(r_safe), np.nan)

    positions        = np.arange(1, max_pos + 1, dtype=np.float64)
    clip_rate_fixed  = np.full(max_pos, np.nan)
    clip_rate_la     = np.full(max_pos, np.nan)
    counts           = np.zeros(max_pos, dtype=np.int64)

    for i, t in enumerate(range(1, max_pos + 1)):
        idx      = (pos == t) & m
        log_vals = log_r[idx]
        adv_vals = adv[idx]

        # keep only finite log-ratio entries
        finite   = np.isfinite(log_vals)
        log_vals = log_vals[finite]
        adv_vals = adv_vals[finite]

        n = len(log_vals)
        if n < min_count:
            continue
        counts[i] = n

        # Fixed clip: ratio in [0.5, 5]
        # A > 0: clipped when log_ratio > LOG_CLIP_HIGH_FIXED
        # A < 0: clipped when log_ratio < LOG_CLIP_LOW_FIXED
        clipped_fixed = (
            ((adv_vals > 0) & (log_vals > LOG_CLIP_HIGH_FIXED)) |
            ((adv_vals < 0) & (log_vals < LOG_CLIP_LOW_FIXED))
        )
        clip_rate_fixed[i] = clipped_fixed.mean()

        # CTPO-LA clip: position-scaled log-space bounds
        scale      = float(t) ** la_power
        hi_la      = eps_high * scale
        lo_la      = eps_low  * scale
        clipped_la = (
            ((adv_vals > 0) & (log_vals > hi_la)) |
            ((adv_vals < 0) & (log_vals < -lo_la))
        )
        clip_rate_la[i] = clipped_la.mean()

    return positions, clip_rate_fixed, clip_rate_la, counts


def bin_clip_rates(
    positions: np.ndarray,
    clip_rates: np.ndarray,
    counts: np.ndarray,
    bin_size: int,
    max_pos: int,
    min_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    bin_edges   = np.arange(0, max_pos + bin_size, bin_size)
    bin_centers = []
    bin_vals    = []
    for lo in bin_edges[:-1]:
        hi  = lo + bin_size
        sel = (
            (positions >= lo + 1) &
            (positions <= hi) &
            np.isfinite(clip_rates) &
            (counts >= min_count)
        )
        if sel.sum() == 0:
            continue
        w    = counts[sel].astype(float)
        vals = clip_rates[sel]
        bin_centers.append((lo + hi) / 2)
        bin_vals.append(float(np.average(vals, weights=w)))
    return np.array(bin_centers), np.array(bin_vals)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_clip_rates(
    positions: np.ndarray,
    clip_rate_fixed: np.ndarray,
    clip_rate_la: np.ndarray,
    counts: np.ndarray,
    *,
    eps_low: float,
    eps_high: float,
    la_power: float,
    out_path: Path,
    max_pos: int,
    bin_size: int,
    min_count: int,
    dpi: int,
    step: int,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    centers_fixed, vals_fixed = bin_clip_rates(
        positions, clip_rate_fixed, counts, bin_size, max_pos, min_count
    )
    centers_la, vals_la = bin_clip_rates(
        positions, clip_rate_la, counts, bin_size, max_pos, min_count
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f9fa")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.7, color="#999999")

    ax.plot(
        centers_fixed, vals_fixed * 100,
        "o-", markersize=4, linewidth=1.8,
        color="#e76f51",
        label="Fixed clip",
        zorder=3,
    )

    if len(vals_la) > 0:
        mean_la = float(np.nanmean(vals_la)) * 100
        ax.plot(
            centers_la, vals_la * 100,
            "s-", markersize=4, linewidth=1.8,
            color="#2d6a4f",
            label=f"Adaptive clip (mean={mean_la:.1f}%)",
            zorder=3
        )

        ax.axhline(
            mean_la, color="#2d6a4f", linewidth=1.0,
            linestyle=":", alpha=0.6,
        )

    ax.set_xlabel("Token position within response $t$", fontsize=12)
    ax.set_ylabel("Clip rate (%)", fontsize=12)
    ax.set_title(
        f"Clip rate vs. token position  (step {step})",
        fontsize=12, pad=10,
    )
    ax.set_xlim(0, max_pos * 1.02)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=10, framealpha=0.9, loc="upper left")

    # ax.text(
    #     0.97, 0.08,
    #     "Fixed clip rate $\\nearrow$ with $t$\nCTPO-LA clip rate $\\approx$ uniform",
    #     transform=ax.transAxes, fontsize=9,
    #     ha="right", va="bottom",
    #     bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
    #               edgecolor="#cccccc", alpha=0.9),
    # )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot_clip_rate] Saved → {out_path.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--step",       type=int,   required=True)
    p.add_argument("--dump-dir",   type=Path,  default=None)
    p.add_argument("--exp-dir",    type=Path,  default=DEFAULT_EXP_DIR)
    p.add_argument("-o", "--output", type=Path,
                   default=Path("figures/clip_rate_vs_position.pdf"))
    p.add_argument("--eps-low",    type=float, default=0.025,
                   help="base log-space lower clip threshold for CTPO-LA (default 0.025)")
    p.add_argument("--eps-high",   type=float, default=0.05,
                   help="base log-space upper clip threshold for CTPO-LA (default 0.05)")
    p.add_argument("--la-power",   type=float, default=0.5,
                   help="position scaling exponent p for CTPO-LA (default 0.5)")
    p.add_argument("--max-pos",    type=int,   default=5000)
    p.add_argument("--bin-size",   type=int,   default=50)
    p.add_argument("--min-count",  type=int,   default=30)
    p.add_argument("--dpi",        type=int,   default=300)
    args = p.parse_args()

    dump_dir = args.dump_dir or (args.exp_dir / "importance_weights")

    print(f"[plot_clip_rate] Loading step {args.step} from {dump_dir} ...")
    ratio, mask, adv = load_merged_step(dump_dir, args.step)
    print(f"[plot_clip_rate] Loaded: ratio shape={ratio.shape}, adv shape={adv.shape}")

    print("[plot_clip_rate] Computing clip rates ...")
    positions, clip_rate_fixed, clip_rate_la, counts = compute_clip_rates(
        ratio, mask, adv,
        eps_low=args.eps_low,
        eps_high=args.eps_high,
        la_power=args.la_power,
        max_pos=args.max_pos,
        min_count=args.min_count,
    )

    valid = np.isfinite(clip_rate_fixed) & (counts >= args.min_count)
    print(f"[plot_clip_rate] Valid positions: {valid.sum()} / {args.max_pos}")
    print(f"[plot_clip_rate] Fixed clip rate:   mean={np.nanmean(clip_rate_fixed[valid]):.3f}")
    print(f"[plot_clip_rate] CTPO-LA clip rate: mean={np.nanmean(clip_rate_la[valid]):.3f}")

    plot_clip_rates(
        positions, clip_rate_fixed, clip_rate_la, counts,
        eps_low=args.eps_low,
        eps_high=args.eps_high,
        la_power=args.la_power,
        out_path=args.output,
        max_pos=args.max_pos,
        bin_size=args.bin_size,
        min_count=args.min_count,
        dpi=args.dpi,
        step=args.step,
    )


if __name__ == "__main__":
    main()