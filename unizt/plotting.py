"""Publication-grade 300 DPI plotting utilities for UniZT."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from unizt.core.zt import ZTResult

PathLike = Union[str, Path]


def _apply_pub_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.2,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def plot_kappa_l_vs_t(
    T: np.ndarray,
    kappa_slack: np.ndarray,
    kappa_cahill: Optional[np.ndarray] = None,
    custom_points: Optional[tuple[np.ndarray, np.ndarray]] = None,
    output_path: PathLike = "kappa_l_vs_t.png",
    title: Optional[str] = None,
) -> Path:
    """Plot Slack lattice thermal conductivity vs Cahill-Pohl amorphous minimum."""
    _apply_pub_style()
    T_arr = np.asarray(T, dtype=np.float64)
    order = np.argsort(T_arr)
    T_s = T_arr[order]
    k_s = np.asarray(kappa_slack, dtype=np.float64)[order]

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot(
        T_s,
        k_s,
        color="#0072B2",
        linewidth=2.2,
        marker="o",
        markersize=4.5,
        label=r"Slack $\kappa_L(T)$",
    )

    if kappa_cahill is not None:
        kc_s = np.asarray(kappa_cahill, dtype=np.float64)[order]
        ax.plot(
            T_s,
            kc_s,
            color="#D55E00",
            linestyle="--",
            linewidth=2.0,
            label=r"Cahill–Pohl $\kappa_{\min}(T)$",
        )

    if custom_points is not None:
        t_c, k_c = custom_points
        ax.scatter(
            t_c,
            k_c,
            color="#009E73",
            marker="s",
            s=36,
            zorder=5,
            label=r"Custom / Reference $\kappa_L$",
        )

    ax.set_xlabel("Temperature $T$ (K)")
    ax.set_ylabel(r"Lattice Thermal Conductivity $\kappa_L$ (W m$^{-1}$ K$^{-1}$)")
    ax.set_title(title or r"Lattice Thermal Conductivity: Slack vs. Cahill–Pohl Limit")
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, edgecolor="#CCCCCC")
    ax.set_ylim(bottom=0.0)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_zt_vs_t(
    zt_result: ZTResult,
    zt_const_result: Optional[ZTResult] = None,
    output_path: PathLike = "zt_vs_t.png",
    title: Optional[str] = None,
) -> Path:
    """Plot optimal n-type and p-type peak ZT vs. T (optionally comparing DPT vs CRTA)."""
    _apply_pub_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.7))

    T = zt_result.T_grid
    n_zt = zt_result.n_type["zt_max"]
    p_zt = zt_result.p_type["zt_max"]

    if np.any(np.isfinite(n_zt)):
        ax.plot(
            T,
            n_zt,
            color="#0072B2",
            linewidth=2.2,
            marker="o",
            markersize=5,
            label="n-type (primary)",
        )
    if np.any(np.isfinite(p_zt)):
        ax.plot(
            T,
            p_zt,
            color="#D55E00",
            linewidth=2.2,
            marker="^",
            markersize=5,
            label="p-type (primary)",
        )

    if zt_const_result is not None:
        Tc = zt_const_result.T_grid
        nc = zt_const_result.n_type["zt_max"]
        pc = zt_const_result.p_type["zt_max"]
        if np.any(np.isfinite(nc)):
            ax.plot(
                Tc,
                nc,
                color="#0072B2",
                linestyle="--",
                linewidth=1.6,
                alpha=0.75,
                label=r"n-type (constant $\tau$)",
            )
        if np.any(np.isfinite(pc)):
            ax.plot(
                Tc,
                pc,
                color="#D55E00",
                linestyle="--",
                linewidth=1.6,
                alpha=0.75,
                label=r"p-type (constant $\tau$)",
            )

    ax.set_xlabel("Temperature $T$ (K)")
    ax.set_ylabel("Optimal Figure of Merit $ZT_{\\max}$")
    ax.set_title(title or "Optimal Thermoelectric Figure of Merit $ZT_{\\max}(T)$")
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, edgecolor="#CCCCCC")
    ax.set_ylim(bottom=0.0)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_single_regime_contour(
    ax: plt.Axes,
    log_n: np.ndarray,
    T: np.ndarray,
    zt: np.ndarray,
    xlabel: str,
    subtitle: str,
    vmin: float,
    vmax: float,
):
    # Downsample dense DFT grids (> 15k points per regime) so triangulation stays fast
    if log_n.size > 15000:
        stride = max(1, log_n.size // 12000)
        log_n = log_n[::stride]
        T = T[::stride]
        zt = zt[::stride]
    triang = mtri.Triangulation(log_n, T)
    levels = np.linspace(vmin, max(vmax, vmin + 1e-6), 24)
    cf = ax.tricontourf(triang, zt, levels=levels, cmap="inferno", extend="both")
    ax.tricontour(triang, zt, levels=8, colors="white", linewidths=0.45, alpha=0.45)
    ax.set_xlabel(xlabel)
    ax.set_title(subtitle)
    return cf


def plot_zt_heatmap(
    zt_result: ZTResult,
    output_path: PathLike = "zt_heatmap.png",
    title: Optional[str] = None,
) -> Path:
    """Plot 2D contour/heatmap of true ``ZT`` vs ``log10(|n|)`` and ``T``."""
    _apply_pub_style()
    n_abs = np.abs(zt_result.n)
    valid = (n_abs > 0.0) & np.isfinite(zt_result.zt) & np.isfinite(zt_result.T)
    if not np.any(valid):
        raise ValueError("No valid nonzero carrier concentration points to plot ZT heatmap.")

    max_n = float(np.max(n_abs[valid]))
    if max_n > 1e12:
        # Convert SI m^-3 -> cm^-3 for standard thermoelectric convention
        n_disp = n_abs * 1e-6
        xlabel = r"$\log_{10}(|n|\;[\mathrm{cm}^{-3}])$"
    else:
        n_disp = n_abs
        xlabel = r"$\log_{10}(|n|)$"

    log_n = np.log10(np.maximum(n_disp, 1e-30))
    n_mask = valid & (zt_result.n < 0.0)
    p_mask = valid & (zt_result.n > 0.0)

    has_n = int(np.sum(n_mask)) >= 6 and len(np.unique(zt_result.T[n_mask])) >= 2
    has_p = int(np.sum(p_mask)) >= 6 and len(np.unique(zt_result.T[p_mask])) >= 2

    vmax = float(np.nanmax(zt_result.zt[valid]))
    vmin = 0.0

    if has_n and has_p:
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True)
        cf_n = _plot_single_regime_contour(
            axes[0],
            log_n[n_mask],
            zt_result.T[n_mask],
            zt_result.zt[n_mask],
            xlabel,
            "n-type ($n < 0$)",
            vmin,
            vmax,
        )
        axes[0].set_ylabel("Temperature $T$ (K)")
        cf_p = _plot_single_regime_contour(
            axes[1],
            log_n[p_mask],
            zt_result.T[p_mask],
            zt_result.zt[p_mask],
            xlabel,
            "p-type ($n > 0$)",
            vmin,
            vmax,
        )
        cbar = fig.colorbar(cf_p, ax=axes.ravel().tolist(), pad=0.03)
        cbar.set_label(r"Thermoelectric Figure of Merit $ZT$")
        if title:
            fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    else:
        mask = n_mask if has_n else (p_mask if has_p else valid)
        regime_label = "n-type ($n < 0$)" if has_n else ("p-type ($n > 0$)" if has_p else "All carriers")
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        cf = _plot_single_regime_contour(
            ax,
            log_n[mask],
            zt_result.T[mask],
            zt_result.zt[mask],
            xlabel,
            title or f"True $ZT(T, n)$ — {regime_label}",
            vmin,
            vmax,
        )
        ax.set_ylabel("Temperature $T$ (K)")
        cbar = fig.colorbar(cf, ax=ax, pad=0.03)
        cbar.set_label(r"Thermoelectric Figure of Merit $ZT$")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_efficiency_vs_th(
    eff_data: dict[str, np.ndarray],
    output_path: PathLike = "efficiency_vs_th.png",
    T_c: float = 300.0,
    title: Optional[str] = None,
) -> Path:
    """Plot device conversion efficiency ``eta_max (%)`` and ``ZT_eng`` vs ``T_h``."""
    _apply_pub_style()
    Th = np.asarray(eff_data["T_h"], dtype=np.float64)
    eta_pct = np.asarray(eff_data["eta_max_pct"], dtype=np.float64)
    zt_eng = np.asarray(eff_data["ZT_eng"], dtype=np.float64)

    fig, ax1 = plt.subplots(figsize=(6.4, 4.6))
    color_eta = "#0072B2"
    color_zt = "#D55E00"

    l1 = ax1.plot(
        Th,
        eta_pct,
        color=color_eta,
        linewidth=2.3,
        marker="o",
        markersize=4.5,
        label=r"Max Efficiency $\eta_{\max}$ (%)",
    )
    ax1.set_xlabel(r"Hot-Side Temperature $T_h$ (K)")
    ax1.set_ylabel(r"Device Efficiency $\eta_{\max}$ (%)", color=color_eta)
    ax1.tick_params(axis="y", labelcolor=color_eta)
    ax1.set_ylim(bottom=0.0)
    ax1.grid(True, linestyle=":", alpha=0.45)

    ax2 = ax1.twinx()
    l2 = ax2.plot(
        Th,
        zt_eng,
        color=color_zt,
        linestyle="--",
        linewidth=2.1,
        marker="s",
        markersize=4.2,
        label=r"Engineering $ZT_{\mathrm{eng}}$",
    )
    ax2.set_ylabel(r"Engineering Figure of Merit $ZT_{\mathrm{eng}}$", color=color_zt)
    ax2.tick_params(axis="y", labelcolor=color_zt)
    ax2.set_ylim(bottom=0.0)

    lines = l1 + l2
    labels = [line.get_label() for line in lines]
    ax1.legend(
        lines,
        labels,
        loc="upper left",
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        edgecolor="#CCCCCC",
    )
    ax1.set_title(
        title or rf"Device Efficiency & $ZT_{{\mathrm{{eng}}}}$ vs. $T_h$ ($T_c = {T_c:.0f}\;\mathrm{{K}}$)"
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out
