"""Leaf Angle Distribution (LAD) Comparison Experiment.

This script compares the three leaf angle distribution models available in
SCOPE-RTM and demonstrates their impact on canopy directional reflectance.

Theory and references
---------------------
The leaf angle distribution (LAD, also called leaf inclination distribution
function — LIDF) describes the probability density of leaf normal zenith
angles within a canopy. It is one of the key structural inputs to any
turbid-medium radiative transfer model because it controls:

  - **Extinction coefficients** (ks, ko): how rapidly direct solar and
    observer beams are attenuated as they penetrate the canopy.
  - **Scattering phase function** (sob, sof): the directional pattern of
    single-scattering from individual leaves.
  - **Bidirectional gap fraction**: the probability that a ray traverses
    the canopy without interception.

Three main parameterizations are widely used in the vegetation remote
sensing literature:

1. **Verhoef two-parameter bimodal** (Verhoef, 1998)
   - Parameters: a (primary shape), b (secondary shape)
   - Generates six canonical distribution types:
       Planophile  (a= 1, b= 0)  — horizontal leaves
       Erectophile (a=-1, b= 0)  — vertical leaves
       Plagiophile (a= 0, b=-1)  — oblique leaves (peak ~45deg)
       Extremophile(a= 0, b= 1)  — bimodal (very flat + very steep)
       Spherical   (a=-0.35, b=-0.15) — uniform solid angle
       Uniform     (a= 0, b= 0)  — uniform in angle space
   - In SCOPE-RTM: ``scope_lidf(lidfa, lidfb)``

2. **Campbell ellipsoidal** (Campbell, 1986, 1990)
   - Parameter: alpha = mean leaf inclination angle (degrees)
   - Models the leaf population as oriented normals on an ellipsoid of
     revolution; a single parameter smoothly interpolates between planophile
     (alpha → 0) and erectophile (alpha → 90).  alpha ≈ 57 deg gives the
     spherical distribution.
   - In SCOPE-RTM: ``campbell_lidf(alpha)``

3. **Beta distribution** (Goel & Strebel, 1984)
   - Parameters: mu (mean), nu (variance)
   - Uses the Beta probability density to model any unimodal leaf-angle
     distribution.  Flexible but less physically interpretable.
   - Not yet in SCOPE-RTM — implemented here for comparison.

This experiment:
  (a) Visualizes each LAD as a polar histogram
  (b) Computes canopy directional reflectance (BRF) for a range of view
      zenith angles using 4SAIL, holding all other parameters constant
  (c) Tabulates the resulting extinction coefficients (ks, ko)
  (d) Saves comparison figures and a summary JSON

Usage::

    python examples/leaf_angle_distribution_comparison.py [--scope-root PATH] [--output-dir DIR]

References
----------
- Campbell, G.S. (1986). Extinction coefficients for radiation in plant
  canopies calculated using an ellipsoidal inclination angle distribution.
  Agricultural and Forest Meteorology, 36(4), 317-321.
  https://doi.org/10.1016/0168-1923(86)90010-9

- Campbell, G.S. (1990). Derivation of an angle density function for canopies
  with ellipsoidal leaf angle distributions. Agricultural and Forest
  Meteorology, 49(3), 173-176.
  https://doi.org/10.1016/0168-1923(90)90030-A

- Goel, N.S. & Strebel, D.E. (1984). Simple Beta distribution representation
  of leaf orientation in vegetation canopies. Agronomy Journal, 76(5), 800-802.
  https://doi.org/10.2134/agronj1984.00021962007600050021x

- Verhoef, W. (1998). Theory of radiative transfer models applied in optical
  remote sensing of vegetation canopies. Ph.D. thesis, Wageningen Agricultural
  University.

- de Wit, C.T. (1965). Photosynthesis of leaf canopies. Agricultural Research
  Reports, 663, Pudoc, Wageningen.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Leaf angle distribution implementations
# ---------------------------------------------------------------------------
from scope.canopy.foursail import (
    FourSAILModel,
    campbell_lidf,
    scope_lidf,
    scope_litab,
)
from scope.spectral.fluspect import FluspectModel
from scope.spectral.loaders import load_fluspect_resources

# ---- Beta distribution LAD (Goel & Strebel, 1984) -------------------------


def beta_lidf(
    mu: float,
    nu: float,
    n_elements: int = 13,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Leaf angle distribution based on the Beta density (Goel & Strebel, 1984).

    The Beta distribution is evaluated over the interval [0, pi/2] with shape
    parameters derived from the mean leaf angle *mu* (degrees) and a
    concentration parameter *nu* (larger = more peaked).

    Parameters
    ----------
    mu : float
        Mean leaf inclination angle in degrees (0 = planophile, 90 = erectophile).
    nu : float
        Concentration / peakedness.  Typical range 1-5.
        nu = 1 recovers a near-uniform distribution; nu = 4-5 is sharply peaked.
    n_elements : int
        Number of angular bins spanning 0-90 degrees.
    """
    device = device or torch.device("cpu")
    dtype = dtype or torch.float64

    # Convert mean angle to [0, 1] interval of the Beta distribution
    t_mean = mu / 90.0
    # Beta shape parameters:  alpha_b = t_mean * nu,  beta_b = (1 - t_mean) * nu
    alpha_b = max(t_mean * nu, 0.01)
    beta_b = max((1.0 - t_mean) * nu, 0.01)

    step = 90.0 / n_elements
    freq = torch.zeros(n_elements, dtype=dtype, device=device)
    for i in range(n_elements):
        t1 = i * step / 90.0
        t2 = (i + 1) * step / 90.0
        t_mid = 0.5 * (t1 + t2)
        # Beta density:  f(t) = t^(a-1) * (1-t)^(b-1) / B(a,b)
        # We skip the normalising constant because we renormalise at the end.
        freq[i] = t_mid ** (alpha_b - 1.0) * (1.0 - t_mid) ** (beta_b - 1.0)

    total = freq.sum()
    return freq / total.clamp(min=1e-30)


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

# Canonical Verhoef distribution types
VERHOEF_TYPES: dict[str, tuple[float, float]] = {
    "Planophile": (1.0, 0.0),
    "Erectophile": (-1.0, 0.0),
    "Plagiophile": (0.0, -1.0),
    "Extremophile": (0.0, 1.0),
    "Spherical": (-0.35, -0.15),
    "Uniform": (0.0, 0.0),
}

# Campbell parameterisations at representative mean leaf angles
CAMPBELL_ANGLES: dict[str, float] = {
    "alpha=10 (planophile)": 10.0,
    "alpha=30": 30.0,
    "alpha=57 (spherical)": 57.0,
    "alpha=70": 70.0,
    "alpha=85 (erectophile)": 85.0,
}

# Beta distribution parameterisations
BETA_PARAMS: dict[str, tuple[float, float]] = {
    "Beta mu=20, nu=3": (20.0, 3.0),
    "Beta mu=45, nu=2": (45.0, 2.0),
    "Beta mu=57, nu=4": (57.0, 4.0),
    "Beta mu=75, nu=3": (75.0, 3.0),
}


def _default_scope_root() -> str | None:
    candidate = Path(__file__).resolve().parents[1] / "upstream" / "SCOPE"
    return str(candidate) if candidate.exists() else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare leaf angle distribution models and their impact on canopy BRF."
    )
    parser.add_argument("--scope-root", help="Upstream SCOPE root (for spectral data).")
    parser.add_argument(
        "--output-dir",
        default="examples/output",
        help="Directory for output figures and JSON (default: examples/output).",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib figures.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core experiment functions
# ---------------------------------------------------------------------------


def build_all_distributions(device: torch.device, dtype: torch.dtype) -> dict[str, dict[str, object]]:
    """Build every LAD variant and return angle grids + weights.

    Returns a dict mapping name -> {angles, lidf, family, params}.
    """
    results: dict[str, dict[str, object]] = {}

    # 13-element litab for Verhoef distributions
    litab_13 = scope_litab(device=device, dtype=dtype)

    for name, (a, b) in VERHOEF_TYPES.items():
        lidf = scope_lidf(a, b, device=device, dtype=dtype)
        results[f"Verhoef: {name}"] = {
            "angles": litab_13,
            "lidf": lidf,
            "family": "Verhoef (1998)",
            "params": {"a": a, "b": b},
            "n_elements": 13,
        }

    for name, alpha in CAMPBELL_ANGLES.items():
        lidf = campbell_lidf(alpha, n_elements=13, device=device, dtype=dtype)
        results[f"Campbell: {name}"] = {
            "angles": litab_13,
            "lidf": lidf,
            "family": "Campbell (1990)",
            "params": {"alpha": alpha},
            "n_elements": 13,
        }

    for name, (mu, nu) in BETA_PARAMS.items():
        lidf = beta_lidf(mu, nu, n_elements=13, device=device, dtype=dtype)
        results[f"Goel-Strebel: {name}"] = {
            "angles": litab_13,
            "lidf": lidf,
            "family": "Goel & Strebel (1984)",
            "params": {"mu": mu, "nu": nu},
            "n_elements": 13,
        }

    return results


def compute_extinction_coefficients(
    distributions: dict[str, dict[str, object]],
    sail: FourSAILModel,
    tts: float = 30.0,
    tto: float = 0.0,
    psi: float = 0.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, dict[str, float]]:
    """Compute canopy-averaged extinction coefficients for each distribution."""
    device = device or torch.device("cpu")
    tts_t = torch.tensor([tts], device=device, dtype=dtype)
    tto_t = torch.tensor([max(tto, 0.01)], device=device, dtype=dtype)  # avoid /0
    psi_t = torch.tensor([psi], device=device, dtype=dtype)

    results = {}
    for name, dist in distributions.items():
        litab = dist["angles"]
        lidf = dist["lidf"].unsqueeze(0).to(device=device, dtype=dtype)  # (1, n_angles)
        litab = litab.to(device=device, dtype=dtype)
        ks, ko, bf, sob, sof = sail._weighted_sum_over_lidf(tts_t, tto_t, psi_t, lidf, litab)
        results[name] = {
            "ks": ks.item(),
            "ko": ko.item(),
            "bf": bf.item(),
            "sob": sob.item(),
            "sof": sof.item(),
        }
    return results


def compute_brf_vs_vza(
    distributions: dict[str, dict[str, object]],
    scope_root: str | None,
    tts: float = 30.0,
    lai: float = 3.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, dict[str, object]]:
    """Compute BRF at red (670nm) and NIR (800nm) for a sweep of view zenith angles.

    Requires scope_root for spectral data (leaf optics, soil).
    """
    device = device or torch.device("cpu")
    if scope_root is None:
        return {}

    # Load spectral resources
    resources = load_fluspect_resources(scope_root_path=scope_root, device=device, dtype=dtype)
    fluspect = FluspectModel(resources.spectral, resources.optipar, device=device, dtype=dtype)

    # Default leaf optics (single sample)
    from scope.spectral.fluspect import LeafBioBatch

    leaf = LeafBioBatch(
        Cab=torch.tensor([40.0], device=device, dtype=dtype),
        Cw=torch.tensor([0.01], device=device, dtype=dtype),
        Cdm=torch.tensor([0.01], device=device, dtype=dtype),
    )
    leaf_optics = fluspect(leaf)
    rho_leaf = leaf_optics.refl  # (1, nwl)
    tau_leaf = leaf_optics.tran  # (1, nwl)

    # Soil: flat 0.1 reflectance
    nwl = rho_leaf.shape[1]
    soil = torch.full((1, nwl), 0.1, device=device, dtype=dtype)

    # Find wavelength indices closest to 670nm and 800nm
    wl = resources.spectral.wlP.cpu().numpy()
    idx_red = int(np.argmin(np.abs(wl - 670)))
    idx_nir = int(np.argmin(np.abs(wl - 800)))

    vza_range = np.arange(-75, 76, 5, dtype=float)  # negative = backscatter
    results = {}

    for dist_name, dist in distributions.items():
        # Only run a representative subset to keep runtime manageable
        brf_red = []
        brf_nir = []

        for vza in vza_range:
            psi = 0.0 if vza >= 0 else 180.0
            abs_vza = abs(vza)
            if abs_vza < 0.5:
                abs_vza = 0.5  # avoid exact nadir singularity

            sail = FourSAILModel(n_angles=dist["n_elements"])
            out = sail(
                rho=rho_leaf,
                tau=tau_leaf,
                soil_refl=soil,
                lidf=dist["lidf"].unsqueeze(0),
                lai=torch.tensor([lai], device=device, dtype=dtype),
                hotspot=torch.tensor([0.05], device=device, dtype=dtype),
                tts=torch.tensor([tts], device=device, dtype=dtype),
                tto=torch.tensor([abs_vza], device=device, dtype=dtype),
                psi=torch.tensor([psi], device=device, dtype=dtype),
            )
            brf_red.append(out.rsot[0, idx_red].item())
            brf_nir.append(out.rsot[0, idx_nir].item())

        results[dist_name] = {
            "vza": vza_range.tolist(),
            "brf_red_670nm": brf_red,
            "brf_nir_800nm": brf_nir,
        }

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _savefig(fig, output_dir: Path, name: str) -> None:
    """Save figure as both PNG and SVG."""
    fig.savefig(output_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.svg", bbox_inches="tight")
    print(f"  Saved {output_dir / name}.png + .svg")


def plot_distributions(distributions: dict, output_dir: Path) -> None:
    """Create a Cartesian line plot of each LAD family (probability vs angle)."""
    try:
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots.", file=sys.stderr)
        return

    families = {}
    for name, dist in distributions.items():
        fam = dist["family"]
        families.setdefault(fam, []).append((name, dist))

    fig, axes = plt.subplots(1, len(families), figsize=(6 * len(families), 5), sharey=True)
    if len(families) == 1:
        axes = [axes]

    for ax, (fam_name, members) in zip(axes, families.items()):
        ax.set_title(fam_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Leaf inclination angle (deg)")

        colors = cm.tab10(np.linspace(0, 1, max(len(members), 1)))
        for (name, dist), color in zip(members, colors):
            angles_deg = dist["angles"].cpu().numpy()
            lidf = dist["lidf"].cpu().numpy()
            short_name = name.split(": ", 1)[-1]
            ax.fill_between(angles_deg, 0, lidf, alpha=0.12, color=color)
            ax.plot(angles_deg, lidf, "o-", color=color, label=short_name, markersize=3, linewidth=1.5)

        ax.set_xlim(0, 90)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Probability")
    fig.suptitle(
        "Leaf Angle Distribution Functions",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    _savefig(fig, output_dir, "lad_distributions")
    plt.close(fig)


def plot_brf_comparison(brf_results: dict, distributions: dict, output_dir: Path) -> None:
    """Plot BRF in the principal plane for representative distributions."""
    if not brf_results:
        return
    try:
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Select representative distributions for cleaner plots
    representative = [
        k
        for k in brf_results
        if any(
            tag in k
            for tag in [
                "Planophile",
                "Erectophile",
                "Spherical",
                "alpha=57",
                "alpha=10",
                "alpha=85",
                "mu=20",
                "mu=75",
            ]
        )
    ]
    if not representative:
        representative = list(brf_results.keys())[:8]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    colors = cm.tab10(np.linspace(0, 1, len(representative)))
    for name, color in zip(representative, colors):
        data = brf_results[name]
        vza = data["vza"]
        short = name.split(": ", 1)[-1]
        ax1.plot(vza, data["brf_red_670nm"], "-", color=color, label=short, linewidth=1.8)
        ax2.plot(vza, data["brf_nir_800nm"], "-", color=color, label=short, linewidth=1.8)

    for ax, band in [(ax1, "Red (670 nm)"), (ax2, "NIR (800 nm)")]:
        ax.set_xlabel("View Zenith Angle (deg)\n\u2190 Forward scatter     Backscatter \u2192")
        ax.set_ylabel("Bidirectional Reflectance Factor (BRF)")
        ax.set_title(f"Principal Plane BRF \u2014 {band}", fontweight="bold")
        ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Impact of Leaf Angle Distribution on Canopy Reflectance\n"
        "(LAI = 3, SZA = 30\u00b0, hotspot = 0.05, flat soil reflectance = 0.1)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    _savefig(fig, output_dir, "lad_brf_comparison")
    plt.close(fig)


def plot_extinction_coefficients(ext_results: dict, output_dir: Path) -> None:
    """Grouped bar chart of ks, ko, and bf for each distribution."""
    if not ext_results:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    names = list(ext_results.keys())
    short_names = [n.split(": ", 1)[-1] for n in names]
    ks_vals = [ext_results[n]["ks"] for n in names]
    ko_vals = [ext_results[n]["ko"] for n in names]
    bf_vals = [ext_results[n]["bf"] for n in names]

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(names))
    w = 0.25
    ax.bar(x - w, ks_vals, w, label="ks (solar extinction)", color="#e07b39")
    ax.bar(x, ko_vals, w, label="ko (observer extinction)", color="#3971e0")
    ax.bar(x + w, bf_vals, w, label="bf (bilamb. scattering)", color="#39b54a")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Coefficient value")
    ax.set_title(
        "Canopy Extinction and Scattering Coefficients by LAD\n(SZA = 30\u00b0, VZA = nadir)",
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, output_dir, "lad_extinction_coefficients")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    scope_root = args.scope_root or _default_scope_root()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    dtype = torch.float64

    print("=" * 70)
    print("Leaf Angle Distribution Comparison Experiment")
    print("=" * 70)

    # 1. Build all distributions
    print("\n[1/4] Building leaf angle distributions...")
    distributions = build_all_distributions(device, dtype)
    for name, dist in distributions.items():
        lidf = dist["lidf"].cpu().numpy()
        peak_angle = dist["angles"][lidf.argmax()].item()
        print(f"  {name:40s}  peak={peak_angle:5.1f}°  max_prob={lidf.max():.4f}")

    # 2. Compute extinction coefficients
    print("\n[2/4] Computing extinction coefficients (SZA=30°, VZA=nadir)...")
    sail = FourSAILModel(n_angles=13)
    ext_results = compute_extinction_coefficients(distributions, sail, device=device, dtype=dtype)
    print(f"  {'Distribution':40s}  {'ks':>8s}  {'ko':>8s}  {'bf':>8s}")
    print(f"  {'-' * 40}  {'-' * 8}  {'-' * 8}  {'-' * 8}")
    for name, vals in ext_results.items():
        print(f"  {name:40s}  {vals['ks']:8.4f}  {vals['ko']:8.4f}  {vals['bf']:8.4f}")

    # 3. BRF angular sweep (requires spectral data)
    print("\n[3/4] Computing principal-plane BRF sweep...")
    if scope_root:
        # Run BRF only for a representative subset to keep runtime reasonable
        representative_names = [
            n
            for n in distributions
            if any(
                tag in n
                for tag in [
                    "Planophile",
                    "Erectophile",
                    "Spherical",
                    "alpha=57",
                    "alpha=10",
                    "alpha=85",
                    "mu=20",
                    "mu=75",
                ]
            )
        ]
        representative_dists = {k: distributions[k] for k in representative_names}
        brf_results = compute_brf_vs_vza(representative_dists, scope_root, device=device, dtype=dtype)
        print(
            f"  Computed BRF for {len(brf_results)} distributions across"
            f" {len(next(iter(brf_results.values()))['vza'])} view angles"
        )
    else:
        print("  Skipped (no --scope-root or upstream/SCOPE directory found)")
        brf_results = {}

    # 4. Plots and output
    print("\n[4/4] Generating outputs...")
    if not args.no_plot:
        plot_distributions(distributions, output_dir)
        plot_extinction_coefficients(ext_results, output_dir)
        plot_brf_comparison(brf_results, distributions, output_dir)

    # JSON summary
    summary = {
        "experiment": "leaf_angle_distribution_comparison",
        "distributions": {
            name: {
                "family": str(dist["family"]),
                "params": dist["params"],
                "lidf": dist["lidf"].cpu().tolist(),
                "angles_deg": dist["angles"].cpu().tolist(),
            }
            for name, dist in distributions.items()
        },
        "extinction_coefficients": ext_results,
    }
    json_path = output_dir / "leaf_angle_distribution_comparison.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
