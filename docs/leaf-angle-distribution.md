# Leaf Angle Distribution Functions

The **leaf angle distribution** (LAD), also called the leaf inclination
distribution function (LIDF), describes the probability density of leaf-normal
zenith angles within a canopy.  It is one of the most influential structural
inputs to any turbid-medium radiative transfer model: it controls the canopy
extinction coefficients, the single-scattering phase function, and the
directional gap-fraction profile.

SCOPE-RTM provides two built-in parameterizations — the Verhoef two-parameter
bimodal and the Campbell ellipsoidal — and the experiment script demonstrates
a third (the Beta distribution of Goel & Strebel) for comparison.


## Why does the LAD matter?

In the 4SAIL radiative transfer equations the LAD enters through a single
integration step that averages the angle-dependent volume-scattering function
over all possible leaf orientations:

$$
G_s = \sum_{i} \frac{\chi_s(\theta_l^{(i)})}{\cos\theta_s}\, f(\theta_l^{(i)})
$$

where $\theta_l^{(i)}$ are the discrete leaf inclination bins, $\chi_s$ is the
Ross–Nilson G-function (projected area of a leaf in the solar direction), and
$f(\theta_l)$ is the LAD weight.  An analogous summation produces $G_o$ for
the observer direction.  These two coefficients ($k_s$, $k_o$) propagate into
every reflectance, transmittance, and gap-probability quantity computed by the
model.

**In practice** a planophile canopy (mostly horizontal leaves) has large
extinction at small zenith angles and produces high near-infrared reflectance,
whereas an erectophile canopy (mostly vertical leaves) transmits near-nadir
light deep into the canopy and produces lower reflectance with a weaker
hotspot.


## 1. Verhoef Two-Parameter Bimodal (1998)

### Theory

Verhoef (1998, ch. 3) proposed a two-parameter family that generates a cumulative
density function on $[0, \pi/2]$ through the implicit equation

$$
a\,\sin x + \tfrac{1}{2}\,b\,\sin 2x = y
$$

where $y = 2\theta/\pi$ (normalized angle), and $a$, $b$ are the shape
parameters.  The equation is solved iteratively (Newton-like half-step) for
each angular bin boundary, and the bin frequencies are obtained by differencing
the cumulative values.

When $a > 1$ the distribution degenerates to the *de Wit cosine* form
$F(\theta) = 1 - \cos\theta$, which is the spherical distribution in the
angle domain (not solid-angle domain).

**Six canonical types** cover the classical leaf-angle archetypes introduced
by de Wit (1965):

| Name          | a      | b      | Description                              |
|---------------|--------|--------|------------------------------------------|
| Planophile    |  1.0   |  0.0   | Horizontal leaves; peak at 0°            |
| Erectophile   | −1.0   |  0.0   | Vertical leaves; peak near 90°           |
| Plagiophile   |  0.0   | −1.0   | Oblique leaves; peak near 45°            |
| Extremophile  |  0.0   |  1.0   | Bimodal — very flat + very steep          |
| Spherical     | −0.35  | −0.15  | Uniform in solid-angle space              |
| Uniform       |  0.0   |  0.0   | Uniform in inclination-angle space        |

### SCOPE-RTM API

```python
from scope import scope_lidf

# Planophile distribution → 13-element tensor
lidf = scope_lidf(lidfa=1.0, lidfb=0.0)

# Spherical
lidf = scope_lidf(lidfa=-0.35, lidfb=-0.15)
```

The returned tensor has 13 elements corresponding to the angular bins centred
at {5, 15, 25, 35, 45, 55, 65, 75, 81, 83, 85, 87, 89}° — eight coarse
10° bins followed by five fine 2° bins.  The finer discretization near 90°
captures the rapid variation of near-vertical leaf populations.

### Reference

> Verhoef, W. (1998). *Theory of radiative transfer models applied in optical
> remote sensing of vegetation canopies.* Ph.D. thesis, Wageningen Agricultural
> University.
>
> de Wit, C.T. (1965). *Photosynthesis of leaf canopies.* Agricultural Research
> Reports 663, Pudoc, Wageningen.


## 2. Campbell Ellipsoidal (1986, 1990)

### Theory

Campbell (1986, 1990) models the leaf population as if the leaf normals were
distributed uniformly over the surface of an **ellipsoid of revolution**.  A
single parameter — the mean leaf inclination angle $\bar\alpha$ (degrees) —
controls the ellipsoid eccentricity:

$$
\varepsilon = \exp\!\bigl(-1.6184\times10^{-5}\,\bar\alpha^3
                        + 2.1145\times10^{-3}\,\bar\alpha^2
                        - 1.2390\times10^{-1}\,\bar\alpha
                        + 3.2491\bigr)
$$

The probability density for each angular bin $[\theta_1, \theta_2]$ is then
evaluated in three regimes:

| Eccentricity       | Ellipsoid shape     | Integration formula                      |
|--------------------|--------------------|------------------------------------------|
| $\varepsilon = 1$  | Sphere             | $\lvert\cos\theta_1 - \cos\theta_2\rvert$ |
| $\varepsilon > 1$  | Oblate (flat)      | Hyperbolic + logarithmic terms           |
| $\varepsilon < 1$  | Prolate (elongated)| Elliptic + arcsin terms                  |

The key property is that a **single, physically interpretable parameter**
(mean leaf angle) smoothly interpolates between planophile ($\bar\alpha\to 0$)
and erectophile ($\bar\alpha\to 90$), with $\bar\alpha \approx 57°$
recovering the spherical distribution.

### SCOPE-RTM API

```python
from scope import campbell_lidf

# Spherical distribution with 13 bins (to match SCOPE litab)
lidf = campbell_lidf(alpha=57.0, n_elements=13)

# Planophile with the default 18-bin discretization
lidf = campbell_lidf(alpha=10.0)
```

The `n_elements` parameter controls the number of uniform bins spanning 0–90°.
When used with `FourSAILModel`, set `n_elements` to match the model's
`n_angles` constructor argument (default 13 for SCOPE, 18 for PROSAIL
convention).

### Reference

> Campbell, G.S. (1986). Extinction coefficients for radiation in plant canopies
> calculated using an ellipsoidal inclination angle distribution.
> *Agricultural and Forest Meteorology*, 36(4), 317–321.
> [doi:10.1016/0168-1923(86)90010-9](https://doi.org/10.1016/0168-1923(86)90010-9)
>
> Campbell, G.S. (1990). Derivation of an angle density function for canopies
> with ellipsoidal leaf angle distributions. *Agricultural and Forest
> Meteorology*, 49(3), 173–176.
> [doi:10.1016/0168-1923(90)90030-A](https://doi.org/10.1016/0168-1923(90)90030-A)


## 3. Beta Distribution (Goel & Strebel, 1984)

### Theory

Goel and Strebel (1984) proposed using the **Beta probability density** to
represent leaf angle distributions.  The Beta density on the normalized
interval $t = \theta / (π/2)$ is

$$
f(t;\,\mu_b,\,\nu_b) = \frac{t^{\mu_b - 1}\,(1 - t)^{\nu_b - 1}}
                              {B(\mu_b,\,\nu_b)}
$$

where $\mu_b$ and $\nu_b$ are shape parameters derived from the desired mean
leaf angle $\mu$ and a concentration parameter $\nu$:

$$
\mu_b = \frac{\mu}{90}\,\nu, \qquad
\nu_b = \left(1 - \frac{\mu}{90}\right)\nu
$$

**Advantages:**

- Can represent any unimodal distribution, including skewed and narrow peaks
- Mathematically well understood, with closed-form moments

**Disadvantages:**

- Two parameters ($\mu$, $\nu$) are less physically intuitive than Campbell's
  single mean-angle
- Cannot naturally produce the bimodal extremophile shape that Verhoef's
  parameterization supports
- Not commonly used in operational retrieval algorithms

### Experiment API

The Beta LAD is implemented in the comparison script but is not part of the
core SCOPE-RTM library:

```python
from examples.leaf_angle_distribution_comparison import beta_lidf

lidf = beta_lidf(mu=45.0, nu=3.0, n_elements=13)
```

### Reference

> Goel, N.S. & Strebel, D.E. (1984). Simple Beta distribution representation
> of leaf orientation in vegetation canopies. *Agronomy Journal*, 76(5),
> 800–802.
> [doi:10.2134/agronj1984.00021962007600050021x](https://doi.org/10.2134/agronj1984.00021962007600050021x)


## Comparison: which LAD to use?

| Criterion                          | Verhoef        | Campbell       | Beta           |
|------------------------------------|----------------|----------------|----------------|
| Number of parameters               | 2 (a, b)       | 1 (α)          | 2 (μ, ν)      |
| Bimodal distributions              | Yes            | No             | No             |
| Physical interpretability          | Moderate       | High           | Low            |
| PROSAIL compatibility              | Yes (type 1)   | Yes (type 2)   | No             |
| Canonical archetypes (de Wit)      | Direct mapping | Via α only     | Indirect       |
| Inversion / retrieval friendliness | Good           | Best (1 param) | Fair           |

**Recommendations:**

- **For PROSAIL intercomparison** or retrieval studies: use `campbell_lidf`
  with `n_elements=18` and the PROSAIL litab grid.  This matches the standard
  PROSAIL parameterization (`TypeLidf=2`) exactly.

- **For SCOPE workflows** or when you need bimodal leaf populations: use
  `scope_lidf` with the canonical Verhoef parameters.  The 13-element
  discretization matches the upstream SCOPE MATLAB code.

- **For research** exploring novel LAD shapes: the Beta distribution offers
  maximum flexibility for unimodal distributions but requires custom
  integration into the model.


## Interoperability between SCOPE and PROSAIL

The `FourSAILModel` accepts any LAD as a `(batch, n_angles)` tensor, so it is
agnostic to the generation method.  The only constraint is that the model's
`n_angles` constructor argument must match the length of the LIDF vector:

```python
from scope import FourSAILModel, campbell_lidf, scope_lidf, scope_litab

# PROSAIL-style: 18 bins
lidf_prosail = campbell_lidf(alpha=57.0, n_elements=18)
sail_prosail = FourSAILModel(n_angles=18)

# SCOPE-style: 13 bins
lidf_scope = scope_lidf(lidfa=-0.35, lidfb=-0.15)
sail_scope = FourSAILModel(n_angles=13)

# Both models can also use the other's LIDF by matching n_elements:
lidf_campbell_13 = campbell_lidf(alpha=57.0, n_elements=13)
# Use with the default FourSAILModel(n_angles=13) — works seamlessly
```


## Running the comparison experiment

A self-contained comparison script is provided at
`examples/leaf_angle_distribution_comparison.py`:

```bash
# Basic run (prints tables, saves JSON)
python examples/leaf_angle_distribution_comparison.py --no-plot

# Full run with figures (requires matplotlib)
python examples/leaf_angle_distribution_comparison.py --scope-root upstream/SCOPE

# Custom output directory
python examples/leaf_angle_distribution_comparison.py --output-dir results/lad/
```

The script produces:

1. **`lad_distributions.png`** — Polar histogram of all LAD variants grouped
   by family
2. **`lad_extinction_coefficients.png`** — Bar chart comparing $k_s$ and $k_o$
   across distributions
3. **`lad_brf_comparison.png`** — Principal-plane BRF at 670 nm and 800 nm
   for representative distributions
4. **`leaf_angle_distribution_comparison.json`** — Machine-readable summary of
   all distributions, parameters, and extinction coefficients
