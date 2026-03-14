# Reflectance Variables

This page is generated from the in-repo variable registry for the `reflectance` workflow family.

Use `scope vars --workflow reflectance` for terminal lookup.

## Geometry

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `direction` | - | Index over directional viewing geometries. | - | docs/input-output-reference.md | - |
| `directional_tto` | deg | Viewing zenith angles for directional outputs. | - | docs/input-output-reference.md | tto grid |
| `directional_psi` | deg | Relative azimuth angles for directional outputs. | - | docs/input-output-reference.md | psi grid |
| `tts` | deg | Solar zenith angle. | - | docs/input-output-reference.md | sun zenith |
| `tto` | deg | Viewing zenith angle. | - | docs/input-output-reference.md | observer zenith |
| `psi` | deg | Relative azimuth angle between sun and sensor. | - | docs/input-output-reference.md | relative azimuth |

## Leaf Biochemistry

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `Cab` | ug cm-2 | Leaf chlorophyll a+b content. | - | docs/input-output-reference.md | reflectance, fluorescence, thermal, energy balance; chlorophyll |
| `Cca` | ug cm-2 | Leaf carotenoid content. | - | docs/input-output-reference.md | carotenoids |
| `Cw` | g cm-2 | Equivalent leaf water thickness. | - | docs/input-output-reference.md | leaf water |
| `Cdm` | g cm-2 | Leaf dry matter content per area. | - | docs/input-output-reference.md | Cm, dry matter |
| `Cs` | - | Senescent or brown pigment content used by FLUSPECT. | - | docs/input-output-reference.md | cbrown |
| `Cant` | ug cm-2 | Anthocyanin content. | - | docs/input-output-reference.md | - |
| `Cbc` | ug cm-2 | Brown carbon or additional biochemical absorber term when provided. | - | docs/input-output-reference.md | - |
| `Cp` | g cm-2 | Protein content parameter for extended leaf optics. | - | docs/input-output-reference.md | - |
| `N` | - | Leaf mesophyll structure parameter in PROSPECT/FLUSPECT. | - | docs/input-output-reference.md | - |
| `fqe` | - | Leaf fluorescence quantum efficiency scaling used to build fluorescence source terms. | - | docs/input-output-reference.md | fluorescence efficiency |

## Canopy Structure

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `ala` | deg | Mean leaf angle parameter used to define the canopy leaf angle distribution function (LIDF). | Controls the overall canopy inclination distribution used by the LIDF. | docs/input-output-reference.md | LIDFa |
| `LAI` | m2 m-2 | Leaf area index. | - | docs/input-output-reference.md | - |

## Soil

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `soil_refl` | - | Explicit soil reflectance spectrum on the model optical wavelength grid. | - | docs/input-output-reference.md | - |
| `soil_spectrum` | index | Index into the upstream SCOPE soil reflectance library. | - | docs/input-output-reference.md | soilspectrum |
| `BSMBrightness` | - | Brightness parameter for the SCOPE BSM soil model. | - | docs/input-output-reference.md | - |
| `BSMlat` | deg | Latitude-like mineral composition parameter used by the SCOPE BSM soil model. | - | docs/input-output-reference.md | - |
| `BSMlon` | deg | Longitude-like mineral composition parameter used by the SCOPE BSM soil model. | - | docs/input-output-reference.md | - |
| `SMC` | fraction | Soil moisture content used by the BSM soil model. | - | docs/input-output-reference.md | - |
| `rss` | s m-1 | Soil resistance to evaporation. | - | docs/input-output-reference.md | soil surface resistance |
| `rbs` | s m-1 | Boundary resistance near the soil surface. | - | docs/input-output-reference.md | - |
| `GAM` | W m-2 | Optional soil heat flux forcing or initialization term. | - | docs/input-output-reference.md | - |
| `Tsold` | degC | Previous-step soil temperature state for transient soil heat treatment. | - | docs/input-output-reference.md | - |
| `dt_seconds` | s | Time-step length used by transient soil heat treatment. | - | docs/input-output-reference.md | - |

## Reflectance

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `leaf_refl` | - | Leaf hemispherical reflectance from FLUSPECT. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `leaf_tran` | - | Leaf hemispherical transmittance from FLUSPECT. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `rsot` | - | Total top-of-canopy reflectance factor in the observation direction. | rsot = rsost + rsodt | https://scope-model.readthedocs.io/en/master/outfiles.html | apparent reflectance, reflectance.csv |
| `rso` | - | Bidirectional reflectance factor. | rso = rsos + rsod | https://scope-model.readthedocs.io/en/master/outfiles.html | BRF; Matches the original SCOPE rso definition. |
| `rsd` | - | Directional-hemispherical reflectance factor. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | DHRF |
| `rdd` | - | Bi-hemispherical reflectance factor. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | BHRF |
| `rdo` | - | Hemispherical-directional reflectance factor. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | HDRF |
| `refl_` | - | Directional apparent reflectance for a requested angle grid. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `tdd` | - | Diffuse transmittance for diffuse incident radiation through the canopy. | Diffuse in -> diffuse out canopy transmittance term. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `tsd` | - | Direct-to-diffuse canopy transmittance for solar illumination. | Direct sun -> diffuse canopy transmittance term. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `tdo` | - | Diffuse transmittance from canopy layers toward the observation direction. | Diffuse canopy field -> observation-direction transmittance term. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `rsos` | - | Bidirectional reflectance contribution from the hotspot or single-scattering term. | Direct/hotspot contribution to rso. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `rsod` | - | Bidirectional reflectance contribution from the diffuse multiple-scattering term. | Diffuse multiple-scattering contribution to rso. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `rddt` | - | Bi-hemispherical reflectance including the soil boundary condition. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `rsdt` | - | Directional-hemispherical reflectance including the soil boundary condition. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `rdot` | - | Hemispherical-directional reflectance including the soil boundary condition. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `rsodt` | - | Diffuse part of total bidirectional reflectance including the soil boundary condition. | Diffuse multiple-scattering part of rsot. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `rsost` | - | Hotspot or direct part of total bidirectional reflectance including the soil boundary condition. | Direct/hotspot part of rsot. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `tss` | - | Direct solar transmittance through the canopy. | Beer-Lambert-like direct sun transmission term. | https://scope-model.readthedocs.io/en/master/outfiles.html | sun transmittance |
| `too` | - | Direct transmittance along the observation path. | Beer-Lambert-like observation-path transmission term. | https://scope-model.readthedocs.io/en/master/outfiles.html | observer transmittance |
| `tsstoo` | - | Joint direct-transmittance term for the sun-observer hotspot path. | Combined direct sun and observation-path transmission term. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `rso_` | - | Directional bidirectional reflectance factor on the requested angle grid. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | brdf_, directional BRDF |

## Profiles

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `Ps` | - | Cumulative direct-solar interception profile along canopy depth. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Po` | - | Cumulative observation-path attenuation profile along canopy depth. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Pso` | - | Joint sun-observer hotspot weighting profile across the canopy. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Es_` | W m-2 um-1 | Downward direct-plus-diffuse shortwave flux profile at canopy interfaces. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Emin_` | W m-2 um-1 | Downward diffuse shortwave flux profile at canopy interfaces. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Eplu_` | W m-2 um-1 | Upward diffuse shortwave flux profile at canopy interfaces. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `layer_thermal_upward` | W m-2 | Integrated upward thermal transport per canopy layer. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Es_direct_` | W m-2 um-1 | Direct shortwave irradiance profile at canopy interfaces. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Emin_direct_` | W m-2 um-1 | Downward shortwave flux profile originating from direct illumination. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Eplu_direct_` | W m-2 um-1 | Upward shortwave flux profile originating from direct illumination. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Es_diffuse_` | W m-2 um-1 | Diffuse shortwave irradiance profile at canopy interfaces. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Emin_diffuse_` | W m-2 um-1 | Downward diffuse shortwave flux profile at canopy interfaces. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Eplu_diffuse_` | W m-2 um-1 | Upward diffuse shortwave flux profile at canopy interfaces. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `layer_fluorescence` | W m-2 | Upward fluorescence contribution aggregated per canopy layer. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | layer_fluorescence.dat |
