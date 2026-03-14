# Thermal Variables

This page is generated from the in-repo variable registry for the `thermal` workflow family.

Use `scope vars --workflow thermal` for terminal lookup.

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

## Canopy Structure

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `ala` | deg | Mean leaf angle parameter used to define the canopy leaf angle distribution function (LIDF). | Controls the overall canopy inclination distribution used by the LIDF. | docs/input-output-reference.md | LIDFa |
| `LAI` | m2 m-2 | Leaf area index. | - | docs/input-output-reference.md | - |

## Thermal Optics

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `rho_thermal` | - | Thermal reflectance of the soil surface. | - | docs/input-output-reference.md | - |
| `tau_thermal` | - | Thermal transmittance of the canopy in the simplified thermal optics parameterization. | - | docs/input-output-reference.md | - |
| `rs_thermal` | - | Thermal soil reflectance/emissivity control in the simplified thermal optics parameterization. | - | docs/input-output-reference.md | - |

## Thermal State

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `Tcu` | degC | Sunlit canopy temperature. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Tch` | degC | Shaded canopy temperature. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Tsu` | degC | Sunlit soil temperature. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Tsh` | degC | Shaded soil temperature. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Tcu` | degC | Solved sunlit canopy temperature. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Tch` | degC | Solved shaded canopy temperature. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Tsu` | degC | Solved sunlit soil temperature. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Tsh` | degC | Solved shaded soil temperature. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |

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

## Thermal

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `Lot_` | W m-2 um-1 sr-1 | Thermal radiance spectrum in the observation direction. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Eoutte_` | W m-2 um-1 | Emitted longwave flux spectrum leaving the canopy-soil system. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Emint_` | W m-2 um-1 | Downward internal thermal flux profile. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Eplut_` | W m-2 um-1 | Upward internal thermal flux profile. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `LotBB_` | W m-2 um-1 sr-1 | Blackbody-equivalent observation-direction thermal radiance. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Loutt` | W m-2 um-1 sr-1 | Integrated observation-direction thermal radiance scalar. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Eoutt` | W m-2 | Integrated hemispherical thermal flux scalar. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `BrightnessT` | K | Brightness temperature inferred from the directional thermal radiance. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `canopyemis` | - | Effective canopy emissivity. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
