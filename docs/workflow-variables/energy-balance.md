# Energy Balance Variables

This page is generated from the in-repo variable registry for the `energy-balance` workflow family.

Use `scope vars --workflow energy-balance` for terminal lookup.

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

## Meteorology

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `Ta` | degC | Air temperature. | - | docs/input-output-reference.md | - |
| `ea` | hPa | Ambient vapor pressure. | - | docs/input-output-reference.md | - |
| `Ca` | ppm | Ambient CO2 concentration. | - | docs/input-output-reference.md | - |
| `Oa` | permil or mmol mol-1 | Ambient oxygen concentration used by leaf biochemistry. | - | docs/input-output-reference.md | - |
| `p` | hPa | Air pressure. | - | docs/input-output-reference.md | - |
| `z` | m | Reference meteorological height. | - | docs/input-output-reference.md | - |
| `u` | m s-1 | Wind speed at the reference height. | - | docs/input-output-reference.md | - |
| `L` | m | Monin-Obukhov length when supplied as an external stability input. | - | docs/input-output-reference.md | - |

## Canopy Aerodynamics

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `Cd` | - | Drag coefficient for canopy aerodynamic exchange. | - | docs/input-output-reference.md | - |
| `rwc` | - | Relative water content or canopy resistance scaling parameter used by the energy-balance closure. | - | docs/input-output-reference.md | - |
| `z0m` | m | Momentum roughness length. | - | docs/input-output-reference.md | - |
| `d` | m | Zero-plane displacement height. | - | docs/input-output-reference.md | - |
| `h` | m | Canopy height. | - | docs/input-output-reference.md | - |
| `kV` | - | Vertical extinction or partitioning parameter for layered canopy coupling. | - | docs/input-output-reference.md | - |
| `fV` | - | Vertical partitioning profile used by layered canopy physiology and energy balance. | - | docs/input-output-reference.md | - |

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

## Boundary State

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `Csu` | ppm | CO2 concentration at the sunlit leaf surface. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Csh` | ppm | CO2 concentration at the shaded leaf surface. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `ebu` | hPa | Vapor pressure at the sunlit leaf surface. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `ebh` | hPa | Vapor pressure at the shaded leaf surface. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Csu` | ppm | Solved CO2 concentration at the sunlit leaf surface. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Csh` | ppm | Solved CO2 concentration at the shaded leaf surface. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `ebu` | hPa | Solved vapor pressure at the sunlit leaf surface. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `ebh` | hPa | Solved vapor pressure at the shaded leaf surface. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |

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

## Physiology

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `Pnu_Cab` | umol m-2 s-1 | Sunlit absorbed PAR by chlorophyll a+b used to drive leaf biochemistry. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Pnh_Cab` | umol m-2 s-1 | Shaded absorbed PAR by chlorophyll a+b used to drive leaf biochemistry. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `sunlit_*` | varies | Prefix used for leaf-biochemistry outputs of sunlit leaves. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | Search for base names such as A, Ci, rcw, eta, or Ja. |
| `shaded_*` | varies | Prefix used for leaf-biochemistry outputs of shaded leaves. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | Search for base names such as A, Ci, rcw, eta, or Ja. |
| `sunlit_A` | umol m-2 s-1 | Net assimilation rate of sunlit leaves. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `shaded_A` | umol m-2 s-1 | Net assimilation rate of shaded leaves. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `sunlit_Ci` | ppm | Intercellular CO2 concentration of sunlit leaves. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `shaded_Ci` | ppm | Intercellular CO2 concentration of shaded leaves. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `sunlit_rcw` | s m-1 | Canopy water-vapor resistance of sunlit leaves. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `shaded_rcw` | s m-1 | Canopy water-vapor resistance of shaded leaves. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `sunlit_eta` | - | Fluorescence efficiency term returned by the sunlit leaf biochemistry solve. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `shaded_eta` | - | Fluorescence efficiency term returned by the shaded leaf biochemistry solve. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |

## Energy Balance

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `Rnctot` | W m-2 | Net radiation of the canopy. | Rnctot = Rnuc + Rnhc | https://scope-model.readthedocs.io/en/master/outfiles.html | fluxes.csv |
| `lEctot` | W m-2 | Latent heat flux of the canopy (transpiration). | lEctot = lEcu + lEch | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Hctot` | W m-2 | Sensible heat flux of the canopy. | Hctot = Hcu + Hch | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Actot` | umol m-2 s-1 | Net photosynthesis of the canopy. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | Photosynthesis |
| `Tcave` | degC | Average canopy temperature. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Rnstot` | W m-2 | Net radiation of the soil. | Rnstot = Rnus + Rnhs | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `lEstot` | W m-2 | Latent heat flux of the soil (evaporation). | lEstot = lEsu + lEsh | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Hstot` | W m-2 | Sensible heat flux of the soil. | Hstot = Hsu + Hsh | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Gtot` | W m-2 | Soil heat flux. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Tsave` | degC | Average soil temperature. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Rntot` | W m-2 | Total net radiation. | Rntot = Rnctot + Rnstot | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `lEtot` | W m-2 | Total latent heat flux. | lEtot = lEctot + lEstot | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Htot` | W m-2 | Total sensible heat flux. | Htot = Hctot + Hstot | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Rnuc_sw` | W m-2 | Sunlit canopy net shortwave radiation. | Shortwave component of Rnuc. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Rnhc_sw` | W m-2 | Shaded canopy net shortwave radiation. | Shortwave component of Rnhc. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Rnus_sw` | W m-2 | Sunlit soil net shortwave radiation. | Shortwave component of Rnus. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Rnhs_sw` | W m-2 | Shaded soil net shortwave radiation. | Shortwave component of Rnhs. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Rnuct` | W m-2 | Sunlit canopy net thermal radiation. | Thermal component of Rnuc. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Rnhct` | W m-2 | Shaded canopy net thermal radiation. | Thermal component of Rnhc. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Rnust` | W m-2 | Sunlit soil net thermal radiation. | Thermal component of Rnus. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Rnhst` | W m-2 | Shaded soil net thermal radiation. | Thermal component of Rnhs. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Rnuc` | W m-2 | Total net radiation of the sunlit canopy fraction. | Rnuc = Rnuc_sw + Rnuct | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Rnhc` | W m-2 | Total net radiation of the shaded canopy fraction. | Rnhc = Rnhc_sw + Rnhct | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Rnus` | W m-2 | Total net radiation of the sunlit soil fraction. | Rnus = Rnus_sw + Rnust | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Rnhs` | W m-2 | Total net radiation of the shaded soil fraction. | Rnhs = Rnhs_sw + Rnhst | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `lEcu` | W m-2 | Latent heat flux of the sunlit canopy fraction. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `lEch` | W m-2 | Latent heat flux of the shaded canopy fraction. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `lEsu` | W m-2 | Latent heat flux of the sunlit soil fraction. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `lEsh` | W m-2 | Latent heat flux of the shaded soil fraction. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Hcu` | W m-2 | Sensible heat flux of the sunlit canopy fraction. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Hch` | W m-2 | Sensible heat flux of the shaded canopy fraction. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Hsu` | W m-2 | Sensible heat flux of the sunlit soil fraction. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Hsh` | W m-2 | Sensible heat flux of the shaded soil fraction. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Gsu` | W m-2 | Soil heat flux associated with the sunlit soil fraction. | Sunlit-soil contribution to Gtot. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Gsh` | W m-2 | Soil heat flux associated with the shaded soil fraction. | Shaded-soil contribution to Gtot. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |

## Resistance

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `raa` | s m-1 | Aerodynamic resistance above the canopy. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `raws` | s m-1 | Aerodynamic resistance within the soil/canopy lower boundary layer. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | within-soil aerodynamic resistance |
| `rawc` | s m-1 | Aerodynamic resistance within the canopy air space. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `rac` | s m-1 | Boundary-layer resistance for canopy heat and vapor exchange. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `ras` | s m-1 | Boundary-layer resistance for soil heat and vapor exchange. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `ustar` | m s-1 | Friction velocity. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |

## Solver

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `max_error` | W m-2 | Maximum residual or closure error in the final energy-balance iteration. | - | docs/variable-glossary.md | - |
| `converged` | 0/1 | Whether the coupled energy-balance iteration converged. | - | docs/variable-glossary.md | - |
| `counter` | - | Number of energy-balance iterations performed. | - | docs/variable-glossary.md | - |

## Structured Outputs

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `sunlit` | LeafBiochemistryResult | Nested structured result holding sunlit leaf-biochemistry outputs. | - | docs/variable-glossary.md | - |
| `shaded` | LeafBiochemistryResult | Nested structured result holding shaded leaf-biochemistry outputs. | - | docs/variable-glossary.md | - |

## Solver Diagnostics

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `sunlit_Cs_input` | ppm | CO2 boundary condition passed into the sunlit leaf solve during the final energy-balance iteration. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | Phase-lagged diagnostic; not the main same-state parity signal. |
| `shaded_Cs_input` | ppm | CO2 boundary condition passed into the shaded leaf solve during the final energy-balance iteration. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | Phase-lagged diagnostic; not the main same-state parity signal. |
| `sunlit_eb_input` | hPa | Leaf-surface vapor-pressure boundary condition passed into the sunlit leaf solve. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | Phase-lagged diagnostic; not the main same-state parity signal. |
| `shaded_eb_input` | hPa | Leaf-surface vapor-pressure boundary condition passed into the shaded leaf solve. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | Phase-lagged diagnostic; not the main same-state parity signal. |
| `sunlit_T_input` | degC | Leaf temperature passed into the sunlit leaf solve during the final iteration. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | Phase-lagged diagnostic; not the main same-state parity signal. |
| `shaded_T_input` | degC | Leaf temperature passed into the shaded leaf solve during the final iteration. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | Phase-lagged diagnostic; not the main same-state parity signal. |
