# Legacy TM5 convection was attached upside down

Date: 2026-10-03.

## Finding

`scripts/preprocessing/attach_catrine_tm5_convection_cs.jl` attaches the
legacy CATRINE 1-degree, 3-hourly TM5 convection files
(`convec_YYYYMMDD_00p03.nc`, variables `eu`, `du`, `ed`, `dd`) to cubed-sphere
transport binaries. The files store the 137 ERA5 levels surface first: level 1
has `b = 0.998815`, and the file's `ap`/`b` equal the ERA5 L137 layer
midpoints reversed, with zero difference, in all 4,383 files for 2014-2025.
The transport binary's `merge_map` and the runtime TM5 operator use k = 1 at
the top of the atmosphere (`src/Operators/Convection/TM5Convection.jl`). The
script summed file levels through `merge_map` without reversing them.

Measured in attached binaries written before the fix:

| Quantity | Before the fix | After the fix |
|---|---|---|
| Updraft entrainment peak level (of 66) | 3 (about 0.3 hPa) | 56 (about 960 hPa) |
| Updraft entrainment mean level (of 66) | 4.0 | 51.5 to 52.3 |
| Fossil CO2 above about 870 hPa after 24 h, C90, 2014-01-01 | 11.2% | 16.5% |

In an independent single-column check, one hour of TM5 convection lifted 51%
of a bottom-layer tracer with the correct orientation and 0.06% with the
inverted fields. Column closure (entrainment equals detrainment), horizontal
placement, units, sign, and time slots were correct before and after the fix;
the closure check cannot detect an inversion.

## Affected products

- `~/data/AtmosTransport/met/era5/n320_to_c90/transport_binary_v4_l66_f32_tm5_convection_1deg_3hour`
  (2018-12-01 to 2019-12-31) and the run `catrine_c90_2019_rt_co2` built on it.
- `/temp2/catrine-runs/met/era5_c30_2021_fullphysics_experimental`,
  `era5_c30_2022_fullphysics_experimental`, and the combined
  `era5_c30_dec2021_through_2022_fullphysics_experimental`, plus the CATRINE
  C30 full-physics runs built on them.

## Fix

The script derives the file's level order from `ap`/`b`, checks the 1-degree
grid, the date pairing, and the three-hour slots (from `timevalues_bounds`,
because `time` decodes to the 1880s in the 2022-04-30 to 2023-12-31 files), and
fails when the updraft entrainment's mean level is in the upper half of the
column or column closure exceeds 1e-4. Outputs are tagged
`legacy_catrine_1deg_3hour_tm5_to_cs_v3` and go to new `_v3` folders.

## Known remaining difference

The legacy fluxes are moist-air mass fluxes; the binaries are dry basis. The
native ERA5 path has the same gap. The relative error is about q/(1 - q),
estimated at up to 2% in the tropical boundary layer.
