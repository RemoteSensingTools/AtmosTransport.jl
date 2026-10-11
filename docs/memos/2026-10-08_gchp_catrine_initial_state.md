# GCHP CATRINE C90 run: initial state shifted by half a layer (2026-10-08)

**Finding.** GCHP's CATRINE C90 run did not start from the protocol initial
state as defined by its pressure levels. Its first output (2021-12-01 03z)
equals the protocol CO₂ and SF₆ fields with each LMDz layer's value placed at
the layer's **top interface** pressure, instead of at the middle of the layer.
The whole column is shifted up by half a layer.

## Data

- Protocol initial state: `startCO2_202112010000.nc` and
  `startSF6_202112010000.nc` (LSCE, LMDz posterior on 1°, 79 hybrid layers,
  `ap`/`bp`/`Psurf`, stored surface first, values at mid-layer).
- GCHP: `GEOSChem.CATRINE_inst.20211201_0300z.nc4` (`SpeciesConcVV_CO2`,
  `SpeciesConcVV_SF6`, `Met_PMIDDRY`, `Met_AREAM2`), from
  https://data.nas.nasa.gov/catrine/CATRINE_C90/instant/.

## Method

Global area-weighted means on pressure surfaces:
- GCHP: each column interpolated linearly in log-pressure to the target
  pressure, using `Met_PMIDDRY`.
- Protocol file: the same interpolation, with the layer values placed at
  three trial pressures: the mid-layer pressure, the bottom interface
  (`ap[k] + bp[k] Psurf`) or the top interface (`ap[k+1] + bp[k+1] Psurf`).

Scripts and output are in `/temp1/cfranken/scratch/overnight_2026_10_07/`:
`ic_source_check{,_sf6}.{py,txt}` and
`ic_halflayer_check{,_trop}.{py,txt}`.

## Result

GCHP minus protocol file, CO₂ in ppm:

| p (hPa) | mid-layer | bottom interface | top interface |
|---|---|---|---|
| 900 | +0.071 | +0.107 | +0.036 |
| 500 | +0.026 | +0.048 | +0.004 |
| 200 | +0.127 | +0.237 | +0.022 |
| 150 | +0.187 | +0.384 | −0.005 |
| 100 | +0.477 | +0.978 | +0.006 |
| 70 | +0.620 | +1.169 | +0.018 |
| 50 | +0.523 | +0.883 | +0.017 |
| 30 | +0.122 | +0.200 | +0.008 |

SF₆, ppt:

| p (hPa) | mid-layer | top interface |
|---|---|---|
| 200 | +0.015 | +0.002 |
| 100 | +0.048 | +0.001 |
| 70 | +0.071 | +0.002 |
| 50 | +0.064 | +0.001 |

- AtmosTransport's own initial state matches the file at mid-layer pressures
  to within 0.05 ppm CO₂ and 0.009 ppt SF₆ (`ic_source_check*.txt`).
- Mapping the file by altitude (`height_above_reference_ellipsoid` against
  GCHP's `Height_asl`) does not reproduce GCHP (`ic_height_check.txt`).
- The offset is largest in the lower stratosphere, where CO₂ and SF₆ fall
  steeply with height and the LMDz layers are about 16 % apart in pressure
  (interfaces at 62.2, 53.4 and 45.8 hPa).
  At 70 hPa GCHP holds the protocol value from about 76 hPa: the profile is
  moved up by about 6 hPa, which makes GCHP's lower stratosphere look
  younger at t = 0.

## Consequence for model intercomparison

The initial difference decays as stratospheric air is replaced, and the decay
looks like a drift between the models. AtmosTransport (MERRA-2 C90,
`hybrid_mass` closure, FV3 vertical profile) compared with GCHP over
December 2021 – March 2022 (`compare_c90_gcic_test/`):

| | from the protocol state | from GCHP's 03z state |
|---|---|---|
| CO₂ bias above 100 hPa at 03z | −0.363 ppm | +0.001 ppm |
| growth of that bias (last 10 d − first 10 d) | +0.140 ppm | +0.050 ppm |
| SF₆ growth above 100 hPa | +0.018 ppt | +0.006 ppt |

Transport comparisons with the GCHP CATRINE C90 run should therefore start
from GCHP's own first output, or allow for this offset.

## Addendum: near-surface state (2026-10-11)

The half-layer placement does not explain most of GCHP's near-surface excess.
GCHP's 00z state, estimated as its archived 03z state minus AtmosTransport's
own three-hour tendency, is about **+0.25 ppm** above the protocol file in the
bottom 10% of column air mass. Placing file values at layer tops accounts for
only +0.032 ppm globally (about +0.082 ppm at 30–60°N and +0.139 ppm at
60–90°N). Of the tested reconstructions, top-interface values on GCHP's
moist-pressure levels are closest, but still leave about +0.09 ppm unexplained;
the moist-pressure interpretation is therefore suggestive, not established.

A separate AtmosTransport error had obscured this comparison. File ICs were
point-interpolated in absolute pressure using the file's `Psurf` for source
levels and the binary's pressure for target levels. Changing the dry-mass pin
therefore moved the initialized near-surface CO₂ by −0.11 ppm. The corrected
mapping rebuilds source interfaces as `ap + bp * ps_target` and conservatively
remaps layer integrals. Against the protocol state on its own levels, its C90
bottom-band global error is within 0.001 ppm (0.007 ppm columnwise RMS), and it
is insensitive to the uniform pin shift. This strengthens the recommendation
above: comparisons intended to isolate transport should start from GCHP's own
03z state, because matching GCHP's near-surface protocol-file reconstruction
is not a valid initialization criterion.
