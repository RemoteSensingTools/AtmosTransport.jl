# TRENDY and GFED5 land TransCom tracers on C30

Requested 2026-09-08, following the 23-model global TRENDY campaign.
Run root: `/temp1/cfranken/trendy_transcom_c30_20260908`.

The campaign uses 11 sequential batches on physical GPU 1, the same L40S
used by the parent campaign. Each batch contains all 23 models' NPP and Rh
tracers and one GFED5 fire tracer for a single land TransCom region: 47
tracers per integration, 517 regional tracers total. NPP already carries
the atmospheric Ra−GPP sign. No ocean-region tracers are created.

Period, meteorology and numerics match the parent: 2014-09-01 through
2024-12-31, ERA5 C30/L66, FP32, PPM, TM5 DKG diffusion, no convection,
stepwise daily surface fluxes, and preserve-tracer-mass air-mass resets.
The parent source snapshot and Manifest are copied into an isolated source
directory; only campaign/preprocessing Python scripts are added. No Julia
transport kernels or flux-loading code are changed. GPU selection is by UUID
`GPU-d67e323b-b699-9138-7640-24f4774841e3`.

## Regional attribution

The source mask is
`/kiwi-data/Data/groupMembers/evametz/masks/regions.nc`, variable
`transcom_regions`. The original file is copied to the campaign and hashed.
IDs 1–11 and their names are read from the file:

1. North American Boreal
2. North American Temperate
3. South American Tropical
4. South American Temperate
5. Northern Africa
6. Southern Africa
7. Eurasia Boreal
8. Eurasia Temperate
9. Tropical Asia
10. Australia
11. Europe

This is a partition of the **existing C30 drivers**, preserving the runs
being compared. It does not reassign the original source-grid flux pixels
before regridding. The 1-degree categorical mask is sampled at 0.25 degrees,
using the same nearest-center overlap approximation as the parent remapper.
In each C30 cell, sampled land-region areas are normalized to sum to one.
A cell with no sampled land is assigned to its closest labeled 1-degree
land pixel. This preserves flux from coarse-grid/coastal/mask discrepancies.
The exact fallback fraction is audited separately for every model/component;
initial samples were 0.09% for GFED5 and 0.2–0.6% for selected TRENDY fields.
These are fractions of absolute flux, not signed net flux.
The completed all-model audit found larger fallback fractions for CLM-FATES
(4.22% NPP, 4.19% Rh) and ELM (1.26% NPP, 1.44% Rh); every other component
was below 1%. Treat the regional attribution of those unmatched cells as an
additional approximation, especially for CLM-FATES. The largest pointwise
FP32 partition closure error across all 47 components was 5.93e-8, passing
the 2e-7 tolerance.

`transcom_weights_c30.nc` stores all 11 fractional weights, the fallback flag
and nearest-land distance. Before running, the preflight explicitly adds all
11 FP32 regional fluxes for every source day and cell and checks closure to
the original within 2e-7 relative to nonzero flux. `partition_validation.json`
records closure and attribution audits. Three focused tests cover fractional
coast handling, fallback allocation, signed FP32 conservation and rejection
of invalid weights.

## Execution and validation

The detached controller is `scripts/run_transcom_campaign.py`. State and logs
are in `status.json`, `campaign.log`, and `batches/tcNN/`. Each region's input
files are prepared before its batch, with signed budget integrals computed
from the actual FP32 files. A three-day smoke integration precedes the full
integration. A completed batch is retained; partial outputs are never
silently overwritten on restart.

Every smoke/full output is checked for expected days and times, finite
positive columns, and integrated tracer storage. The parent's scientific
budget tolerance remains **1e-4 relative plus one FP32 carrier ULP**.
Failures of that tolerance remain explicit in each batch report. A separate
operational guard (**1e-3 relative plus four carrier ULPs**) stops larger
budget errors; passing this guard does not turn a scientific failure into a
pass. Final campaign status is `complete_with_budget_flags` if any regional
budget fails the original tolerance. Invalid/missing outputs, exhausted
carriers, or a failed operational guard stop the campaign.

This distinction follows the prior campaign's TEM NPP failure at 0.0127%
relative error: modest accumulated FP32 budget errors must remain visible
without silently preventing all subsequent regions from running.

## Outputs and interpretation

Daily files:
`batches/tcNN/output/transcom_tcNN_YYYYMMDD.nc`.
Tracers are named `co2_<model>_<npp|rh>_tcNN` and
`co2_gfed5_fire_tcNN`. As before, initial carriers are 1300 ppm for each NPP
tracer and 10 ppm for each Rh or fire tracer. Subtract the carrier **for each
region** before adding regional responses. For a given model and region:

    NEE + fire enhancement = (NPP − 1300) + (Rh − 10) + (GFED5 − 10) ppm

After all integrations, the controller writes
`regional_monthly_hovmoller.npz`: 2015–2024 monthly enhancements in 40
equal-area latitude bands, dimensions `(region, month, component, band)`.
It also writes `regional_sum_vs_parent.npz` and
`regional_reconstruction.json`, comparing sums of regional enhancements
with the parent global-model and old GFED5 outputs. Flux partition closure
does not guarantee bitwise transport superposition: PPM limiters, FP32
carrier errors and the older GFED5 code snapshot can cause differences.

This campaign reruns GFED5 under the same transport snapshot as the regional
TRENDY tracers. It does not merely partition the previously transported
GFED5 XCO2 field geographically; the source fluxes are tagged before transport.

## Launch verification

The detached controller started successfully (PID 3625848). The region-1
three-day, 47-tracer smoke integration passed finite/positive/complete checks
and the original strict budget tolerance for every tracer. The first full
integration started at 2026-09-08 21:14:35 UTC (initial PID 3626388).
Subsequent region transitions and completion/failure are recorded in
`status.json`. Based on the preceding global run, the eleven sequential
integrations are expected to take roughly one day; this is an estimate,
not a measured regional-campaign completion time.
