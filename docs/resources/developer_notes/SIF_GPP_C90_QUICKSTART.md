# Running SIF-derived daily GPP with the C90 transport model

This is the operational guide for running a SIF-derived GPP experiment on
**wurst** or an **Apple-Silicon Mac**. The example uses the completed
ERA5-N320-to-C90, L66, Float32, format-4 transport archive for 2014–2025 and
the portable configuration
**config/runs/sif_gpp_daily_c90_2021_template.toml**.

## 1. What the model expects

AtmosTransport does not infer GPP directly from raw SIF. The SIF retrieval
must first be calibrated to daily GPP outside the transport model. For the
existing 2021 SIF/FLUXCOM-X hybrid, the calibration and gap-filling method is
documented in **docs/memos/SIF_DERIVED_GPP_2021_ATBD.md**; its implementation
is **/home/cfranken/code/gitHub/OCO-dashboard/pilot/analyze_2021.py**.

The recommended transport input is one NetCDF containing daily mean
atmospheric CO2 flux on the native GEOS C90 cubed sphere:

- Dimensions on disk: (time, nf, Ydim, Xdim) = (365 or 366, 6, 90, 90).
- One timestamp at 00:00 UTC for each day, in ascending order.
- Time units such as “hours since 2021-01-01 00:00:00 UTC”.
- Variable name GPP_CO2_FLUX.
- Units kg CO2 m-2 s-1, defined per total C90 cell area.
- Negative values for photosynthetic atmospheric uptake.
- No NaNs or unhandled fill values.

The run uses a stepwise temporal scheme, so each daily value is held constant
until the next daily timestamp.

If calibrated GPP is a positive daily total GPP_gC_m2_day in
g C m-2 day-1 over total cell area, convert it as:

    GPP_CO2_FLUX =
        -GPP_gC_m2_day * 1e-3 * (44.0095 / 12.0107) / 86400

If GPP is defined per vegetated area, multiply by vegetated-land fraction
before applying this conversion. If starting from the existing hourly SIF
driver, the arithmetic mean of its 24 hourly fluxes is the conservative daily
mean. The supplied converter performs that aggregation:

    python scripts/preprocessing/hourly_c90_flux_to_daily.py INPUT_HOURLY.nc OUTPUT_DAILY.nc

Raw SIF has no universal unit conversion to GPP: the SIF-to-GPP calibration,
quality filtering, gap filling, and remapping to C90 are scientific
preprocessing choices, not transport settings.

Check the finished file before running:

    ncdump -h "$ATMOSTRANSPORT_DATA_ROOT/fluxes/sif_gpp_daily_co2flux_c90_2021.nc"

## 2. Runtime and data locations

The current C90 files require the format-4 runtime snapshot. Until that work is
published on GitHub, use the frozen snapshot supplied with this experiment;
an older checkout that expects format 3 will reject these binaries.

The shared C90 archive is:

    /kiwi-data/Data/groupMembers/cfranken/AtmosTransport/met/era5/n320_to_c90/transport_binary_v4_l66_f32_no_convection

All 4,383 daily files for 2014–2025 are present and validated. One 2021 year
is about 628 GB. The .validated files are zero-byte completion markers and
are not needed by the runtime.

The frozen runtime snapshot is:

    /kiwi-data/Data/groupMembers/cfranken/AtmosTransport/runtime/AtmosTransportModel-format4-20260813

Copy the snapshot into your own writable directory before installing Julia
packages.

## 3. Set up and run on wurst

The following uses a personal run root on wurst's NVMe:

    export MODEL_ROOT="$HOME/code/AtmosTransportModel-format4-20260813"
    export ATMOSTRANSPORT_DATA_ROOT="/temp2/$USER/sif_gpp_c90"

    mkdir -p "$HOME/code" "$ATMOSTRANSPORT_DATA_ROOT/met" "$ATMOSTRANSPORT_DATA_ROOT/fluxes" "$ATMOSTRANSPORT_DATA_ROOT/output/sif_gpp_c90_2021"

    rsync -a /kiwi-data/Data/groupMembers/cfranken/AtmosTransport/runtime/AtmosTransportModel-format4-20260813/ "$MODEL_ROOT/"

    ln -s /kiwi-data/Data/groupMembers/cfranken/AtmosTransport/met/era5/n320_to_c90/transport_binary_v4_l66_f32_no_convection "$ATMOSTRANSPORT_DATA_ROOT/met/c90_l66"

    cp /kiwi-data/Data/satellite/FLUXCOM-X/X-BASE/2021/pilot/sif_gpp_daily_co2flux_c90_2021.nc "$ATMOSTRANSPORT_DATA_ROOT/fluxes/"

    julia --project="$MODEL_ROOT" -e 'using Pkg; Pkg.resolve(); Pkg.instantiate(); Pkg.add("CUDA")'

Copy the supplied one-day smoke configuration first:

    cp "$MODEL_ROOT/config/runs/sif_gpp_daily_c90_2021_smoke.toml" "$ATMOSTRANSPORT_DATA_ROOT/sif_gpp_run.toml"

The C90 files live on network storage, but no input staging is required and
none is available: the format-4 snapshot does **not** implement
`[input.staging]`, and because the run schema permits additional properties the
key is accepted and then silently ignored. Measured sequential read from
kiwi-data is about 778 MB/s, against roughly 110 MB/s consumed by a C90 run, so
reading the archive in place is not a bottleneck. (Verified 2026-08-28 on
wurst.)

Launch interactively to see progress:

    export ATMOSTR_TIMERS=1
    /usr/bin/time -v julia --project="$MODEL_ROOT" "$MODEL_ROOT/scripts/run_transport.jl" "$ATMOSTRANSPORT_DATA_ROOT/sif_gpp_run.toml" 2>&1 | tee "$ATMOSTRANSPORT_DATA_ROOT/output/sif_gpp_c90_2021/run.log"

Startup must contain:

    [gpu verified] backend=cuda

This exact smoke case was validated on wurst on 2026-08-13. After package
compilation, the one-day forward run took 43 seconds and wrote a valid C90
column-mean NetCDF.

After the smoke test succeeds, replace the run configuration with the full
year template and rerun in a persistent terminal (tmux or screen):

    cp "$MODEL_ROOT/config/runs/sif_gpp_daily_c90_2021_template.toml" "$ATMOSTRANSPORT_DATA_ROOT/sif_gpp_run.toml"

## 4. Set up and run on an Apple-Silicon Mac

Install Julia 1.12 with juliaup. Plan for about 630 GB of free space for one
C90 year plus output. From the Mac, copy only the required year:

    export ATMOSTRANSPORT_DATA_ROOT="$HOME/data/sif_gpp_c90"
    export MODEL_ROOT="$HOME/code/AtmosTransportModel-format4-20260813"
    mkdir -p "$MODEL_ROOT" "$ATMOSTRANSPORT_DATA_ROOT/met/c90_l66" "$ATMOSTRANSPORT_DATA_ROOT/fluxes" "$ATMOSTRANSPORT_DATA_ROOT/output/sif_gpp_c90_2021"

    rsync -av --partial --progress CALTECH_USER@wurst.gps.caltech.edu:/kiwi-data/Data/groupMembers/cfranken/AtmosTransport/runtime/AtmosTransportModel-format4-20260813/ "$MODEL_ROOT/"

    rsync -av --partial --progress --include='era5_n320_transport_2021????_float32.bin' --exclude='*' CALTECH_USER@wurst.gps.caltech.edu:/kiwi-data/Data/groupMembers/cfranken/AtmosTransport/met/era5/n320_to_c90/transport_binary_v4_l66_f32_no_convection/ "$ATMOSTRANSPORT_DATA_ROOT/met/c90_l66/"

    rsync -av --partial --progress CALTECH_USER@wurst.gps.caltech.edu:/kiwi-data/Data/satellite/FLUXCOM-X/X-BASE/2021/pilot/sif_gpp_daily_co2flux_c90_2021.nc "$ATMOSTRANSPORT_DATA_ROOT/fluxes/"

    julia --project="$MODEL_ROOT" -e 'using Pkg; Pkg.resolve(); Pkg.instantiate(); Pkg.add("Metal")'

    cp "$MODEL_ROOT/config/runs/sif_gpp_daily_c90_2021_smoke.toml" "$ATMOSTRANSPORT_DATA_ROOT/sif_gpp_run.toml"

Input staging is not applicable on the Mac either -- the files are already on
its local SSD, and the snapshot ignores the key regardless. Run the one-day
smoke test first:

    export ATMOSTR_TIMERS=1
    /usr/bin/time -l julia --project="$MODEL_ROOT" "$MODEL_ROOT/scripts/run_transport.jl" "$ATMOSTRANSPORT_DATA_ROOT/sif_gpp_run.toml" 2>&1 | tee "$ATMOSTRANSPORT_DATA_ROOT/output/sif_gpp_c90_2021/run.log"

Startup must contain:

    [gpu verified] backend=metal backing=MtlArray

After it succeeds, select the full-year configuration and rerun:

    cp "$MODEL_ROOT/config/runs/sif_gpp_daily_c90_2021_template.toml" "$ATMOSTRANSPORT_DATA_ROOT/sif_gpp_run.toml"

## 5. Interpreting output

The example writes one daily C90 column-mean field. The tracer starts with a
400 ppm carrier because GPP is a negative source. Subtract 400 ppm from the
output VMR/XCO2 to obtain the transported SIF-GPP contribution.

These L66 binaries include PPM advection and TM5 DKG diffusion/PBL transport,
but do not include convection mass fluxes. Keep convection disabled unless a
separate convection-enabled C90 archive is prepared.

For troubleshooting, retain run.log and the generated timings CSV. Useful
checks are:

    grep -E 'gpu verified|Surface source|Done:|Forward run wall' "$ATMOSTRANSPORT_DATA_ROOT/output/sif_gpp_c90_2021/run.log"

    find "$ATMOSTRANSPORT_DATA_ROOT/output/sif_gpp_c90_2021" -name '*.nc' | wc -l

The detailed provenance of the existing 2021 SIF-derived product is in
**docs/memos/SIF_DERIVED_GPP_2021_ATBD.md**.

## 6. Multi-year interannual-variability experiment (2018-2025)

The single-year case above is a pilot. To run the full TROPOMI SIF record and
isolate the XCO2 signal from interannual variability in SIF-derived GPP, use:

    # 2018-05-01 .. 2025-12-31 daily C90 flux driver, ~20 s
    python3 scripts/preprocessing/build_sif_gpp_multiyear_c90_flux.py \
      --out $ATMOSTRANSPORT_DATA_ROOT/fluxes/sif_gpp_daily_co2flux_c90_2018_2025.nc

    # two tracers, one met stream, ~3 h on one L40S
    julia --project="$MODEL_ROOT" --threads=4 "$MODEL_ROOT/scripts/run_transport.jl" \
      config/runs/sif_gpp_iav_c90_2018_2025.toml

    python3 scripts/diagnostics/analyze_sif_gpp_iav_xco2.py
    python3 scripts/diagnostics/plot_sif_gpp_iav_xco2.py

That configuration carries `co2_gpp_anom` (forced by the GPP flux *anomaly*,
100 ppm carrier) and `co2_gpp_full` (full uptake, 900 ppm carrier). Because
transport is linear in the tracer, the anomaly tracer *is* the interannual
signal -- no post-hoc detrending is required, and its global mean does not
drift. The full tracer removes about 469 ppm over the record, which is why its
carrier is 900 ppm rather than 400.

Design, calibration provenance, validation numbers and caveats:
**docs/memos/SIF_GPP_IAV_XCO2_2018_2025.md**.
