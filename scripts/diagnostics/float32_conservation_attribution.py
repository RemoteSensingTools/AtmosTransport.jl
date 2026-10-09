#!/usr/bin/env python3
"""Float32 tracer-conservation attribution: configs and budget tables.

Runs one week of transport with operator subsets in Float32 and Float64 and
compares global tracer masses. Two met archives are covered:

  merra2  MERRA-2 C90 L72 hourly windows (GEOS-Chem archive), 2021-12-01..07,
          advection / +geoschem_nonlocal_vdiff / +cmfmc (DQRCU) / full;
          tracers: a no-source 400 ppm background, CO2 (CAMS flux), SF6
          (EDGAR, LSCE initial state), fossil CO2 (GridFED, from zero).
  era5    ERA5 C90 L66 with TM5 convection, 2021-12-01..03, no sources,
          advection / +tm5_dkg / +tm5 convection / full; tracers: the 400 ppm
          background and SF6 from its initial state.

Without sources, total tracer mass is invariant, so any drift is arithmetic.
With sources, Float32 is compared with a Float64 run of the same code.
Totals come from the `<tracer>_total_mass` variables, accumulated in Float64
with compensated summation during the run. `budget` prints, per set and
tracer: the Float32 drift; Float32 - Float64 net of the initial offset
(Float32(400e-6) is 2.5e-8 low), relative to the emitted mass for tracers that
start at zero; and for source-free tracers the mean, standard deviation and
t-value of the 3-hourly increments.

    python3 float32_conservation_attribution.py write-configs --out DIR [--data-root ~/data/AtmosTransport]
    julia --project=. scripts/run_transport.jl DIR/merra2_full_f32.toml     # etc.
    python3 float32_conservation_attribution.py budget --out DIR
"""
import argparse, glob, os
import numpy as np

MERRA2_SETS = {
    "adv": "",
    "adv_diff": '[diffusion]\nkind = "geoschem_nonlocal_vdiff"\n',
    "adv_conv": '[convection]\nkind = "cmfmc"\ncloud_base = "dqrcu"\n',
    "full": '[diffusion]\nkind = "geoschem_nonlocal_vdiff"\n\n[convection]\nkind = "cmfmc"\ncloud_base = "dqrcu"\n',
}
TM5_CONVECTION = ('[convection]\nkind = "tm5"\ntile_workspace_gib = 0.25\nuse_collab_lu = true\n'
                  'lmax_conv = 66\nn_merge = 1\n')
ERA5_SETS = {
    "adv": "",
    "adv_dkg": '[diffusion]\nkind = "tm5_dkg"\nsurface_flux_boundary = true\n',
    "adv_tm5conv": TM5_CONVECTION,
    "full": '[diffusion]\nkind = "tm5_dkg"\nsurface_flux_boundary = true\n\n' + TM5_CONVECTION,
}
MERRA2_TRACERS = ("background", "co2_natural", "sf6", "fossil")
ERA5_TRACERS = ("background", "sf6")
SOURCE_FREE = {"merra2": ("background",), "era5": ERA5_TRACERS}


def tracers(root, with_sources):
    background = ('[tracers.background]\n  [tracers.background.init]\n  kind = "uniform"\n'
                  '  background = 400e-6               # no source: total mass is invariant\n')
    sf6_init = ('[tracers.sf6]\n  [tracers.sf6.init]\n  kind = "file"\n'
                f'  file = "{root}/catrine/InitialConditions/startSF6_202112010000.nc"\n  variable = "SF6"\n')
    if not with_sources:
        return background + "\n" + sf6_init
    return background + f'''
[tracers.co2_natural]
  [tracers.co2_natural.init]
  kind = "catrine_co2"
  [tracers.co2_natural.surface_flux]
  kind = "lmdz_co2"
  files = ["{root}/catrine/Emissions/LMDZ_fluxes/z_cams_l_cams55_202112_FT24r2_ra_sfc_3h_co2_flux.nc"]
  time_varying = true
  temporal_scheme = "stepwise"

''' + sf6_init + f'''  [tracers.sf6.surface_flux]
  kind = "cs_native"
  file = "{root}/catrine/fluxes_c90/catrine_sf6_c90.nc"
  variable = "sf6"
  molar_mass_kg_mol = 0.146055
  time_varying = true
  temporal_scheme = "stepwise"

[tracers.fossil]
  [tracers.fossil.init]
  kind = "uniform"
  background = 0.0
  [tracers.fossil.surface_flux]
  kind = "gridfed_fossil_co2"
  files = ["{root}/catrine/Emissions/gridfed/GCP-GridFEDv2024.0_2021.short.nc"]
  molar_mass_kg_mol = 0.0440095
  time_varying = true
  temporal_scheme = "stepwise"
'''


def config(name, met, physics, ft, root, out):
    if met == "merra2":
        folder = f"{root}/met/merra2/c90/transport_binary_v4_l72_f32_physics"
        pattern, end, body = "merra2_transport_{YYYYMMDD}_float32.bin", "2021-12-07", tracers(root, True)
    else:
        folder = f"{root}/met/era5/n320_to_c90/transport_binary_v4_l66_f32_tm5_convection_1deg_3hour_v3"
        pattern, end, body = "era5_n320_transport_{YYYYMMDD}_float32.bin", "2021-12-03", tracers(root, False)
    return f'''# Float32 conservation attribution: {met}, {name}, {ft}.
[input]
folder       = "{folder}"
start_date   = "2021-12-01"
end_date     = "{end}"
file_pattern = "{pattern}"

[architecture]
use_gpu = true
backend = "cuda"

[numerics]
float_type = "{ft}"

[advection]
scheme = "ppm"

{body}
{physics}
[output]
format = "netcdf"
split = "daily"
snapshot_interval_hours = 3
path = "{out}/out/{met}_{name}_{ft.lower()}/attribution_{{YYYYMMDD}}.nc"
  [output.fields]
  layers = "none"
  column_mean = false
  column_mass_per_area = false
  air_mass = false
'''


def write_configs(out, root):
    for met, sets in (("merra2", MERRA2_SETS), ("era5", ERA5_SETS)):
        for name, physics in sets.items():
            for ft in ("Float32", "Float64"):
                tag = f"{met}_{name}_{ft.lower()}"
                os.makedirs(f"{out}/out/{tag}", exist_ok=True)
                with open(f"{out}/{tag}.toml", "w") as fh:
                    fh.write(config(name, met, physics, ft, root, out))
    print(f"configs written to {out}")


def totals(out, tag, names):
    from netCDF4 import Dataset
    files = sorted(glob.glob(f"{out}/out/{tag}/attribution_*.nc"))
    if not files:
        return None
    series = {n: [] for n in names}
    for f in files:
        with Dataset(f) as d:
            for n in names:
                series[n].extend(np.asarray(d[f"{n}_total_mass"][:], float))
    return {n: np.array(v) for n, v in series.items()}


def f32_minus_f64(a, b):
    """Float32 - Float64 at the last common time, net of the initial offset;
    relative to the emitted mass for a tracer that starts at zero."""
    k = min(len(a), len(b)) - 1
    if b[0] > 0:
        return (a[k] - b[k]) / b[k] - (a[0] - b[0]) / b[0]
    return (a[k] - b[k]) / (b[k] - b[0])


def increment_stats(a):
    d = np.diff(a) / a[0]
    sd = d.std(ddof=1)
    return d.mean(), sd, d.mean() / (sd / np.sqrt(len(d))) if sd > 0 else 0.0


def budget(out):
    blank = f"{'-':>10s}"
    print(f"{'archive':7s} {'operators':10s} {'tracer':12s} {'days':>5s} {'F32 drift':>10s} "
          f"{'F32-F64':>10s} {'mean/3h':>10s} {'std/3h':>10s} {'t':>6s}")
    for met, sets, names in (("merra2", MERRA2_SETS, MERRA2_TRACERS), ("era5", ERA5_SETS, ERA5_TRACERS)):
        for name in sets:
            s32, s64 = totals(out, f"{met}_{name}_float32", names), totals(out, f"{met}_{name}_float64", names)
            if s32 is None:
                continue
            for n in names:
                a = s32[n]
                drift = f"{(a[-1] - a[0]) / a[0]:10.2e}" if a[0] > 0 else blank
                diff = f"{f32_minus_f64(a, s64[n]):10.2e}" if s64 is not None else blank
                stats = f"{blank} {blank} {'-':>6s}"
                if n in SOURCE_FREE[met]:
                    mean, sd, t = increment_stats(a)
                    stats = f"{mean:10.1e} {sd:10.1e} {t:6.1f}"
                print(f"{met:7s} {name:10s} {n:12s} {(len(a) - 1) / 8:5.1f} {drift} {diff} {stats}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("write-configs", "budget"))
    ap.add_argument("--out", required=True, help="directory for configs and outputs")
    ap.add_argument("--data-root", default=os.path.expanduser("~/data/AtmosTransport"))
    args = ap.parse_args()
    out = os.path.abspath(os.path.expanduser(args.out))
    write_configs(out, args.data_root) if args.command == "write-configs" else budget(out)


if __name__ == "__main__":
    main()
