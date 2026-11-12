#!/usr/bin/env python3
"""Run configurations of the process-isolated tests with layered tracers (paper Sect. 6.2).

    python3 examples/paper/layer_tracers/make_configs.py --met era5 --binaries <folder of daily binaries> \
        --resolution c90 --configs <dir for the TOML files> --output <dir for the run output>

Seven tracers start as Gaussian profiles of the dry mole fraction in sigma = p/ps, the same in every
column: centered at the surface (sigma = 1) and at sigma = 0.95, 0.85, 0.70, 0.50, 0.25 and 0.07.
The standard deviation of each is the layer thickness of the coarser grid (72 layers) at its center,
so that the 72- and the 117-layer grid start from the same profile (`init.kind = "pressure_gaussian"`
stores the mean of the profile over each layer). `--single-layer` gives the earlier initial condition,
one model layer per column, whose thickness differs between the grids. All tracers carry the same
number of molecules. There are no surface fluxes.

Each month is run with every combination of the three transport processes (advection A,
boundary-layer mixing P, convection C): A, P, C, AP, AC, PC and APC, in Float32 and Float64.
The process options are those of the CATRINE configurations: PPM with the FV3 vertical profile;
for ERA5 TM5 convection and the `tm5_dkg` diffusion, for MERRA-2 CMFMC convection and the
GEOS-Chem non-local boundary-layer scheme.

`--convection` replaces the convection option of the case, for the runs that exchange the solver or
the convective fields: `ifs` (ERA5 fields in the IFS solver), `cmfmc_matrix` (MERRA-2 mass fluxes in
the TM5 matrix solver) and `tm5` (for MERRA-2 binaries that carry ERA5-based TM5 fields, written by
`scripts/preprocessing/attach_catrine_tm5_convection_cs.jl --vertical-mapping pressure_overlap`).
`--name` sets the first part of the run names, which the analysis scripts use as the case name
(`era5ifs`, `merra2tm5matrix`, `merra2era5conv`).

Output: full fields at hours 0, 1, 2, 3, then 3-hourly to day 1, 6-hourly to day 10 and daily
afterwards (hours since the start of the run).
"""
import argparse
import calendar
from pathlib import Path

SIGMA_LAYERS = {"s095": 0.95, "s085": 0.85, "s070": 0.70, "s050": 0.50, "s025": 0.25, "s007": 0.07}
# name: (center, standard deviation) in sigma. The widths are the layer thickness of the 72-layer grid there.
GAUSSIANS = {"sfc": (1.0, 0.015), "s095": (0.95, 0.015), "s085": (0.85, 0.015), "s070": (0.70, 0.025),
             "s050": (0.50, 0.04), "s025": (0.25, 0.045), "s007": (0.07, 0.012)}
TOTAL_MOLECULES = 1.0e36          # gives mole fractions of order 1e-6 in a layer holding 1 % of the air
ARMS = ("A", "P", "C", "AP", "AC", "PC", "APC")
MONTHS = ("2022-01", "2022-04", "2022-07", "2022-10")

MET = {
    "era5": dict(pattern="era5_n320_transport_{YYYYMMDD}_float32.bin",
                 diffusion='kind = "tm5_dkg"',
                 convection='kind = "tm5"\nlmax_conv = 85\nn_merge = 1\ntile_workspace_gib = 0.25\nuse_collab_lu = true'),
    "merra2": dict(pattern="merra2_transport_{YYYYMMDD}_float32.bin",
                   diffusion='kind = "geoschem_nonlocal_vdiff"',
                   convection='kind = "cmfmc"\ncloud_base = "dqrcu"'),
}


# Convection blocks that replace the one of the case (`--convection`).
CONVECTION = {
    "ifs": 'kind = "ifs"\nvariant = "monotone"',
    "cmfmc_matrix": 'kind = "cmfmc_matrix"\ntile_workspace_gib = 0.25\nuse_collab_lu = true',
    "tm5": 'kind = "tm5"\nlmax_conv = 0\nn_merge = 1\ntile_workspace_gib = 0.25\nuse_collab_lu = true',
}


def snapshot_hours(n_days):
    hours = [0, 1, 2, 3] + list(range(6, 25, 3)) + list(range(30, 241, 6)) + list(range(264, 24 * n_days + 1, 24))
    return [h for h in hours if h <= 24 * n_days]


def tracer_blocks(single_layer):
    if single_layer:
        blocks = ['[tracers.sfc.init]\nkind = "pressure_layer"\nlowest_layer = true\n'
                  f"total_molecules = {TOTAL_MOLECULES:.1e}\n"]
        return blocks + [f'[tracers.{name}.init]\nkind = "pressure_layer"\npsurf_fraction = {sigma}\n'
                         f"total_molecules = {TOTAL_MOLECULES:.1e}\n" for name, sigma in SIGMA_LAYERS.items()]
    return [f'[tracers.{name}.init]\nkind = "pressure_gaussian"\npsurf_fraction = {center}\nwidth = {width}\n'
            f"total_molecules = {TOTAL_MOLECULES:.1e}\n" for name, (center, width) in GAUSSIANS.items()]


def config(met, binaries, month, arm, precision, out_dir, days=None, single_layer=False, convection=None):
    year, mon = (int(x) for x in month.split("-"))
    n_days = days or calendar.monthrange(year, mon)[1]
    m = dict(MET[met])
    if convection:
        m["convection"] = CONVECTION[convection]
    tracers = tracer_blocks(single_layer)
    advection = 'scheme = "ppm"\nvertical = "fv3_kord8"' if "A" in arm else 'scheme = "none"'
    label = met + (f" with convection option {convection}" if convection else "")
    return f"""# Layered-tracer process test: {label}, {month}, processes {arm}, {precision}. Written by make_configs.py.
[input]
folder = "{binaries}"
start_date = "{month}-01"
end_date = "{month}-{n_days:02d}"
file_pattern = "{m['pattern']}"

[architecture]
use_gpu = true
backend = "cuda"

[numerics]
float_type = "{precision}"

[advection]
{advection}

[diffusion]
{m['diffusion'] if 'P' in arm else 'kind = "none"'}

[convection]
{m['convection'] if 'C' in arm else 'kind = "none"'}

{chr(10).join(tracers)}
[output]
format = "netcdf"
split = "single"
path = "{out_dir}/output.nc"
hours = {snapshot_hours(n_days)}
deflate_level = 1
shuffle = true

[output.fields]
layers = "full"
column_mean = true
column_mass_per_area = true
air_mass = false
air_mass_layers = "full"
air_mass_per_area = true
column_air_mass_per_area = true
"""


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--met", required=True, choices=sorted(MET))
    p.add_argument("--binaries", required=True, help="folder with the daily transport binaries")
    p.add_argument("--resolution", required=True, help="label of the grid, e.g. c90")
    p.add_argument("--configs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--months", nargs="+", default=list(MONTHS))
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    p.add_argument("--days", type=int, help="run only the first days of each month (test runs)")
    p.add_argument("--single-layer", action="store_true", help="one model layer per column instead of Gaussians")
    p.add_argument("--convection", choices=sorted(CONVECTION), help="replace the convection option of the case")
    p.add_argument("--name", help="first part of the run names (default: --met)")
    p.add_argument("--precisions", nargs="+", default=["Float32", "Float64"], choices=["Float32", "Float64"])
    a = p.parse_args()
    configs = Path(a.configs)
    configs.mkdir(parents=True, exist_ok=True)
    names = []
    for month in a.months:
        for arm in a.arms:
            for precision in a.precisions:
                name = f"{a.name or a.met}_{a.resolution}_{month.replace('-', '')}_{arm}_f{precision[-2:]}"
                (configs / f"{name}.toml").write_text(
                    config(a.met, a.binaries, month, arm, precision, f"{a.output}/{name}", a.days, a.single_layer, a.convection))
                names.append(name)
    print(f"wrote {len(names)} configurations to {configs}")


if __name__ == "__main__":
    main()
