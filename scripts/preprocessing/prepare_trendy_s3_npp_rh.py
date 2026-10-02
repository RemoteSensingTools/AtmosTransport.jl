#!/usr/bin/env python3
"""Prepare the TRENDY S3 ensemble's daily -(GPP-Ra) and Rh on native C30.

Uses the earlier ensemble's source interpretation and conservative remapper.
Each component is interpolated separately, preserving its monthly mean before
forming Ra-GPP. LPJ-GUESS uses its annual arh fallback. Never clips net NPP.
Writes files atomically and records source/remap/monthly carbon-budget checks.
"""
from __future__ import annotations

import argparse
import calendar
import datetime as dt
import hashlib
import json
from pathlib import Path

import netCDF4
import numpy as np

import prepare_trendy_s3_gpp_ter_c90 as T


def daily_from_monthly(monthly, keys, centres, day0, nday):
    cx = np.array([(c - day0).total_seconds() / 86400 for c in centres])
    dx = np.arange(nday) + 0.5
    flat = monthly.reshape(len(keys), -1)
    daily = np.stack([np.interp(dx, cx, flat[:, j])
                      for j in range(flat.shape[1])], axis=1)
    dates = [day0 + dt.timedelta(days=i) for i in range(nday)]
    lookup = {k: i for i, k in enumerate(keys)}
    error = 0.0
    for y, m in sorted({(d.year, d.month) for d in dates}):
        sel = np.array([(d.year, d.month) == (y, m) for d in dates])
        target = flat[lookup[y, m]]
        mean = daily[sel].mean(axis=0)
        daily[sel] *= np.divide(target, mean, out=np.zeros_like(target), where=mean > 0)
        active = target > 1e-20
        if active.any():
            error = max(error, float(np.max(np.abs(
                daily[sel].mean(axis=0)[active] / target[active] - 1))))
    assert error < 1e-10, error
    return daily, error


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=T.DEFAULT_ROOT)
    ap.add_argument('--geometry', type=Path, required=True)
    ap.add_argument('--outdir', type=Path, required=True)
    ap.add_argument('--staging-dir', type=Path, required=True)
    ap.add_argument('--models', nargs='+')
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(args.geometry) as ds:
        lat = np.asarray(ds['lats'][:], dtype=np.float64)
        lon = np.asarray(ds['lons'][:], dtype=np.float64)
        area3 = T.c90_cell_areas(np.asarray(ds['corner_lats'][:]),
                                np.asarray(ds['corner_lons'][:]))
    assert lat.shape == (6, 30, 30), lat.shape
    area = area3.ravel()
    assert abs(area.sum() / (4 * np.pi * T.R_EARTH**2) - 1) < 1e-6
    print('Building conservative C30 overlap map', flush=True)
    fine = T.fine_grid_target_map(lat, lon, 0.25)
    keys = [(2013, 12)] + [(y, m) for y in range(2014, 2025) for m in range(1, 13)]
    centres = [dt.datetime(y, m, 1) + dt.timedelta(days=calendar.monthrange(y, m)[1]/2)
               for y, m in keys]
    day0 = dt.datetime(2014, 1, 1)
    nday = (dt.datetime(2025, 1, 1) - day0).days
    models = args.models or [p.name for p in sorted(args.root.iterdir()) if p.is_dir()]
    manifest = {'start': '2014-01-01', 'end': '2024-12-31', 'models': {},
                'geometry': str(args.geometry), 'sign': 'NPP=Ra-GPP; RH=Rh',
                'method': '0.25 degree approximate conservative overlap; separate component monthly-conservative daily interpolation'}
    cache = {}
    for model in models:
        paths = T.model_inputs(args.root / model)
        if paths is None:
            if model not in ('ISBA-CTRIP', 'OCN'):
                raise ValueError(f'Unexpected missing inputs for {model}')
            manifest['models'][model] = {'status': 'excluded', 'reason': 'missing gpp/ra/rh'}
            continue
        out = args.outdir / f'TRENDYv14_S3_{model}_npp_rh_daily_co2flux_c30_2014_2024.nc'
        audit_path = out.with_suffix('.json')
        if out.exists() and audit_path.exists():
            audit = json.loads(audit_path.read_text())
            assert audit['status'] == 'complete'
            manifest['models'][model] = audit
            print(f'[{model}] reusing completed file', flush=True)
            continue
        monthly, audit = {}, {'status': 'preparing', 'components': {}, 'output': str(out)}
        lf = source_lat = source_lon = None
        for kind in ('gpp', 'ra', 'rh'):
            actual = 'arh' if kind == 'rh' and T.container_stem(paths[kind]).lower().endswith('_arh') else kind
            print(f'[{model}] reading {actual}: {paths[kind]}', flush=True)
            with T.materialized(paths[kind], args.staging_dir) as readable, netCDF4.Dataset(readable) as ds:
                var = T.flux_variable(ds, actual)
                shape = tuple(var.shape[-2:])
                slat, slon = T.coordinate(ds, 'lat', shape[0]), T.coordinate(ds, 'lon', shape[1])
                if kind == 'gpp':
                    source_lat, source_lon = slat, slon
                    lf, lf_note = T.read_land_fraction(model, args.root/model, ds, shape, args.staging_dir)
                else:
                    assert np.array_equal(slat, source_lat) and np.array_equal(slon, source_lon), (model, actual, 'grid mismatch')
                gkey = hashlib.sha1(slat.tobytes() + slon.tobytes()).hexdigest()
                if gkey not in cache:
                    cache[gkey] = T.conservative_matrix(slat, slon, fine[2], fine[0], fine[1], fine[3], area.size)
                matrix, source_area = cache[gkey]
                read = T.read_annual_as_months if actual == 'arh' else T.read_months
                raw, units, negfrac, minimum = read(ds, actual, keys)
            mapped = T.remap_months(matrix, raw, lf, area)
            src_total = (raw.reshape(len(keys), -1) * lf.ravel() * source_area.ravel()).sum(axis=1)
            dst_total = (mapped.reshape(len(keys), -1) * area).sum(axis=1)
            rel = float(np.max(np.abs(dst_total-src_total)/np.maximum(np.abs(src_total), 1e-30)))
            assert rel < 1e-10, (model, actual, rel)
            assert np.isfinite(mapped).all()
            monthly[kind] = mapped
            audit['components'][kind] = {'source': str(paths[kind]), 'source_variable': actual,
                'units': units, 'shape': list(raw.shape), 'dtype': str(raw.dtype),
                'negative_fraction_clipped': negfrac, 'source_minimum': minimum,
                'land_fraction': lf_note, 'max_remap_relative_error': rel}
            print(f'[{model}] {actual}: {raw.shape}, global {src_total[-1]:.6g} kg C/s; remap error {rel:.3g}', flush=True)
            del raw
        daily = {}
        for kind in ('gpp', 'ra', 'rh'):
            daily[kind], err = daily_from_monthly(monthly[kind], keys, centres, day0, nday)
            audit['components'][kind]['max_daily_monthly_relative_error'] = err
        fluxes = {'NPP_CO2_FLUX': ((daily['ra']-daily['gpp'])*T.KG_CO2_PER_KG_C).astype('f4'),
                  'RH_CO2_FLUX': (daily['rh']*T.KG_CO2_PER_KG_C).astype('f4')}
        assert all(np.isfinite(f).all() for f in fluxes.values())
        assert np.min(fluxes['RH_CO2_FLUX']) >= 0
        audit['annual_PgC'] = {}
        for year in range(2014, 2025):
            i = (dt.datetime(year, 1, 1)-day0).days
            j = (dt.datetime(year+1, 1, 1)-day0).days
            budget = {name: float((f[i:j].astype('f8')*area).sum()*86400*12/44/1e12)
                      for name, f in fluxes.items()}
            audit['annual_PgC'][str(year)] = budget
            print(f'[{model}] {year}: {budget} PgC', flush=True)
        # Integrated drawdown used to assess the positive carrier margin.
        first = (dt.datetime(2014, 9, 1)-day0).days
        uptake = fluxes['NPP_CO2_FLUX'][first:].astype('f8')
        cumulative_ppm = np.cumsum((uptake*area).sum(axis=1)*86400*0.02896546/0.0440095/5.13531539605612e18*1e6)
        audit['npp_max_global_drawdown_ppm'] = float(-min(0, cumulative_ppm.min()))
        assert audit['npp_max_global_drawdown_ppm'] < 1000, audit
        tmp = out.with_suffix('.partial.nc')
        with netCDF4.Dataset(tmp, 'w') as ds:
            for name, size in [('time', nday), ('nf', 6), ('Ydim', 30), ('Xdim', 30)]:
                ds.createDimension(name, size)
            time = ds.createVariable('time', 'f8', ('time',))
            time.units = 'hours since 2014-01-01 00:00:00 UTC'
            time.calendar = 'proleptic_gregorian'
            time[:] = np.arange(nday)*24
            for name, data in [('lats', lat), ('lons', lon), ('cell_area', area3)]:
                ds.createVariable(name, 'f8', ('nf','Ydim','Xdim'))[:] = data
            for name, data in fluxes.items():
                v = ds.createVariable(name, 'f4', ('time','nf','Ydim','Xdim'),
                                      zlib=True, complevel=1, shuffle=True, chunksizes=(1,6,30,30))
                v.units = 'kg CO2 m-2 s-1'
                v.positive = 'to_atmosphere'
                v.long_name = 'Ra - GPP (negative NPP)' if name.startswith('NPP') else 'heterotrophic respiration'
                v[:] = data.reshape((nday,6,30,30))
            ds.source_model = model
            ds.Conventions = 'CF-1.8'
            ds.method = manifest['method']
            ds.rh_source_variable = audit['components']['rh']['source_variable']
            ds.history = dt.datetime.now(dt.timezone.utc).isoformat()
        tmp.replace(out)
        audit['status'] = 'complete'
        audit_path.write_text(json.dumps(audit, indent=2)+'\n')
        manifest['models'][model] = audit
        (args.outdir/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print(f'[{model}] complete: {out}', flush=True)
    (args.outdir/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Ensemble preparation complete', flush=True)


if __name__ == '__main__':
    main()
