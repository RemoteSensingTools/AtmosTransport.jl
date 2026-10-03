#!/usr/bin/env python3
"""Check every daily ensemble output and component's integrated flux budget."""
import argparse
import datetime as dt
import json
from pathlib import Path

import netCDF4
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-root', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--end-date', required=True)
    ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads((args.run_root/'fluxes/manifest.json').read_text())
    models = {k: v for k, v in manifest['models'].items() if v['status'] == 'complete'}
    assert len(models) == 23, len(models)
    names = {f"co2_{m.lower().replace('-', '_')}_{c}": (m, c)
             for m in models for c in ('npp', 'rh')}
    start = dt.date(2014, 9, 1)
    nday = (dt.date.fromisoformat(args.end_date) - start).days + 1
    initial, final, carrier_ulp = {}, {}, {}
    for day in range(nday):
        date = start + dt.timedelta(days=day)
        path = args.output_dir / f'trendy_allmodels_c30_{date:%Y%m%d}.nc'
        with netCDF4.Dataset(path) as ds:
            times = np.asarray(ds['time'][:])
            np.testing.assert_allclose(times, [0,24] if day == 0 else [(day+1)*24], atol=1e-9)
            for name in names:
                columns = np.asarray(ds[name+'_column_mean'][:])
                masses = np.asarray(ds[name+'_total_mass'][:])
                assert np.isfinite(columns).all() and np.isfinite(masses).all(), (date, name)
                assert columns.min() > 0, (date, name, 'carrier exhausted')
                if day == 0:
                    initial[name] = float(masses[0])
                    carrier = 1.3e-3 if names[name][1] == 'npp' else 1e-5
                    # Small source increments are differences of large FP32
                    # carrier fields. Allow one carrier ULP in addition to
                    # 1e-4 of the integrated source; retain the strict relative
                    # check as the source signal grows over the full record.
                    air_mass = initial[name] / carrier
                    carrier_ulp[name] = float(np.spacing(np.float32(carrier))) * air_mass
                final[name] = float(masses[-1])
    checks = {}
    first = (start - dt.date(2014,1,1)).days
    for model, audit in models.items():
        with netCDF4.Dataset(audit['output']) as ds:
            area = np.asarray(ds['cell_area'][:], dtype='f8')
            for comp in ('npp','rh'):
                name = f"co2_{model.lower().replace('-', '_')}_{comp}"
                # Float64 global storage is VMR * dry-air mass, not kg CO2.
                expected = 0.0
                for i in range(first, first+nday, 64):
                    flux = np.asarray(ds[comp.upper()+'_CO2_FLUX'][i:min(i+64, first+nday)], dtype='f8')
                    expected += float((flux*area).sum())*86400*0.02896546/0.0440095
                actual = final[name]-initial[name]
                relative = abs(actual-expected)/max(abs(expected), 1)
                tolerance = 1e-4*abs(expected) + carrier_ulp[name]
                checks[name] = {'expected_storage_change_kg': expected,
                                'actual_storage_change_kg': actual,
                                'relative_error': relative,
                                'absolute_tolerance_kg': tolerance,
                                'passed': abs(actual-expected) <= tolerance}
    worst = max(v['relative_error'] for v in checks.values())
    report = {'days': nday, 'models': len(models), 'tracers': len(names),
              'max_relative_budget_error': worst, 'checks': checks,
              'tolerance': '1e-4 relative plus one FP32 ULP of each initial carrier',
              'passed': all(v['passed'] for v in checks.values())}
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'checks'}, indent=2), flush=True)
    assert report['passed'], f'Flux budget mismatch: {worst:.6g}'


if __name__ == '__main__':
    main()
