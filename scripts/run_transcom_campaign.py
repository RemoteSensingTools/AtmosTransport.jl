#!/usr/bin/env python3
"""Detached 11-batch, 517-tracer TRENDY + GFED land TransCom campaign."""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

import netCDF4
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'preprocessing'))
import prepare_transcom_c30 as P


def config(root,region,tracers,smoke=False):
    folder=root/'batches'/f'tc{region:02d}'
    out=folder/('smoke_output' if smoke else 'output')
    out.mkdir(parents=True,exist_ok=True)
    text=f'''[architecture]
use_gpu = true
backend = "cuda"
[numerics]
float_type = "Float32"
[input]
folder = "/temp1/cfranken/sif_gpp_iav/met/c30_l66"
start_date = "2014-09-01"
end_date = "{'2014-09-03' if smoke else '2024-12-31'}"
file_pattern = "era5_c30_transport_{{YYYYMMDD}}_float32.bin"
[input.staging]
enabled = true
dir = "{root}/met_stage"
lookahead_days = 3
keep_behind_days = 0
[run]
air_mass_reset_mode = "preserve_tracer_mass"
[advection]
scheme = "ppm"
[diffusion]
kind = "tm5_dkg"
surface_flux_boundary = true
[convection]
kind = "none"
'''
    for s in tracers:
        text+=f'''\n[tracers.{s['name']}.init]
kind = "uniform"
background = {s['carrier']}
[tracers.{s['name']}.surface_flux]
kind = "cs_native"
file = {json.dumps(s['file'])}
variable = "{s['variable']}"
molar_mass_kg_mol = 0.0440095
time_varying = true
temporal_scheme = "stepwise"
'''
    text+=f'''\n[output]
format = "netcdf"
split = "daily"
snapshot_interval_hours = 24
path = "{out}/transcom_tc{region:02d}_{{YYYYMMDD}}.nc"
deflate_level = 1
shuffle = true
[output.fields]
tracers = {json.dumps([s['name'] for s in tracers])}
layers = "none"
column_mean = true
column_mass_per_area = false
air_mass_layers = "none"
air_mass = false
air_mass_per_area = false
column_air_mass_per_area = false
'''
    path=folder/('smoke.toml' if smoke else 'run.toml');path.write_text(text)
    return path


def validate(root,region,specs,smoke=False):
    folder=root/'batches'/f'tc{region:02d}'
    out=folder/('smoke_output' if smoke else 'output')
    n=3 if smoke else P.NDAYS
    assert len(list(out.glob('*.nc')))==n
    initial={};final={};mins={s['name']:float('inf') for s in specs}
    for i in range(n):
        day=P.START+dt.timedelta(days=i)
        with netCDF4.Dataset(out/f'transcom_tc{region:02d}_{day:%Y%m%d}.nc') as ds:
            np.testing.assert_allclose(ds['time'][:],[0,24] if i==0 else [(i+1)*24])
            for s in specs:
                name=s['name'];x=np.asarray(ds[name+'_column_mean'][:])
                masses=np.asarray(ds[name+'_total_mass'][:])
                assert np.isfinite(x).all() and np.isfinite(masses).all(),(day,name)
                mins[name]=min(mins[name],float(x.min()))
                assert x.min()>0,(day,name,'carrier exhausted')
                if i==0:initial[name]=float(masses[0])
                final[name]=float(masses[-1])
    checks={}
    for s in specs:
        name=s['name'];air=initial[name]/s['carrier']
        expected=s['expected_smoke_storage_change_kg' if smoke else 'expected_full_storage_change_kg']
        actual=final[name]-initial[name]
        ulp=float(np.spacing(np.float32(s['carrier'])))*air
        error=abs(actual-expected)
        strict=1e-4*abs(expected)+ulp
        # Preserve the prior scientific tolerance and report all failures.
        # A distinct operational guard stops gross budget errors; flagged
        # batches remain explicitly provisional rather than silently passing.
        guard=1e-3*abs(expected)+4*ulp
        checks[name]={'expected_storage_change_kg':expected,'actual_storage_change_kg':actual,
                     'error_ppm':(actual-expected)/air*1e6,
                     'relative_error':error/max(abs(expected),1),
                     'strict_tolerance_kg':strict,'strict_passed':error<=strict,
                     'operational_tolerance_kg':guard,'operational_passed':error<=guard,
                     'minimum_column_ppm':mins[name]*1e6}
    report={'region':region,'days':n,'tracers':len(specs),'finite_positive_complete':True,
            'strict_passed':all(v['strict_passed'] for v in checks.values()),
            'operational_passed':all(v['operational_passed'] for v in checks.values()),
            'strict_tolerance':'1e-4 relative plus one carrier FP32 ULP',
            'operational_guard':'1e-3 relative plus four carrier FP32 ULPs; does not replace scientific validation',
            'checks':checks}
    P.save_json(folder/('smoke_validation.json' if smoke else 'validation.json'),report)
    assert report['operational_passed'],f'tc{region:02d} exceeds operational budget guard'
    return report


def summarize(root,all_specs,parent):
    """Monthly regional zonal responses and reconstruction against parent runs."""
    regions=json.loads((root/'regions.json').read_text())
    months=np.arange('2015-01','2025-01',dtype='datetime64[M]')
    dates=np.arange('2015-01-01','2025-01-01',dtype='datetime64[D]')
    with netCDF4.Dataset(root/'transcom_weights_c30.nc') as ds:
        lat=np.asarray(ds['lats'][:]).ravel();area=np.asarray(ds['cell_area'][:]).ravel()
    edges=np.arange(41)/40*2-1;bands=np.clip(np.digitize(np.sin(np.deg2rad(lat)),edges)-1,0,39)
    denom=np.bincount(bands,weights=area,minlength=40)
    monthly=np.zeros((11,120,47,40));counts=np.zeros((11,120),dtype=int)
    for ri,region in enumerate(regions):
        specs=all_specs[str(region['id'])]
        for date in dates:
            mi=int(np.searchsorted(months,date.astype('datetime64[M]')))
            tag=str(date).replace('-','')
            with netCDF4.Dataset(root/f"batches/tc{region['id']:02d}/output/transcom_tc{region['id']:02d}_{tag}.nc") as ds:
                for j,s in enumerate(specs):
                    x=(np.asarray(ds[s['name']+'_column_mean'][-1],dtype='f8').ravel()-s['carrier'])*1e6
                    monthly[ri,mi,j]+=np.bincount(bands,weights=x*area,minlength=40)/denom
            counts[ri,mi]+=1
        monthly[ri]/=counts[ri,:,None,None]
        print(f"Summarized region {region['id']}",flush=True)
    base_names=[s['name'].rsplit('_tc',1)[0] for s in all_specs['1']]
    np.savez_compressed(root/'regional_monthly_hovmoller.npz',
                        months=months,region_ids=np.arange(1,12),
                        region_names=[r['name'] for r in regions],
                        tracers=base_names,sin_edges=edges,xco2_enhancement_ppm=monthly)
    # Independent parent outputs; GFED parent uses the older transport code.
    parent_monthly=np.zeros((120,47,40));ct=np.zeros(120,dtype=int)
    for date in dates:
        mi=int(np.searchsorted(months,date.astype('datetime64[M]')));tag=str(date).replace('-','')
        with netCDF4.Dataset(parent/f'output/trendy_allmodels_c30_{tag}.nc') as ds:
            for j,s in enumerate(all_specs['1'][:-1]):
                x=(np.asarray(ds[base_names[j]+'_column_mean'][-1],dtype='f8').ravel()-s['carrier'])*1e6
                parent_monthly[mi,j]+=np.bincount(bands,weights=x*area,minlength=40)/denom
        with netCDF4.Dataset(f'/temp1/cfranken/sif_gpp_iav/output/gfed5_fire_c30/gfed5_fire_c30_{tag}.nc') as ds:
            x=(np.asarray(ds['co2_gfed_fire_column_mean'][-1],dtype='f8').ravel()-1e-5)*1e6
            parent_monthly[mi,-1]+=np.bincount(bands,weights=x*area,minlength=40)/denom
        ct[mi]+=1
    parent_monthly/=ct[:,None,None]
    delta=monthly.sum(axis=0)-parent_monthly
    np.savez_compressed(root/'regional_sum_vs_parent.npz',months=months,tracers=base_names,
                        parent_ppm=parent_monthly,regional_sum_ppm=monthly.sum(axis=0),
                        difference_ppm=delta,sin_edges=edges)
    P.save_json(root/'regional_reconstruction.json',{
        'note':'Flux partition closes to FP32 precision; separately transported sums can differ through limiters, accumulated FP32 carrier error, and the older GFED transport snapshot.',
        'tracers':{name:{'monthly_band_rmse_ppm':float(np.sqrt(np.mean(delta[:,j]**2))),
                        'maximum_absolute_difference_ppm':float(np.max(abs(delta[:,j])))}
                   for j,name in enumerate(base_names)}})


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-root',type=Path,required=True)
    ap.add_argument('--parent',type=Path,required=True)
    ap.add_argument('--mask',type=Path,required=True)
    ap.add_argument('--fire',type=Path,required=True)
    ap.add_argument('--julia',required=True)
    args=ap.parse_args();root=args.run_root.resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'campaign.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status={'controller_pid':os.getpid(),'gpu':os.environ['CUDA_VISIBLE_DEVICES'],
            'source_snapshot':str(Path.cwd()),'regions_completed':[]}
    previous=root/'status.json'
    if previous.exists():
        status['regions_completed']=json.loads(previous.read_text()).get('regions_completed',[])
    def state(phase,**kw):
        status.update(phase=phase,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw)
        P.save_json(root/'status.json',status);print(json.dumps(status),flush=True)
    try:
        state('preflight')
        specs=P.sources(args.parent,args.fire)
        if not (root/'partition_validation.json').exists():
            w,missing,area=P.build_mask(args.mask,Path(specs[0]['file']),root)
            P.audit_partition(specs,w,missing,area,root)
        else:
            assert json.loads((root/'partition_validation.json').read_text())['passed']
            with netCDF4.Dataset(root/'transcom_weights_c30.nc') as ds:w=np.asarray(ds['weight'][:]).reshape(11,5400)
        all_specs={}
        for region in range(1,12):
            batch=root/'batches'/f'tc{region:02d}';batch.mkdir(parents=True,exist_ok=True)
            state('preparing_region',region=region)
            tracer_specs=P.prepare_region(region,specs,w,root)
            all_specs[str(region)]=tracer_specs;P.save_json(batch/'tracers.json',tracer_specs)
            if region in status['regions_completed']:continue
            for smoke in (True,False):
                report=batch/('smoke_validation.json' if smoke else 'validation.json')
                if report.exists() and json.loads(report.read_text())['operational_passed']:continue
                cfg=config(root,region,tracer_specs,smoke)
                output=batch/('smoke_output' if smoke else 'output')
                assert not any(output.glob('*.nc')),f'Partial output exists at {output}; refusing to overwrite'
                log=batch/('smoke.log' if smoke else 'transport.log')
                with log.open('w') as stream:
                    child=subprocess.Popen([args.julia,'--project=.','--threads=4','scripts/run_transport.jl',str(cfg)],
                                           stdout=stream,stderr=subprocess.STDOUT)
                    state('smoke' if smoke else 'transport',region=region,child_pid=child.pid,log=str(log))
                    code=child.wait()
                assert code==0,f'tc{region:02d} transport exited {code}: {log}'
                state('smoke_validation' if smoke else 'validation',child_pid=None)
                result=validate(root,region,tracer_specs,smoke)
                print(f"tc{region:02d}: strict budget pass = {result['strict_passed']}",flush=True)
            status['regions_completed'].append(region)
            state('region_complete',region=region)
        P.save_json(root/'tracers.json',all_specs)
        state('summarizing',child_pid=None)
        summarize(root,all_specs,args.parent)
        strict=all(json.loads((root/f'batches/tc{i:02d}/validation.json').read_text())['strict_passed'] for i in range(1,12))
        state('complete' if strict else 'complete_with_budget_flags',strict_budget_passed=strict,child_pid=None)
    except Exception as exc:
        state('failed',error=str(exc),child_pid=None);raise


if __name__=='__main__':main()
