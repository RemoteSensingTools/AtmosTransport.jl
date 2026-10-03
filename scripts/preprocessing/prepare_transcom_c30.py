#!/usr/bin/env python3
"""Partition existing C30 TRENDY/GFED drivers into 11 land TransCom regions.

This tags the already remapped C30 fluxes, not the original fine-grid sources.
Within a C30 cell, use normalized sampled land-region areas. Cells with no
sampled land inherit the closest labeled land pixel; report their flux share.
The resulting nonnegative weights sum to one everywhere, preserving drivers.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
from pathlib import Path

import netCDF4
import numpy as np
from scipy.spatial import cKDTree

import prepare_trendy_s3_gpp_ter_c90 as T

START = dt.date(2014,9,1)
NDAYS = (dt.date(2025,1,1)-START).days
CONVERSION = 86400*0.02896546/0.0440095


def save_json(path, obj):
    tmp = path.with_suffix('.partial.json')
    tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)


def sources(parent, fire):
    manifest=json.loads((parent/'fluxes/manifest.json').read_text())
    result=[]
    for model,audit in manifest['models'].items():
        if audit['status']=='complete':
            result.append({'model':model,'file':audit['output'],
                           'variables':{'npp':'NPP_CO2_FLUX','rh':'RH_CO2_FLUX'}})
    assert len(result)==23
    result.append({'model':'GFED5','file':str(fire),'variables':{'fire':'FIRE_CO2_FLUX'}})
    return result


def normalize_land_weights(raw, fallback_region):
    """Normalize fractional land coverage; use 0-based fallback where absent."""
    assert raw.shape[0]==11 and np.isfinite(raw).all() and (raw>=0).all()
    sums=raw.sum(axis=0)
    missing=sums==0
    w=np.divide(raw,sums[None],out=np.zeros_like(raw,dtype='f8'),where=sums[None]>0)
    w[fallback_region[missing],np.where(missing)[0]]=1
    np.testing.assert_allclose(w.sum(axis=0),1,rtol=0,atol=5e-16)
    return w,missing


def build_mask(mask, geometry, root):
    with netCDF4.Dataset(geometry) as ds:
        lat=np.asarray(ds['lats'][:]);lon=np.asarray(ds['lons'][:]);area=np.asarray(ds['cell_area'][:])
    assert lat.shape==(6,30,30)
    with netCDF4.Dataset(mask) as ds:
        labels=np.asarray(ds['transcom_regions'][:])
        slat=np.asarray(ds['latitude'][:]);slon=np.asarray(ds['longitude'][:])
        names=[s.strip() for s in netCDF4.chartostring(ds['transcom_names'][:])[:11]]
    np.testing.assert_allclose(slat,np.arange(-89.5,90,1))
    np.testing.assert_allclose(slon,np.arange(-179.5,180,1))
    fl,fo,target,farea=T.fine_grid_target_map(lat,lon,.25)
    fine_labels=labels[np.floor(fl+90).astype(int)[:,None],np.floor(fo+180).astype(int)[None,:]].ravel()
    raw=np.stack([np.bincount(target[fine_labels==k],weights=farea[fine_labels==k],minlength=lat.size)
                  for k in range(1,12)])
    ii,jj=np.where((labels>=1)&(labels<=11))
    tree=cKDTree(T.xyz(slat[ii],slon[jj]))
    distance,nearest=tree.query(T.xyz(lat.ravel(),lon.ravel()))
    fallback=labels[ii[nearest],jj[nearest]]-1
    w,missing=normalize_land_weights(raw,fallback)
    path=root/'transcom_weights_c30.nc'
    with netCDF4.Dataset(path,'w') as ds:
        for k,n in [('region',11),('nf',6),('Ydim',30),('Xdim',30)]:ds.createDimension(k,n)
        for k,x in [('lats',lat),('lons',lon),('cell_area',area)]:
            ds.createVariable(k,'f8',('nf','Ydim','Xdim'))[:]=x
        ds.createVariable('region_id','i4',('region',))[:]=np.arange(1,12)
        ds.createVariable('weight','f8',('region','nf','Ydim','Xdim'))[:]=w.reshape(11,6,30,30)
        ds.createVariable('nearest_land_fallback','i1',('nf','Ydim','Xdim'))[:]=missing.reshape(6,30,30)
        ds.createVariable('nearest_land_distance_km','f8',('nf','Ydim','Xdim'))[:]=(2*T.R_EARTH*np.arcsin(np.clip(distance/2,0,1))/1000).reshape(6,30,30)
        ds.region_names=json.dumps(names)
        ds.method='C30 tagging; normalized 0.25-degree sampled land-region areas; closest labeled land pixel for cells without sampled land'
        ds.mask_source=str(mask);ds.mask_variable='transcom_regions'
        ds.mask_sha256=hashlib.sha256(mask.read_bytes()).hexdigest()
    save_json(root/'regions.json',[{'id':i+1,'name':n} for i,n in enumerate(names)])
    return w,missing,area.ravel()


def time_indices(ds):
    t=ds['time'];dates=netCDF4.num2date(t[:],t.units,calendar=getattr(t,'calendar','standard'))
    keys=[dt.date(d.year,d.month,d.day) for d in dates]
    first=keys.index(START)
    assert keys[first:first+NDAYS]==[START+dt.timedelta(days=i) for i in range(NDAYS)]
    return first


def audit_partition(specs,w,missing,area,root):
    """Check the actual FP32 partition sum for every source day and cell."""
    checks=[]
    for source in specs:
        with netCDF4.Dataset(source['file']) as ds:
            first=time_indices(ds)
            for comp,var in source['variables'].items():
                total=fallback=error=0.;maxcell=0.
                for i in range(first,first+NDAYS,64):
                    x=np.asarray(ds[var][i:min(i+64,first+NDAYS)],dtype='f8').reshape(-1,5400)
                    assert np.isfinite(x).all()
                    combined=np.zeros_like(x)
                    for weights in w:combined+=(x*weights).astype('f4').astype('f8')
                    delta=abs(combined-x)
                    # Relative to each nonzero source value, not its signed net.
                    active=x!=0
                    if active.any():maxcell=max(maxcell,float(np.max(delta[active]/abs(x[active]))))
                    total+=float((abs(x)*area).sum())
                    fallback+=float((abs(x[:,missing])*area[missing]).sum())
                    error+=float((delta*area).sum())
                assert maxcell<2e-7,(source['model'],comp,maxcell)
                row={'model':source['model'],'component':comp,
                     'max_cell_relative_partition_error':maxcell,
                     'absolute_flux_weighted_partition_error':error/max(total,1e-30),
                     'nearest_land_fallback_absolute_flux_fraction':fallback/max(total,1e-30)}
                checks.append(row)
        print(f"Partition verified: {source['model']}",flush=True)
    save_json(root/'partition_validation.json',{'passed':True,'checks':checks,
               'tolerance': '2e-7 pointwise relative to nonzero original FP32 flux',
               'interpretation':'Regions partition the existing C30 flux field; source-grid region tags are not reconstructed.'})


def prepare_region(region,specs,w,root):
    folder=root/'fluxes'/f'tc{region:02d}';folder.mkdir(parents=True,exist_ok=True)
    result=[]
    for source in specs:
        outfile=folder/(source['model']+'.nc')
        auditfile=outfile.with_suffix('.json')
        if auditfile.exists() and outfile.exists():
            result.extend(json.loads(auditfile.read_text()));continue
        tmp=outfile.with_suffix('.partial.nc')
        entries=[]
        with netCDF4.Dataset(source['file']) as ds,netCDF4.Dataset(tmp,'w') as out:
            first=time_indices(ds)
            for k,n in [('time',NDAYS),('nf',6),('Ydim',30),('Xdim',30)]:out.createDimension(k,n)
            t=out.createVariable('time','f8',('time',));t.units='hours since 2014-09-01 00:00:00 UTC'
            t.calendar='proleptic_gregorian';t[:]=np.arange(NDAYS)*24
            for k in ['lats','lons','cell_area']:
                out.createVariable(k,'f8',('nf','Ydim','Xdim'))[:]=ds[k][:]
            area=np.asarray(ds['cell_area'][:],dtype='f8')
            for comp,var in source['variables'].items():
                v=out.createVariable(var,'f4',('time','nf','Ydim','Xdim'),zlib=True,complevel=1,
                                     shuffle=True,chunksizes=(1,6,30,30))
                v.units='kg CO2 m-2 s-1';v.positive='to_atmosphere'
                full=smoke=0.
                for i in range(0,NDAYS,64):
                    j=min(i+64,NDAYS)
                    x=(np.asarray(ds[var][first+i:first+j],dtype='f8')*w[region-1].reshape(6,30,30)).astype('f4')
                    assert np.isfinite(x).all()
                    v[i:j]=x
                    daily=(x.astype('f8')*area).sum(axis=(1,2,3))*CONVERSION
                    full+=float(daily.sum())
                    if i==0:smoke=float(daily[:3].sum())
                modelslug=source['model'].lower().replace('-','_')
                name=f'co2_{modelslug}_{comp}_tc{region:02d}'
                entries.append({'name':name,'model':source['model'],'component':comp,
                                'region':region,'file':str(outfile),'variable':var,
                                'carrier':1.3e-3 if comp=='npp' else 1e-5,
                                'expected_full_storage_change_kg':full,
                                'expected_smoke_storage_change_kg':smoke})
            out.transcom_region=region;out.parent_flux=source['file']
            out.mask_weights=str(root/'transcom_weights_c30.nc')
        tmp.replace(outfile);save_json(auditfile,entries);result.extend(entries)
        print(f"Prepared tc{region:02d}: {source['model']}",flush=True)
    assert len(result)==47
    return result
