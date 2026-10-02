#!/usr/bin/env python3
"""Rank transported TRENDY + GFED5 against OCO-2 seasonal and IAV Hovmollers.

Requires a successfully validated campaign. All regressions use identical
observed band-months. The observation cache and its extraction source are
preserved with the analysis. No amplitude scaling, time shifting or smoothing
is applied to the primary scores.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import netCDF4
import numpy as np


def save_json(path, obj):
    def finite(value):
        if isinstance(value, dict):
            return {k:finite(v) for k,v in value.items()}
        if isinstance(value, (list, tuple)):
            return [finite(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    tmp = path.with_suffix('.partial.json')
    tmp.write_text(json.dumps(finite(obj), indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corr(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def metrics(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    a, b = x[ok], y[ok]
    return {'r': corr(a, b), 'rmse_ppm': float(np.sqrt(np.mean((a-b)**2))),
            'bias_ppm': float(np.mean(a-b)),
            'amplitude_ratio': float(np.std(a)/np.std(b)),
            'n': int(ok.sum())}


def decompose(fields, seen):
    """Joint least squares: 12 monthly intercepts + linear trend per band.

    fields has shape (series, 120 months, 40 bands). Missing seasonal months
    are excluded. Require >=5 observed years per calendar-month/band for the
    climatology comparison. IAV retains all sufficiently sampled band-months.
    """
    ns, nt, nb = fields.shape
    t = (np.arange(nt) - (nt-1)/2) / 12
    mo = np.arange(nt) % 12
    design = np.column_stack([np.eye(12)[mo], t])
    anomaly = np.full_like(fields, np.nan)
    seasonal = np.full((ns, 12, nb), np.nan)
    detrended = np.full_like(fields, np.nan)
    for b in range(nb):
        ok = seen[:, b]
        if ok.sum() < 24:
            continue
        values = fields[:, ok, b].T
        values = values-values.mean(axis=0)
        coef = np.linalg.lstsq(design[ok], values, rcond=None)[0]
        anomaly[:, ok, b] = (values-design[ok]@coef).T
        detrended[:, ok, b] = values.T - coef[-1, :, None]*t[ok]
        detrended[:, ok, b] -= detrended[:, ok, b].mean(axis=1)[:, None]
        valid_month = np.bincount(mo[ok], minlength=12) >= 5
        cycle = coef[:12].T
        cycle[:, ~valid_month] = np.nan
        cycle -= np.nanmean(cycle, axis=1)[:, None]
        seasonal[:, :, b] = cycle
    return anomaly, seasonal, detrended


def demean_space(field):
    n = np.isfinite(field).sum(axis=-1, keepdims=True)
    mean = np.divide(np.nansum(field, axis=-1, keepdims=True), n,
                     out=np.zeros_like(n, dtype=float), where=n>0)
    return field-mean


def extract(args, models, out):
    dates = np.arange('2015-01-01', '2025-01-01', dtype='datetime64[D]')
    months = np.arange('2015-01', '2025-01', dtype='datetime64[M]')
    first = args.run_root/'output/trendy_allmodels_c30_20150101.nc'
    firefirst = args.fire_dir/'gfed5_fire_c30_20150101.nc'
    with netCDF4.Dataset(first) as ds, netCDF4.Dataset(firefirst) as fs:
        for k in ('lats', 'lons', 'cell_area'):
            np.testing.assert_allclose(ds[k][:], fs[k][:], rtol=1e-7, atol=1e-7)
        for k in ('mass_basis', 'cs_definition', 'panel_convention'):
            assert ds.getncattr(k) == fs.getncattr(k), k
        lat = np.asarray(ds['lats'][:]).ravel()
        area = np.asarray(ds['cell_area'][:], dtype='f8').ravel()
    edges = np.arange(41)/40*2-1
    bands = np.clip(np.digitize(np.sin(np.deg2rad(lat)), edges)-1, 0, 39)
    denom = np.bincount(bands, weights=area, minlength=40)
    assert (denom > 0).all()
    raw = np.zeros((len(models), 120, 40))
    fire = np.zeros((120, 40))
    counts = np.zeros(120, dtype=int)
    names = [[f"co2_{m.lower().replace('-', '_')}_{c}_column_mean"
              for c in ('npp', 'rh')] for m in models]
    for i, date in enumerate(dates):
        tag = str(date).replace('-', '')
        mi = int(np.searchsorted(months, date.astype('datetime64[M]')))
        expected_time = float((date-np.datetime64('2014-09-01')).astype(int)+1)*24
        with netCDF4.Dataset(args.run_root/f'output/trendy_allmodels_c30_{tag}.nc') as ds:
            np.testing.assert_allclose(ds['time'][:], [expected_time])
            for j, (npp, rh) in enumerate(names):
                assert ds[npp].units == ds[rh].units == 'mol mol-1 dry'
                # Convert to F64 before summing the large opposite carriers.
                col = (np.asarray(ds[npp][-1], dtype='f8').ravel()-1.3e-3
                       +np.asarray(ds[rh][-1], dtype='f8').ravel()-1e-5)*1e6
                assert np.isfinite(col).all(), (date, npp)
                raw[j, mi] += np.bincount(bands, weights=col*area, minlength=40)/denom
        with netCDF4.Dataset(args.fire_dir/f'gfed5_fire_c30_{tag}.nc') as ds:
            np.testing.assert_allclose(ds['time'][:], [expected_time])
            assert ds['co2_gfed_fire_column_mean'].units == 'mol mol-1 dry'
            col = (np.asarray(ds['co2_gfed_fire_column_mean'][-1], dtype='f8').ravel()-1e-5)*1e6
            assert np.isfinite(col).all(), date
            fire[mi] += np.bincount(bands, weights=col*area, minlength=40)/denom
        counts[mi] += 1
        if i % 250 == 0:
            print(f'Extracted {i+1}/{len(dates)} days: {date}', flush=True)
    expected_counts = np.diff(np.arange('2015-01', '2025-02', dtype='datetime64[M]').astype('datetime64[D]')).astype(int)
    np.testing.assert_array_equal(counts, expected_counts)
    raw /= counts[None, :, None]
    fire /= counts[:, None]
    np.savez_compressed(out/'monthly_fields.npz', months=months, models=models,
                        nee_ppm=raw, fire_ppm=fire, sin_edges=edges, days_per_month=counts)
    return months, raw, fire, edges


def write_csv(path, rows):
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def make_plots(out, months, edges, models, rankings, fields, summary):
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10})
    for kind, allfields in fields.items():
        seasonal = kind == 'seasonal'
        xedges = np.arange(13) if seasonal else np.arange(121)
        vmax = float(np.nanpercentile(np.abs(allfields), 99))
        order = [models.index(r['model']) for r in rankings[kind]]
        # Observations repeated on every page, same scale for all models.
        with PdfPages(out/f'{kind}_hovmoller_atlas.pdf') as pdf:
            for page, start in enumerate(range(0, len(order), 6)):
                subset = order[start:start+6]
                fig, axs = plt.subplots(len(subset)+1, 1, figsize=(12, 2.1*(len(subset)+1)),
                                        sharex=True, sharey=True, layout='constrained')
                for ax, idx in zip(axs, [-1]+subset):
                    ax.set_facecolor('#d5d9de')
                    field = allfields[0] if idx == -1 else allfields[idx+1]
                    mesh = ax.pcolormesh(xedges, edges, field.T, cmap='RdBu_r',
                                         vmin=-vmax, vmax=vmax, shading='flat')
                    if idx == -1:
                        title = 'OCO-2 observed XCO2'
                    else:
                        row = next(r for r in rankings[kind] if r['model']==models[idx])
                        title = f"#{row['rank']} {models[idx]} + GFED5   r={row['r']:.3f}, RMSE={row['rmse_ppm']:.3f} ppm"
                        if not row.get('budget_passed', True):
                            title += '  [PROVISIONAL: budget check failed]'
                    ax.set_title(title, loc='left')
                    lat = [-60, -30, 0, 30, 60]
                    ax.set_yticks(np.sin(np.deg2rad(lat)), [str(a) for a in lat])
                    ax.set_ylabel('Latitude')
                if seasonal:
                    axs[-1].set_xticks(np.arange(12)+.5, list('JFMAMJJASOND'))
                else:
                    axs[-1].set_xticks(np.arange(0,120,12), [str(y) for y in range(2015,2025)])
                label = {'seasonal':'Mean seasonal cycle', 'anomaly':'Interannual anomalies',
                         'detrended':'Detrended monthly variations'}[kind]
                fig.suptitle(label+
                             ' | TRENDY v14 S3 + GFED5 vs OCO-2 | 2015–2024')
                fig.colorbar(mesh, ax=axs, fraction=.025, pad=.01, label='XCO2 (ppm)')
                pdf.savefig(fig)
                fig.savefig(out/f'{kind}_hovmoller_{page+1}.png', dpi=140)
                plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(14, 9), layout='constrained')
    rows = summary
    y = np.arange(len(rows))
    axs[0].barh(y-.17, [r['anomaly_r'] for r in rows], .34, label='Interannual anomalies', color='#267985')
    axs[0].barh(y+.17, [r['seasonal_r'] for r in rows], .34, label='Seasonal cycle', color='#cf742b')
    axs[0].set_yticks(y, [r['model']+(' *' if not r.get('budget_passed',True) else '') for r in rows]); axs[0].invert_yaxis()
    axs[0].set_xlabel('Correlation with OCO-2'); axs[0].legend()
    axs[0].set_title('Ordered by anomaly agreement')
    for r in rows:
        axs[1].scatter(r['seasonal_rank'],r['anomaly_rank'],s=30,color='#267985')
        axs[1].annotate(r['model']+(' *' if not r.get('budget_passed',True) else ''),(r['seasonal_rank'],r['anomaly_rank']),
                        xytext=(4,3),textcoords='offset points',fontsize=7)
    axs[1].plot([1,23],[1,23],color='#888888',ls='--',lw=1)
    axs[1].set(xlabel='Seasonal rank (1 = best)',ylabel='Anomaly rank (1 = best)',
               xlim=(0,27),ylim=(25,0),title='Does seasonal skill carry over to anomalies?')
    for ax in axs: ax.grid(alpha=.2); ax.set_axisbelow(True)
    if any(not r.get('budget_passed',True) for r in rows):
        fig.suptitle('* Provisional model: at least one tracer failed the original budget tolerance')
    fig.savefig(out/'seasonal_vs_anomaly_ranking.png',dpi=180)
    fig.savefig(out/'seasonal_vs_anomaly_ranking.pdf')
    plt.close(fig)


def analyze(args, out):
    manifest = json.loads((args.run_root/'fluxes/manifest.json').read_text())
    models = [k for k,v in manifest['models'].items() if v['status']=='complete']
    assert len(models)==23
    validation = json.loads((args.run_root/'validation.json').read_text())
    failed_checks = {k:v for k,v in validation.get('checks',{}).items() if not v['passed']}
    months, raw, fire, edges = extract(args, models, out)
    with np.load(args.observations, allow_pickle=False) as z:
        np.testing.assert_array_equal(z['months'].astype('datetime64[M]'), months)
        np.testing.assert_allclose(z['sin_edges'], edges)
        obs = np.where(z['eq_n'] >= args.min_soundings, z['eq'], np.nan)
        obs_counts = z['eq_n'].copy()
    seen = np.isfinite(obs)
    combined = raw+fire[None]
    # Apply one linear projection to observations, models, and no-fire controls.
    inputs = np.concatenate([obs[None], combined, raw, fire[None]], axis=0)
    anomaly, seasonal, detrended = decompose(inputs, seen)
    nm = len(models)
    rankings = {}
    per_band, per_month = [], []
    for kind, allfields in [('anomaly', anomaly), ('seasonal', seasonal), ('detrended', detrended)]:
        target = allfields[0]
        rows = []
        for j, model in enumerate(models):
            field = allfields[j+1]
            prefix = f"co2_{model.lower().replace('-', '_')}_"
            row = {'model':model, 'budget_passed':not any(k.startswith(prefix) for k in failed_checks),
                   **metrics(field, target)}
            plain = metrics(allfields[nm+j+1], target)
            row.update(no_fire_r=plain['r'], delta_r_fire=row['r']-plain['r'],
                       no_fire_rmse_ppm=plain['rmse_ppm'])
            row['spatial_structure_r'] = corr(demean_space(field), demean_space(target))
            br = [corr(field[:,b],target[:,b]) for b in range(40)]
            tr = [corr(field[t],target[t]) for t in range(len(target))]
            row['median_band_temporal_r'] = float(np.nanmedian(br))
            row['median_month_spatial_r'] = float(np.nanmedian(tr))
            for b, r in enumerate(br):
                per_band.append({'kind':kind,'model':model,'band':b,
                                 'latitude':float(np.rad2deg(np.arcsin((edges[b]+edges[b+1])/2))),
                                 **metrics(field[:,b],target[:,b])} if np.isfinite(r) else
                                {'kind':kind,'model':model,'band':b,'latitude':float(np.rad2deg(np.arcsin((edges[b]+edges[b+1])/2))),
                                 'r':float('nan'),'rmse_ppm':float('nan'),'bias_ppm':float('nan'),'amplitude_ratio':float('nan'),'n':int(np.isfinite(target[:,b]).sum())})
            for t, r in enumerate(tr):
                per_month.append({'kind':kind,'model':model,'month':str(months[t]) if kind!='seasonal' else t+1,'spatial_r':r})
            rows.append(row)
        rows.sort(key=lambda r:(-r['r'],r['rmse_ppm']))
        rmse_order = sorted(rows,key=lambda r:r['rmse_ppm'])
        for i,row in enumerate(rows):
            row['rank']=i+1
            row['rmse_rank']=rmse_order.index(row)+1
        rankings[kind]=rows
        write_csv(out/f'{kind}_ranking.csv',rows)
    write_csv(out/'latitude_band_scores.csv',per_band)
    write_csv(out/'monthly_spatial_scores.csv',per_month)
    summary = []
    for a in rankings['anomaly']:
        s = next(r for r in rankings['seasonal'] if r['model']==a['model'])
        summary.append({'model':a['model'],'budget_passed':a['budget_passed'],
                        'anomaly_rank':a['rank'],'seasonal_rank':s['rank'],
                        'rank_change_seasonal_to_anomaly':s['rank']-a['rank'],
                        'anomaly_r':a['r'],'seasonal_r':s['r'],
                        'anomaly_rmse_ppm':a['rmse_ppm'],'seasonal_rmse_ppm':s['rmse_ppm']})
    write_csv(out/'seasonal_vs_anomaly.csv',summary)
    # Leave one calendar year out of scoring, keeping the declared full-period
    # anomaly definition fixed. This is rank sensitivity, not out-of-sample CV.
    sensitivity=[]
    rank_samples = np.zeros((10,nm),dtype=int)
    for yi,year in enumerate(range(2015,2025)):
        keep = np.arange(120)//12 != yi
        rs = [corr(anomaly[j+1,keep],anomaly[0,keep]) for j in range(nm)]
        order = np.argsort(-np.asarray(rs))
        for rank,j in enumerate(order,1):
            rank_samples[yi,j]=rank
            sensitivity.append({'omitted_year':year,'model':models[j],'rank':rank,'r':rs[j]})
    write_csv(out/'anomaly_leave_one_year_out.csv',sensitivity)
    stats = {'rank_spearman':corr(np.array([r['seasonal_rank'] for r in summary]),
                                 np.array([r['anomaly_rank'] for r in summary])),
             'observed_band_months':int(seen.sum()), 'total_band_months':int(seen.size),
             'seasonal_band_months':int(np.isfinite(seasonal[0]).sum()),
             'fire_only_anomaly':metrics(anomaly[-1],anomaly[0]),
             'anomaly_winner':rankings['anomaly'][0], 'seasonal_winner':rankings['seasonal'][0]}
    stats['top_five_overlap'] = sorted({r['model'] for r in rankings['anomaly'][:5]} &
                                       {r['model'] for r in rankings['seasonal'][:5]})
    stats['largest_rank_changes'] = sorted(summary,
        key=lambda r:abs(r['rank_change_seasonal_to_anomaly']),reverse=True)[:5]
    stats['validation_passed'] = validation['passed']
    stats['failed_budget_checks'] = failed_checks
    save_json(out/'summary.json',stats)
    np.savez_compressed(out/'comparison_fields.npz',months=months,models=models,sin_edges=edges,
                        observations_ppm=obs,observation_counts=obs_counts,
                        anomaly_ppm=anomaly[:nm+1],seasonal_ppm=seasonal[:nm+1],
                        detrended_ppm=detrended[:nm+1],anomaly_no_fire_ppm=anomaly[nm+1:2*nm+1])
    make_plots(out,months,edges,models,rankings,
               {'anomaly':anomaly[:nm+1],'seasonal':seasonal[:nm+1],
                'detrended':detrended[:nm+1]},summary)
    lines = ['# TRENDY + GFED5 Hovmöller intercomparison', '',
             f"Anomaly winner: **{stats['anomaly_winner']['model']}**; seasonal winner: **{stats['seasonal_winner']['model']}**.",
             f"Seasonal/anomaly rank Spearman correlation: **{stats['rank_spearman']:.3f}** (23 models).", '',
             f"Top-five overlap: **{len(stats['top_five_overlap'])}/5** ({', '.join(stats['top_five_overlap']) or 'none'}).", '',
             '## Method', '',
             '- Period: January 2015–December 2024; integration begins September 2014.',
             '- Signal: (NPP tracer − 1300 ppm) + (Rh tracer − 10 ppm) + (GFED5 fire tracer − 10 ppm). NPP already has the atmospheric Ra−GPP sign. No fitted scaling of model or fire.',
             '- Monthly averages of daily end-of-day dry-air column snapshots, grouped by file day, following the earlier comparison. Native C30 cells use area weights in 40 equal-area sin(latitude) bands.',
             f'- OCO-2: the preserved earlier sounding-count-weighted XCO2 cache. At least {args.min_soundings} soundings per band-month; {seen.sum()}/{seen.size} cells retained. Each retained band-month has equal score weight.',
             '- Per band, fit a linear trend and 12 calendar-month intercepts jointly by exact least squares, using the same observation mask for every series. This refines the earlier iterative fit, which used full coverage for models and observed coverage for OCO-2.',
             '- Interannual anomalies are residuals of that joint fit. Seasonal scores compare the 12 fitted monthly intercepts after subtracting their band mean; require at least five observed years for each seasonal band-month. The separate detrended score retains both seasonality and interannual variation.',
             '- Primary ranking: pooled Pearson correlation across space and time, descending. RMSE in ppm, amplitude ratio, RMSE rank, no-fire comparison, and spatial/temporal diagnostics are reported separately. No smoothing, lag optimization, or amplitude normalization.',
             '- Spatial-structure correlation first removes each month’s mean across observed bands. Per-band temporal and per-month spatial correlations diagnose how pooled skill is distributed.',
             '- Leave-one-year-out scores omit a year only during scoring, with the full-period anomaly fit fixed. This assesses sensitivity to individual events, not predictive validation.', '',
             '## Results', '',
             '| Model | Anomaly rank | Anomaly r | RMSE (ppm) | Seasonal rank | Seasonal r | RMSE (ppm) |',
             '|---|---:|---:|---:|---:|---:|---:|']
    if failed_checks:
        lines[2:2] = ['**Validation caveat:** all daily fields passed completeness, finite-value and positive-carrier checks, but the original integrated budget gate failed for '+
                      ', '.join(f"{k} ({v['relative_error']:.5%} relative error)" for k,v in failed_checks.items())+
                      '. The original tolerance and failed report are preserved. Asterisked models are provisional diagnostics.', '']
    for r in summary:
        label = r['model']+(' *' if not r['budget_passed'] else '')
        lines.append(f"| {label} | {r['anomaly_rank']} | {r['anomaly_r']:.3f} | {r['anomaly_rmse_ppm']:.3f} | {r['seasonal_rank']} | {r['seasonal_r']:.3f} | {r['seasonal_rmse_ppm']:.3f} |")
    lines += ['', '## Interpretation limits', '',
              '- These are transported land-biosphere plus fire signals. OCO-2 also contains fossil-fuel, ocean and other contributions, so this is agreement with observed total XCO2 variations, not an isolated validation of land fluxes.',
              '- Observations are sounding-weighted while model bands are area-weighted. Matching band-month coverage does not reproduce within-band satellite sampling or apply retrieval averaging kernels. The result follows the previous zonal comparison and is not a sounding-level comparison.',
              '- The supplied historical GFED5 transport uses an older code snapshot. Its documented cumulative budget deficit was approximately 0.29%; it is reused without rescaling. Grid, column units and daily times are checked against the new run. Superposition of separately transported components can retain small limiter/FP32 differences.',
              '- Four months of initial integration are excluded. Longer spin-up sensitivity has not been tested. A common fitted linear trend removes secular changes but preserves nonlinear interannual changes.',
              '- Correlation rewards pattern and timing; RMSE also penalizes amplitude errors. Close rank differences are descriptive, not statistically significant claims.', '',
              '## Files', '',
              '- `seasonal_vs_anomaly_ranking.png`: score comparison and rank scatter.',
              '- `anomaly_hovmoller_atlas.pdf`, `seasonal_hovmoller_atlas.pdf`: all 23 models and repeated observation panels on shared scales.',
              '- `anomaly_ranking.csv`, `seasonal_ranking.csv`, `detrended_ranking.csv`: complete metrics.',
              '- `latitude_band_scores.csv`, `monthly_spatial_scores.csv`, `anomaly_leave_one_year_out.csv`: diagnostic detail.',
              '- `monthly_fields.npz`, `comparison_fields.npz`: reusable fields in ppm.',
              '- `provenance.json`, `inputs/`: input locations, hashes and observation extraction source.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    save_json(out/'provenance.json',{'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
              'script':str(Path(__file__).resolve()),'script_sha256':digest(Path(__file__)),
              'campaign':str(args.run_root),'campaign_provenance':json.loads((args.run_root/'provenance.json').read_text()),
              'validation':json.loads((args.run_root/'validation.json').read_text()),
              'observations':str(args.observations),'observations_sha256':digest(args.observations),
              'observation_original_source':'/kiwi-data/Data/satellite/OCO2/oco2_dashboard/oco2_daily.zarr',
              'fire_directory':str(args.fire_dir),'min_soundings':args.min_soundings,
              'expected_daily_files_per_source':3653,'ranking':'pooled Pearson r descending; RMSE tie-break'})
    print(json.dumps(stats,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-root',type=Path,required=True)
    ap.add_argument('--fire-dir',type=Path,required=True)
    ap.add_argument('--observations',type=Path,required=True)
    ap.add_argument('--min-soundings',type=int,default=2000)
    ap.add_argument('--wait',action='store_true',help='Wait for the detached campaign to pass validation')
    ap.add_argument('--allow-budget-failure',action='store_true',
                    help='Produce explicitly flagged diagnostic plots after complete output validation with failed integrated budgets; preserve the failed gate')
    args=ap.parse_args()
    out=args.run_root/'intercomparison';out.mkdir(exist_ok=True)
    lock=(out/'analysis.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    status=out/'status.json'
    def state(phase,**extra):
        save_json(status,{'phase':phase,'pid':os.getpid(),'updated_utc':dt.datetime.now(dt.timezone.utc).isoformat(),**extra})
    try:
        while True:
            campaign=json.loads((args.run_root/'status.json').read_text())
            if campaign['phase']=='complete': break
            if campaign['phase']=='failed':
                if args.allow_budget_failure and (args.run_root/'validation.json').exists():
                    v=json.loads((args.run_root/'validation.json').read_text())
                    assert v['days']==3775 and v['models']==23 and v['tracers']==46
                    assert len(v['checks'])==46 and 'final_validation exited' in campaign.get('error','')
                    print('Diagnostic analysis: preserving failed integrated budget checks and flagging affected models.',flush=True)
                    break
                raise RuntimeError(f'Campaign failed: {campaign}')
            if not args.wait: raise RuntimeError('Campaign is not complete; use --wait')
            os.kill(campaign['controller_pid'],0)
            state('waiting_for_validated_transport',campaign_phase=campaign['phase'])
            time.sleep(30)
        assert json.loads((args.run_root/'validation.json').read_text())['passed'] or args.allow_budget_failure
        state('analyzing')
        analyze(args,out)
        state('complete',report=str(out/'REPORT.md'),
              validation_passed=json.loads((args.run_root/'validation.json').read_text())['passed'])
    except Exception as exc:
        state('failed',error=str(exc))
        raise


if __name__=='__main__': main()
