#!/bin/bash
# Detached finisher: wait for the ERA5 December rerun to exit, verify its SF6
# emission deficit (post diffusion-anomaly fix), then render the 3-way animation
# with dm + own-flux |F per run. Fully independent of any interactive session.
set -u
LOG=/tmp/finish_era5_anim.log
REPO=/home/cfranken/code/gitHub/AtmosTransportModel
CW=/home/cfranken/data/AtmosTransport/output/campaign_winter2021
ERA5_PID=3124066
OUT=/home/cfranken/www/catrine/column_mean_3way_dec_4tracer_postfix.mp4

echo "=== $(date) finisher start; waiting on ERA5 PID $ERA5_PID ===" >> "$LOG"
while kill -0 "$ERA5_PID" 2>/dev/null; do sleep 30; done
echo "=== $(date) ERA5 proc exited; letting nc flush/close (45s) ===" >> "$LOG"
sleep 45

# --- verify ERA5 deficit (post-fix vs preserved pre-fix) ---
python3 >> "$LOG" 2>&1 <<'PY'
import netCDF4 as nc, numpy as np
M_AIR=28.9644
CW='/home/cfranken/data/AtmosTransport/output/campaign_winter2021/'
TR=[('sf6',146.06,3.201820e-1),('co2_fossil',44.01,1.229399e6),('rn222',222.0,4.443261e-7)]
for path,lbl in [(CW+'era5_4tracer_dec2021_feb2022.nc','ERA5 POST-FIX'),
                 (CW+'era5_4tracer_dec2021_feb2022_PREFIX_diffbug.nc','ERA5 PRE-FIX')]:
    try:
        f=nc.Dataset(path,'r'); t=np.array(f.variables['time'][:])
        am0=np.array(f.variables['air_mass'][0],dtype=np.float64); amL=np.array(f.variables['air_mass'][-1],dtype=np.float64)
        print(f'--- {lbl}  t={t[-1]-t[0]:.0f}h ---')
        for v,M,rate in TR:
            q0=np.array(f.variables[v][0],dtype=np.float64); qL=np.array(f.variables[v][-1],dtype=np.float64)
            dm=(np.sum(qL*amL)-np.sum(q0*am0))*M/M_AIR; iF=rate*(t[-1]-t[0])*3600
            tag=' (decay sink; deficit not meaningful)' if v=='rn222' else ''
            print(f'  {v:11s}: dm={dm:.4e} |F={iF:.4e} deficit={(1-dm/iF)*100:+.3f}% NaN={np.isnan(qL).any()}{tag}')
        f.close()
    except Exception as e:
        print(f'{lbl}: ERROR {e}')
PY

# --- render animation: 2-frame smoke test, then full ---
cd "$REPO" || exit 1
echo "=== $(date) animation smoke test (2 frames) ===" >> "$LOG"
if ANIM_MAXFRAMES=2 python3 scripts/completed_experiments/animate_column_mean_3way_dec.py /tmp/anim_smoke.mp4 >> "$LOG" 2>&1; then
    echo "=== $(date) smoke OK; full render -> $OUT ===" >> "$LOG"
    python3 scripts/completed_experiments/animate_column_mean_3way_dec.py "$OUT" >> "$LOG" 2>&1
    echo "=== $(date) finisher DONE (output: $OUT or .gif fallback) ===" >> "$LOG"
else
    echo "=== $(date) SMOKE TEST FAILED — see above; full render skipped ===" >> "$LOG"
fi
