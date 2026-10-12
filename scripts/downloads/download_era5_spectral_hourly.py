#!/usr/bin/env python3
"""
Download ERA5 hourly spectral data (VO, D, LNSP) for Dec 2021.
One daily file per field, containing all 24 hourly time steps.
Output: ~/data/AtmosTransport/met/era5/spectral_hourly/
"""
import os
import sys
import time as _time
import cdsapi

OUTDIR = os.path.expanduser("~/data/AtmosTransport/met/era5/spectral_hourly")
os.makedirs(OUTDIR, exist_ok=True)

YEAR = "2021"
MONTH = "12"
DAYS = [f"{d:02d}" for d in range(1, 32)]
TIMES = [f"{h:02d}:00:00" for h in range(24)]  # ALL 24 hours

c = cdsapi.Client()


def download(request, outfile, label):
    if os.path.exists(outfile) and os.path.getsize(outfile) > 1e6:
        sz = os.path.getsize(outfile) / 1e6
        print(f"  SKIP ({sz:.0f} MB): {os.path.basename(outfile)}", flush=True)
        return True
    print(f"  Downloading {label} -> {os.path.basename(outfile)} ...", flush=True)
    for attempt in range(3):
        try:
            c.retrieve("reanalysis-era5-complete", request, outfile)
            sz = os.path.getsize(outfile) / 1e6
            print(f"  OK ({sz:.0f} MB): {os.path.basename(outfile)}", flush=True)
            return True
        except Exception as e:
            print(f"  ERROR (attempt {attempt+1}/3): {e}", flush=True)
            if attempt < 2:
                _time.sleep(60)
    return False


for day in DAYS:
    date = f"{YEAR}-{MONTH}-{day}"
    datestr = f"{YEAR}{MONTH}{day}"

    # VO + D (combined in one file per day, all 24 hours, all 137 levels)
    vo_d_file = os.path.join(OUTDIR, f"era5_spectral_{datestr}_vo_d.gb")
    download({
        "class": "ea",
        "date": date,
        "expver": "1",
        "levelist": "/".join(str(i) for i in range(1, 138)),
        "levtype": "ml",
        "param": "138/155",  # VO + D combined
        "stream": "oper",
        "time": "/".join(TIMES),
        "type": "an",
    }, vo_d_file, f"VO+D {date}")

    # LNSP (single level, all 24 hours)
    lnsp_file = os.path.join(OUTDIR, f"era5_spectral_{datestr}_lnsp.gb")
    download({
        "class": "ea",
        "date": date,
        "expver": "1",
        "levelist": "1",
        "levtype": "ml",
        "param": "152",  # LNSP
        "stream": "oper",
        "time": "/".join(TIMES),
        "type": "an",
    }, lnsp_file, f"LNSP {date}")

    print(f"  Day {date} done.", flush=True)

print("All done!", flush=True)
