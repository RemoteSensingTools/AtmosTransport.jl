#!/usr/bin/env python3
"""
Download ERA5 hourly spectral data (VO, D, LNSP) for Dec 2021.
One validated daily file per field, containing all 24 hourly time steps.
Output: ~/data/AtmosTransport/met/era5/spectral_hourly/
"""
import os
import sys
import time as _time

import cdsapi
from eccodes import (codes_get_long, codes_get_string,
                     codes_grib_new_from_file, codes_release)

OUTDIR = os.path.expanduser("~/data/AtmosTransport/met/era5/spectral_hourly")

YEAR = "2021"
MONTH = "12"
DAYS = [f"{d:02d}" for d in range(1, 32)]
TIMES = [f"{h:02d}:00:00" for h in range(24)]


def expected_inventory(date, param_ids, levels):
    date_id = int(date.replace("-", ""))
    return {
        (param_id, date_id, hour * 100, 0, level)
        for param_id in param_ids
        for hour in range(24)
        for level in levels
    }


def validate_grib(path, expected, label):
    """Require the requested inventory and native T639 spectral representation."""
    seen = set()
    count = 0
    try:
        with open(path, "rb") as stream:
            while True:
                gid = codes_grib_new_from_file(stream)
                if gid is None:
                    break
                try:
                    key = (
                        codes_get_long(gid, "paramId"),
                        codes_get_long(gid, "dataDate"),
                        codes_get_long(gid, "dataTime"),
                        codes_get_long(gid, "step"),
                        codes_get_long(gid, "level"),
                    )
                    representation = {
                        "edition": codes_get_long(gid, "edition"),
                        "gridType": codes_get_string(gid, "gridType"),
                        "typeOfLevel": codes_get_string(gid, "typeOfLevel"),
                        "dataType": codes_get_string(gid, "dataType"),
                        "stepType": codes_get_string(gid, "stepType"),
                        "packingType": codes_get_string(gid, "packingType"),
                        "J": codes_get_long(gid, "J"),
                        "K": codes_get_long(gid, "K"),
                        "M": codes_get_long(gid, "M"),
                    }
                finally:
                    codes_release(gid)
                expected_representation = {
                    "edition": 1,
                    "gridType": "sh",
                    "typeOfLevel": "hybrid",
                    "dataType": "an",
                    "stepType": "instant",
                    "packingType": "spectral_complex",
                    "J": 639,
                    "K": 639,
                    "M": 639,
                }
                for name, wanted in expected_representation.items():
                    if representation[name] != wanted:
                        raise ValueError(
                            f"message {key} has {name}={representation[name]!r}; "
                            f"expected {wanted!r}")
                count += 1
                if key in seen:
                    raise ValueError(f"duplicate GRIB message {key}")
                seen.add(key)
        if seen != expected:
            missing = len(expected - seen)
            unexpected = len(seen - expected)
            raise ValueError(
                f"inventory mismatch: {missing} missing and {unexpected} unexpected "
                f"messages (read {count}, expected {len(expected)})")
        return True
    except Exception as exc:
        print(f"  INVALID {label}: {exc}", flush=True)
        return False


def download(client, request, outfile, label, expected):
    if os.path.isfile(outfile) and validate_grib(outfile, expected, label):
        size_mb = os.path.getsize(outfile) / 1e6
        print(f"  SKIP validated ({size_mb:.0f} MB): {os.path.basename(outfile)}",
              flush=True)
        return True

    part = outfile + ".part"
    print(f"  Downloading {label} -> {os.path.basename(outfile)} ...", flush=True)
    for attempt in range(3):
        try:
            if os.path.exists(part):
                os.remove(part)
            client.retrieve("reanalysis-era5-complete", request, part)
            if not validate_grib(part, expected, label):
                raise RuntimeError("downloaded GRIB failed inventory validation")
            os.replace(part, outfile)
            size_mb = os.path.getsize(outfile) / 1e6
            print(f"  OK validated ({size_mb:.0f} MB): {os.path.basename(outfile)}",
                  flush=True)
            return True
        except Exception as exc:
            if os.path.exists(part):
                os.remove(part)
            print(f"  ERROR (attempt {attempt + 1}/3): {exc}", flush=True)
            if attempt < 2:
                _time.sleep(60)
    return False


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    client = cdsapi.Client()
    failures = []

    for day in DAYS:
        date = f"{YEAR}-{MONTH}-{day}"
        datestr = f"{YEAR}{MONTH}{day}"
        day_ok = True

        vo_d_file = os.path.join(OUTDIR, f"era5_spectral_{datestr}_vo_d.gb")
        day_ok &= download(client, {
            "class": "ea",
            "date": date,
            "expver": "1",
            "levelist": "/".join(str(i) for i in range(1, 138)),
            "levtype": "ml",
            "param": "138/155",
            "stream": "oper",
            "time": "/".join(TIMES),
            "type": "an",
        }, vo_d_file, f"VO+D {date}",
            expected_inventory(date, (138, 155), range(1, 138)))

        lnsp_file = os.path.join(OUTDIR, f"era5_spectral_{datestr}_lnsp.gb")
        day_ok &= download(client, {
            "class": "ea",
            "date": date,
            "expver": "1",
            "levelist": "1",
            "levtype": "ml",
            "param": "152",
            "stream": "oper",
            "time": "/".join(TIMES),
            "type": "an",
        }, lnsp_file, f"LNSP {date}",
            expected_inventory(date, (152,), (1,)))

        if day_ok:
            print(f"  Day {date} done.", flush=True)
        else:
            failures.append(date)
            print(f"  Day {date} FAILED.", flush=True)

    if failures:
        print(f"Failed days: {', '.join(failures)}", file=sys.stderr, flush=True)
        return 1
    print("All done!", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
