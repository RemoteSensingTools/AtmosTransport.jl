#!/usr/bin/env python3
"""Copy timing records from a lab data folder into the site's data folder.

    python3 benchmarking/site/import_records.py <lab data> [--data benchmarking/site/data]

For every <device>/records.csv in <lab data>, writes <data>/<device>/records.csv with the site's columns
only and `result_dir` cut to its last three components (run, timing, configuration), so no host paths
enter the repository, and copies <device>/PUBLIC_NOTES.md if present. The internal NOTES.md and raw
result folders are never copied. Existing device folders in <data> are replaced.
"""

import argparse
import csv
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # also under python3 -I
from build_site import FIELDS, HERE  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="lab data folder with <device>/records.csv")
    ap.add_argument("--data", default=os.path.join(HERE, "data"), help="site data folder (default: %(default)s)")
    a = ap.parse_args(argv)
    for dev in sorted(os.listdir(a.source)):
        src = os.path.join(a.source, dev, "records.csv")
        if not os.path.isfile(src):
            continue
        dst = os.path.join(a.data, dev)
        os.makedirs(dst, exist_ok=True)
        with open(src, newline="", encoding="utf-8") as fh:
            rows = [{f: (r.get(f) or "").strip() for f in FIELDS} for r in csv.DictReader(fh)]
        for r in rows:
            r["result_dir"] = "/".join(r["result_dir"].split("/")[-3:])
        with open(os.path.join(dst, "records.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        notes = os.path.join(a.source, dev, "PUBLIC_NOTES.md")
        if os.path.isfile(notes):
            shutil.copyfile(notes, os.path.join(dst, "PUBLIC_NOTES.md"))
        print(f"{dev}: {len(rows)} records", file=sys.stderr)


if __name__ == "__main__":
    main()
