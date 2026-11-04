"""Summarize L40S qualification without concealing precision failures."""
import csv
import math
import pathlib
import re
import sys
from compare_gpu_face_windows import compare

root = pathlib.Path(sys.argv[1])
lines = ["# GPU face quadrature validation", "", "Accuracy uses CPU Float64 references; absolute units are Pa (PS) or kg/s.",
         "The per-face bound is 1e-6 relative plus 2e-14 times max(field peak, 1) for symmetry zeros.",
         "Per-level face errors use the joint u/v RMS at that level, including symmetry-zero directions.",
         "Targets: maximum face error / level RMS <=1e-5; balanced T639 cm relative L2 <=1e-4.", ""]
failed = False
expected = [root / f"device_T{t}_{g}.csv" for t in (255, 639) for g in ("ll", "rg", "cs")]
CASES = {"harmonics": 3, "era5_20211201_00": 137}
FIELD_METRICS = ("max_absolute", "peak_scaled", "max_relative_resolved_faces",
                 "relative_l2", "max_level_rms_scaled", "max_level_relative_l2", "band_peak_scaled")
LEVEL_METRICS = ("reference_rms", "max_absolute", "rms_scaled", "relative_l2")


def finite_metrics(rows, names):
    for row in rows:
        for name in names:
            value = float(row[name])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {name}: {row}")


def accuracy_rows(path):
    rows = list(csv.DictReader(path.open()))
    fields = {(case, "1", field) for case in CASES for field in ("ps", "u", "v", "C")}
    keys = [(r["case"], r["region"], r["field"]) for r in rows]
    if len(keys) != len(fields) or set(keys) != fields:
        raise ValueError("expected exactly both cases, region 1 and ps/u/v/C")
    finite_metrics(rows, FIELD_METRICS)
    for row in rows:
        if int(row["outside_1e6_mixed_bound"]) < 0:
            raise ValueError("negative bound violation count")
        for flag in ("repeat_bitwise", "band_bitwise", "fresh_process_bitwise"):
            if row[flag] not in ("true", "false"):
                raise ValueError(f"invalid {flag}")
    level_path = path.with_name(path.stem + "_levels.csv")
    levels = list(csv.DictReader(level_path.open()))
    coverage = {(case, "1", field, level) for case, count in CASES.items()
                for field in ("ps", "u", "v", "C", "horizontal")
                for level in range(1, (1 if field == "ps" else count) + 1)}
    keys = [(r["case"], r["region"], r["field"], int(r["level"])) for r in levels]
    if len(keys) != len(coverage) or set(keys) != coverage:
        raise ValueError("incomplete or duplicate case/field/native-level coverage (expected 562 rows)")
    finite_metrics(levels, LEVEL_METRICS)
    for row in levels:
        # Exact zero references have zero error and zero scaled metrics. A
        # nonzero residual has no defined relative accuracy and must fail.
        if float(row["reference_rms"]) == 0 and any(float(row[n]) != 0 for n in LEVEL_METRICS[1:]):
            raise ValueError(f"nonzero error against a zero reference: {row}")
        if row["field"] == "horizontal" and float(row["reference_rms"]) <= 0:
            raise ValueError("joint horizontal reference RMS must be positive")
    return rows, levels


for path in expected:
    try:
        rows, level_rows = accuracy_rows(path)
    except (OSError, ValueError, KeyError) as error:
        lines.append(f"Invalid accuracy report {path.name}: {error}")
        failed = True
        continue
    for case in CASES:
        fields = [r for r in rows if r["case"] == case and r["field"] in ("u", "v")]
        violations = sum(int(r["outside_1e6_mixed_bound"]) for r in fields)
        case_rows = [r for r in rows if r["case"] == case]
        repeat = all(r["repeat_bitwise"] == "true" and r["fresh_process_bitwise"] == "true" for r in case_rows)
        band = all(r["band_bitwise"] == "true" for r in case_rows)
        peak = max(float(r["peak_scaled"]) for r in fields)
        relative = max(float(r["max_relative_resolved_faces"]) for r in fields)
        band_error = max(float(r["band_peak_scaled"]) for r in fields)
        rms_scaled = max(float(r["max_level_rms_scaled"]) for r in fields)
        joint_rows = [r for r in level_rows if r["case"] == case and r["field"] == "horizontal"]
        level_l2 = max(float(r["relative_l2"]) for r in joint_rows)
        # Check the native rows themselves, in addition to the field maxima.
        failed |= any(float(r["rms_scaled"]) > 1e-5 for r in joint_rows)
        lines.append(f"- {path.stem}, {case}: face peak-scaled error {peak:.6g}; maximum per-face relative {relative:.6g}; "
                     f"max error/level RMS {rms_scaled:.6g}; max joint-face level relative L2 {level_l2:.6g}; {violations} faces outside historical bound; same-plan/fresh-process bitwise {repeat}; band bitwise {band}, sensitivity {band_error:.6g}.")
        failed |= rms_scaled > 1e-5 or not repeat
lines += ["", "Warmed complete production windows include setup/IO, both endpoints, humidity, balance, closure, replay and write."]
for grid in ("ll", "cs"):
    times = {}
    for backend in ("cpu", "gpu"):
        path = root / f"window_{grid}_{backend}" / "timings.tsv"
        if not path.exists():
            failed = True
            continue
        try:
            rows = [r for r in csv.DictReader(path.open(), delimiter="\t") if r["phase"].startswith("measured")]
            seconds = [float(r["seconds"]) for r in rows]
            if sorted(r["phase"] for r in rows) != ["measured_1", "measured_2"] or not all(math.isfinite(t) and t > 0 for t in seconds):
                raise ValueError("expected two positive finite measured times")
            times[backend] = sum(seconds) / 2
        except (ValueError, KeyError) as error:
            failed = True
            lines.append(f"Invalid timings {grid}/{backend}: {error}")
    if len(times) == 2:
        lines.append(f"- {grid}: CPU16 {times['cpu']:.3f} s, GPU Float32 {times['gpu']:.3f} s; speedup {times['cpu']/times['gpu']:.3f}x.")
    else:
        lines.append(f"- {grid}: missing complete timings.")
try:
    balanced = compare(root)
    lines += ["", "Balanced dry production payloads vs CPU Float64 (T639, same window):"]
    for row in balanced:
        lines.append(f"- {row['grid']} {row['repetition']} {row['field']}: relative L2 {row['relative_l2']:.6g}, max error/field RMS {row['rms_scaled']:.6g}.")
        failed |= row["field"] == "cm" and row["relative_l2"] > 1e-4
except (OSError, RuntimeError, ValueError, KeyError, AssertionError) as error:
    failed = True
    lines.append(f"Balanced comparison incomplete: {error}")
lines += ["", "Pool high-water bytes are in *_pool.toml and window_*/pool_peak.toml. These are exact for the active CUDA pool.",
          "Whole-device peaks below are sampled every 100 ms, include reservations/context/vendor allocations, and may miss short transients."]
pool_paths = [root / f"device_T{t}_{g}_run{r}_pool.toml" for t in (255, 639) for g in ("ll", "rg", "cs") for r in (1, 2)]
pool_paths += [root / f"window_{g}_gpu" / "pool_peak.toml" for g in ("ll", "cs")]
for path in pool_paths:
    if not path.exists():
        lines.append(f"Missing pool high-water report: {path.relative_to(root)}")
        failed = True
        continue
    values = dict(re.findall(r"(pool_\w+_bytes)\s*=\s*(0x[0-9a-fA-F]+|\d+)", path.read_text()))
    if set(values) != {"pool_used_high_bytes", "pool_reserved_high_bytes"}:
        lines.append(f"Incomplete pool report: {path.relative_to(root)}")
        failed = True
    else:
        used, reserved = int(values["pool_used_high_bytes"], 0), int(values["pool_reserved_high_bytes"], 0)
        lines.append(f"- {path.relative_to(root)}: exact pool used high {used} bytes, reserved high {reserved} bytes.")
memory_paths = [root / f"device_T{t}_{g}_run{r}_memory.csv" for t in (255, 639) for g in ("ll", "rg", "cs") for r in (1, 2)]
memory_paths += [root / f"window_{g}_gpu" / "device_memory.csv" for g in ("ll", "cs")]
for path in memory_paths:
    if not path.exists():
        lines.append(f"Missing device samples: {path.relative_to(root)}")
        failed = True
        continue
    used = []
    for row in csv.reader(path.open()):
        try:
            value = float(row[1])
            if not math.isfinite(value) or value < 0:
                failed = True
            else:
                used.append(value)
        except (IndexError, ValueError):
            pass
    if used:
        lines.append(f"- {path.relative_to(root)}: sampled peak {max(used):.0f} MiB (first sample {used[0]:.0f} MiB).")
    else:
        failed = True
lines += ["", "Qualification: " + ("FAIL / residuals require review." if failed else "PASS."), ""]
text = "\n".join(lines)
(root / "RESULTS.md").write_text(text)
print(text)
sys.exit(1 if failed else 0)
