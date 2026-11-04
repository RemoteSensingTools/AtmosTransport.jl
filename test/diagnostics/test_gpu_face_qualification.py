"""Run with python3; exercises the real summary and binary reader without CUDA."""
import csv
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/validation"))
from compare_gpu_face_windows import read_fluxes, compare


def write_csv(path, rows, delimiter=","):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        for t in (255, 639):
            for grid in ("ll", "rg", "cs"):
                name = f"device_T{t}_{grid}"
                fields, levels = [], []
                for case, count in (("harmonics", 3), ("era5_20211201_00", 137)):
                    for field in ("ps", "u", "v", "C"):
                        fields.append(dict(case=case, region=1, field=field, max_absolute=0,
                            peak_scaled=0, max_relative_resolved_faces=0, outside_1e6_mixed_bound=0,
                            relative_l2=0, max_level_rms_scaled=0, max_level_relative_l2=0,
                            repeat_bitwise="true", band_bitwise="true", band_peak_scaled=0,
                            fresh_process_bitwise="true"))
                    for field in ("ps", "u", "v", "C", "horizontal"):
                        for level in range(1, (1 if field == "ps" else count) + 1):
                            levels.append(dict(case=case, region=1, field=field, level=level,
                                reference_rms=1, max_absolute=0, rms_scaled=0, relative_l2=0))
                write_csv(self.root / f"{name}.csv", fields)
                write_csv(self.root / f"{name}_levels.csv", levels)
                for rep in (1, 2):
                    self.memory(self.root, f"{name}_run{rep}_")
        for grid in ("ll", "cs"):
            for backend in ("cpu", "gpu"):
                folder = self.root / f"window_{grid}_{backend}"
                folder.mkdir()
                write_csv(folder / "timings.tsv", [dict(phase=f"measured_{rep}", seconds=1) for rep in (1, 2)], "\t")
                for rep in (1, 2):
                    target = folder / f"measured_{rep}"
                    target.mkdir()
                    self.binary(target / "flux.bin", grid)
                if backend == "gpu":
                    self.memory(folder, "", pool="pool_peak.toml", memory="device_memory.csv")

    @staticmethod
    def memory(folder, prefix, pool="pool.toml", memory="memory.csv"):
        (folder / (prefix + pool)).write_text("pool_used_high_bytes=0x10\npool_reserved_high_bytes=32\n")
        (folder / (prefix + memory)).write_text("timestamp,memory\n0,32\n")

    @staticmethod
    def binary(path, grid):
        header = dict(nwindow=1, spectral_truncation_used=639, float_bytes=8, header_bytes=4096,
            nlevel=2, steps_per_window_by_window=[8], merge_map=[1, 2], mass_basis="dry",
            grid_type="cubed_sphere" if grid == "cs" else "latlon", Nc=1, npanel=6,
            payload_sections=["am", "bm", "cm"], n_am=2, n_bm=2, n_cm=3)
        data = np.arange(1, 67 if grid == "cs" else 8, dtype="<f8")
        with path.open("wb") as stream:
            stream.write(json.dumps(header).encode().ljust(4096, b"\0"))
            data.tofile(stream)

    def summary(self, expected):
        result = subprocess.run([sys.executable, str(REPO / "scripts/validation/summarize_gpu_face_quadrature.py"), str(self.root)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        self.assertIn("Qualification: PASS." if expected == 0 else "Qualification: FAIL", result.stdout)

    def mutate_csv(self, name, edit):
        path = self.root / name
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        edit(rows)
        write_csv(path, rows)

    def test_finite_complete_control(self):
        self.summary(0)

    def test_nonfinite_field_metrics(self):
        path = self.root / "device_T255_ll.csv"
        original = path.read_bytes()
        for column in ("max_absolute", "peak_scaled", "relative_l2", "max_level_rms_scaled", "band_peak_scaled"):
            for value in ("nan", "inf", "-inf"):
                with self.subTest(column=column, value=value):
                    path.write_bytes(original)
                    self.mutate_csv(path.name, lambda rows: rows[1].update({column: value}))
                    self.summary(1)

    def test_native_coverage(self):
        path = self.root / "device_T639_rg_levels.csv"
        original = path.read_bytes()
        for field in ("ps", "u", "v", "C", "horizontal"):
            for case in ("harmonics", "era5_20211201_00"):
                with self.subTest(field=field, case=case):
                    path.write_bytes(original)
                    self.mutate_csv(path.name, lambda rows: rows.remove(next(r for r in rows if r["field"] == field and r["case"] == case)))
                    self.summary(1)
        path.write_bytes(original)
        self.mutate_csv(path.name, lambda rows: rows.__setitem__(1, rows[0]))
        self.summary(1)  # duplicate cannot replace a missing level

    def test_nonfinite_levels_and_reference_norms(self):
        path = self.root / "device_T255_cs_levels.csv"
        original = path.read_bytes()
        for column in ("reference_rms", "max_absolute", "rms_scaled", "relative_l2"):
            for value in ("nan", "inf", "-inf"):
                with self.subTest(column=column, value=value):
                    path.write_bytes(original)
                    self.mutate_csv(path.name, lambda rows: rows[0].update({column: value}))
                    self.summary(1)
        path.write_bytes(original)
        self.mutate_csv(path.name, lambda rows: next(r for r in rows if r["field"] == "horizontal").update(reference_rms="0"))
        self.summary(1)

    def test_nonfinite_payloads(self):
        path = self.root / "window_ll_gpu/measured_1/flux.bin"
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                with path.open("r+b") as stream:
                    stream.seek(4096 + 4 * 8)  # first cm element
                    np.array([value], dtype="<f8").tofile(stream)
                with self.assertRaises(ValueError):
                    read_fluxes(path)
                self.summary(1)

    def test_truncated_payload(self):
        path = self.root / "window_cs_gpu/measured_2/flux.bin"
        os.truncate(path, 4096 + 8)
        with self.assertRaises(ValueError):
            read_fluxes(path)
        self.summary(1)

    def test_zero_reference_policy(self):
        for path in self.root.glob("window_*/*/*.bin"):
            data = np.memmap(path, dtype="<f8", mode="r+", offset=4096)
            data[:] = 0
            data.flush()
        self.summary(0)
        path = self.root / "window_ll_gpu/measured_1/flux.bin"
        data = np.memmap(path, dtype="<f8", mode="r+", offset=4096)
        data[0] = 1
        data.flush()
        with self.assertRaises(ValueError):
            compare(self.root)
        self.summary(1)

    def test_nonfinite_timing(self):
        write_csv(self.root / "window_ll_gpu/timings.tsv", [dict(phase="measured_1",seconds="nan"),dict(phase="measured_2",seconds=1)], "\t")
        self.summary(1)


if __name__ == "__main__":
    unittest.main()
