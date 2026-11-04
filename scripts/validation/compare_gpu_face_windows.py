"""Compare balanced dry binary payloads for the same Float64/Float32 window."""
import csv
import json
import pathlib
import sys
import numpy as np


def read_fluxes(path):
    with path.open("rb") as stream:
        header = json.loads(stream.read(16 * 1024 * 1024).split(b"\0", 1)[0])
    assert header["nwindow"] == 1, "expected exactly one production window"
    assert header["spectral_truncation_used"] == 639, "T639 qualification required"
    if header["float_bytes"] not in (4, 8):
        raise ValueError(f"unsupported binary float width: {path}")
    offset = header["header_bytes"] + header.get("n_geometry_elems", 0) * header["float_bytes"]
    data = np.memmap(path, mode="r", dtype="<f8" if header["float_bytes"] == 8 else "<f4", offset=offset)
    result = {}
    start = 0
    cs_shapes = dict(m=(header.get("Nc",0), header.get("Nc",0), header["nlevel"]),
                     am=(header.get("Nc",0)+1, header.get("Nc",0), header["nlevel"]),
                     bm=(header.get("Nc",0), header.get("Nc",0)+1, header["nlevel"]),
                     cm=(header.get("Nc",0), header.get("Nc",0), header["nlevel"]+1),
                     ps=(header.get("Nc",0), header.get("Nc",0)),
                     dm=(header.get("Nc",0), header.get("Nc",0), header["nlevel"]))
    for section in header["payload_sections"]:
        count = (int(np.prod(cs_shapes[section])) * header["npanel"] if header["grid_type"] == "cubed_sphere"
                 else header["n_" + section])
        if section in ("am", "bm", "cm"):
            values = np.asarray(data[start:start + count], dtype=np.float64)
            if count <= 0 or values.size != count or not np.isfinite(values).all():
                raise ValueError(f"incomplete or nonfinite {section} payload: {path}")
            result[section] = values
        start += count
        if section == "cm":
            break
    return header, result


def compare(root):
    rows = []
    for grid in ("ll", "cs"):
        for rep in ("measured_1", "measured_2"):
            files = [list((root / f"window_{grid}_{backend}" / rep).glob("*.bin")) for backend in ("cpu", "gpu")]
            if any(len(f) != 1 for f in files):
                raise RuntimeError(f"missing completed {grid} {rep} binaries")
            rh, reference = read_fluxes(files[0][0])
            ah, actual = read_fluxes(files[1][0])
            for key in ("nlevel", "nwindow", "steps_per_window_by_window", "merge_map", "mass_basis"):
                assert rh[key] == ah[key], f"different window contract: {key}"
            for field in ("am", "bm", "cm"):
                r, a = reference[field], actual[field]
                if a.shape != r.shape:
                    raise ValueError(f"different {grid} {rep} {field} payload shapes")
                error = a - r
                reference_norm = np.linalg.norm(r)
                error_norm = np.linalg.norm(error)
                maximum = np.max(np.abs(error))
                if not np.isfinite([reference_norm, error_norm, maximum]).all():
                    raise ValueError(f"nonfinite {grid} {rep} {field} metrics")
                # A zero reference passes only for exactly zero actual data.
                if reference_norm == 0:
                    if np.any(r != 0) or np.any(a != 0):
                        raise ValueError(f"nonzero {field} error against zero reference")
                    relative_l2 = rms_scaled = 0.0
                else:
                    relative_l2 = error_norm / reference_norm
                    rms_scaled = maximum / (reference_norm / np.sqrt(r.size))
                if not np.isfinite([relative_l2, rms_scaled]).all():
                    raise ValueError(f"nonfinite scaled {field} metrics")
                rows.append(dict(grid=grid, repetition=rep, field=field, relative_l2=relative_l2,
                                 max_absolute=maximum, rms_scaled=rms_scaled))
    with (root / "balanced_windows.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    for row in compare(pathlib.Path(sys.argv[1])):
        print(row)
