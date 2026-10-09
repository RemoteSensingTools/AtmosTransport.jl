#!/usr/bin/env python3
"""
Compare a GEOS-IT C180 transport binary against raw GEOS native wind/flux files.

The transport binary stores dry mass amounts per advection substep, while the
raw files expose accumulated CTM mass fluxes and direct winds in different
units. This script therefore reports pattern agreement, best-fit scale factors,
and normalized residuals rather than assuming exact unit equality.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import netCDF4 as nc
import numpy as np


DEFAULT_ROOT = Path("~/data/AtmosTransport").expanduser()
DEFAULT_DATE = "20211202"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare GEOS-IT C180 raw U/V/OMEGA/MFXC/MFYC against a transport binary."
    )
    parser.add_argument("--date", default=DEFAULT_DATE, help="YYYYMMDD date tag.")
    parser.add_argument(
        "--binary",
        default=str(
            DEFAULT_ROOT
            / "met/geosit/C180/transport_binary_v4_merge025hpa_adaptive_f32"
            / f"geos_transport_{DEFAULT_DATE}_float32.bin"
        ),
        help="Path to geos_transport_YYYYMMDD_float*.bin.",
    )
    parser.add_argument(
        "--raw-dir",
        default=None,
        help="Directory containing GEOSIT.YYYYMMDD.CTM_A1.C180.nc and A3dyn.C180.nc.",
    )
    parser.add_argument("--window", type=int, default=1, help="1-based binary/CTM_A1 window.")
    parser.add_argument(
        "--omega-index",
        type=int,
        default=None,
        help="0-based A3dyn index. Default maps window times to nearest 3-hour A3dyn sample.",
    )
    parser.add_argument(
        "--levels",
        default="all",
        help="Binary level selection, e.g. all, 35, 20:50, 20:50:2. 0-based, top-to-bottom.",
    )
    parser.add_argument("--panel", type=int, default=0, help="1..6 panel, or 0 for all panels.")
    parser.add_argument("--lat-min", type=float, default=None, help="Optional minimum latitude mask.")
    parser.add_argument("--lat-max", type=float, default=None, help="Optional maximum latitude mask.")
    parser.add_argument(
        "--raw-level-offset",
        default="auto",
        help="Raw top-level offset. auto aligns the bottom binary levels to raw levels.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Randomly sample this many points per comparison after masking. 0 uses all points.",
    )
    return parser.parse_args()


def find_raw_dir(date: str, raw_dir: Optional[str]) -> Path:
    if raw_dir:
        path = Path(raw_dir).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path
    candidates = [
        DEFAULT_ROOT / "met/geosit/C180/daily/raw" / date,
        DEFAULT_ROOT / "met/geosit/C180/raw_catrine" / date,
    ]
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError("No GEOS-IT C180 raw directory found for " + date)


def read_binary_header(path: Path) -> Dict:
    with path.open("rb") as handle:
        raw = handle.read(262144)
    try:
        end = raw.index(b"\0")
    except ValueError:
        end = len(raw)
    return json.loads(raw[:end].decode("utf-8"))


def section_elements(header: Dict, section: str) -> int:
    ncells = int(header["Nc"])
    npanel = int(header.get("npanel", 6))
    nz = int(header["nlevel"])
    if section in ("m", "dm", "dtrain", "entu", "detu", "entd", "detd"):
        return npanel * ncells * ncells * nz
    if section in ("am", "dam"):
        return npanel * (ncells + 1) * ncells * nz
    if section in ("bm", "dbm"):
        return npanel * ncells * (ncells + 1) * nz
    if section in ("cm", "dcm", "cmfmc"):
        return npanel * ncells * ncells * (nz + 1)
    if section in ("ps", "pblh", "ustar", "pbl_hflux", "hflux", "t2m"):
        return npanel * ncells * ncells
    if section in ("vdiff_u", "vdiff_v", "vdiff_t", "vdiff_qv", "dkg", "qv", "qv_start", "qv_end"):
        return npanel * ncells * ncells * nz
    raise ValueError(f"Unsupported payload section {section!r}")


def section_offsets(header: Dict) -> Dict[str, int]:
    offsets: Dict[str, int] = {}
    offset = 0
    for section in header["payload_sections"]:
        offsets[section.lower()] = offset
        offset += section_elements(header, section.lower())
    if offset != int(header["elems_per_window"]):
        raise ValueError(f"Computed elems_per_window={offset}, header has {header['elems_per_window']}")
    return offsets


def load_binary_section(path: Path, header: Dict, offsets: Dict[str, int], window: int, section: str) -> np.ndarray:
    ncells = int(header["Nc"])
    npanel = int(header.get("npanel", 6))
    nz = int(header["nlevel"])
    dtype = np.float32 if int(header.get("float_bytes", 4)) == 4 else np.float64
    header_bytes = int(header["header_bytes"])
    elems_per_window = int(header["elems_per_window"])
    data = np.memmap(path, dtype=dtype, mode="r", offset=header_bytes)
    base = (window - 1) * elems_per_window + offsets[section]

    if section in ("m", "dm", "dtrain", "entu", "detu", "entd", "detd", "vdiff_u", "vdiff_v", "vdiff_t", "vdiff_qv", "dkg"):
        shape = (ncells, ncells, nz)
    elif section in ("am", "dam"):
        shape = (ncells + 1, ncells, nz)
    elif section in ("bm", "dbm"):
        shape = (ncells, ncells + 1, nz)
    elif section in ("cm", "dcm", "cmfmc"):
        shape = (ncells, ncells, nz + 1)
    elif section in ("ps", "pblh", "ustar", "pbl_hflux", "hflux", "t2m"):
        shape = (ncells, ncells)
    else:
        raise ValueError(section)

    panel_elems = math.prod(shape)
    out = np.empty((npanel,) + shape, dtype=dtype)
    for panel in range(npanel):
        start = base + panel * panel_elems
        stop = start + panel_elems
        out[panel] = np.asarray(data[start:stop]).reshape(shape, order="F")
    return out


def parse_level_indices(text: str, nz: int) -> np.ndarray:
    if text == "all":
        return np.arange(nz)
    if ":" in text:
        parts = [int(p) if p else None for p in text.split(":")]
        if len(parts) > 3:
            raise ValueError("--levels accepts start:stop[:step]")
        parts += [None] * (3 - len(parts))
        return np.arange(nz)[slice(*parts)]
    return np.array([int(text)], dtype=int)


def raw_levels_for_binary(raw_nz: int, bin_nz: int, bin_levels: np.ndarray, raw_level_offset: str) -> np.ndarray:
    if raw_level_offset == "auto":
        offset = max(raw_nz - bin_nz, 0)
    else:
        offset = int(raw_level_offset)
    raw = offset + bin_levels
    if raw.min(initial=0) < 0 or raw.max(initial=0) >= raw_nz:
        raise ValueError(f"Raw level selection out of range: offset={offset}, raw_nz={raw_nz}, levels={raw}")
    return raw


def raw_panel_xyz(var, time_index: int, raw_levels: Sequence[int]) -> np.ndarray:
    # NetCDF shape is (time, lev, nf, Ydim, Xdim). Return (nf, Xdim, Ydim, lev).
    arr = np.asarray(var[time_index, raw_levels, :, :, :], dtype=np.float64)
    return np.transpose(arr, (1, 3, 2, 0))


def raw_lat_mask(dataset: nc.Dataset, ncells: int, npanel: int) -> Optional[np.ndarray]:
    for name in ("lats", "lat", "latitude"):
        if name not in dataset.variables:
            continue
        arr = np.asarray(dataset.variables[name][:], dtype=np.float64)
        if arr.shape == (npanel, ncells, ncells):
            return np.transpose(arr, (0, 2, 1))
        if arr.shape == (ncells, ncells, npanel):
            return np.transpose(arr, (2, 1, 0))
    return None


def omega_index_for_window(window: int, n_omega: int) -> int:
    # CTM_A1 samples hourly from 00:30. A3dyn samples every 3 hours from 01:30.
    return max(0, min(n_omega - 1, round((window - 2) / 3)))


def select_panel(arr: np.ndarray, panel: int) -> np.ndarray:
    if panel == 0:
        return arr
    if panel < 1 or panel > arr.shape[0]:
        raise ValueError("--panel must be 0 or 1..6")
    return arr[panel - 1 : panel]


def expand_spatial_mask(mask3: Optional[np.ndarray], shape: Tuple[int, ...]) -> Optional[np.ndarray]:
    if mask3 is None:
        return None
    if len(shape) == 4:
        return np.broadcast_to(mask3[:, :, :, None], shape)
    return np.broadcast_to(mask3, shape)


def finite_vectors(binary: np.ndarray, raw: np.ndarray, mask: Optional[np.ndarray], sample: int) -> Tuple[np.ndarray, np.ndarray]:
    b = np.asarray(binary, dtype=np.float64)
    r = np.asarray(raw, dtype=np.float64)
    good = np.isfinite(b) & np.isfinite(r)
    if mask is not None:
        good &= mask
    bvec = b[good].ravel()
    rvec = r[good].ravel()
    if sample and bvec.size > sample:
        rng = np.random.default_rng(20260603)
        idx = rng.choice(bvec.size, size=sample, replace=False)
        bvec = bvec[idx]
        rvec = rvec[idx]
    return bvec, rvec


def stats(binary: np.ndarray, raw: np.ndarray, mask: Optional[np.ndarray], sample: int) -> Dict[str, float]:
    b, r = finite_vectors(binary, raw, mask, sample)
    if b.size < 2:
        return {"n": float(b.size)}
    r_mean = float(np.mean(r))
    b_mean = float(np.mean(b))
    r_std = float(np.std(r))
    b_std = float(np.std(b))
    centered_r = r - r_mean
    centered_b = b - b_mean
    denom = float(np.dot(centered_r, centered_r))
    slope = float(np.dot(centered_r, centered_b) / denom) if denom > 0 else float("nan")
    intercept = b_mean - slope * r_mean if math.isfinite(slope) else float("nan")
    fit = slope * r + intercept if math.isfinite(slope) else np.full_like(b, np.nan)
    rmse = float(np.sqrt(np.mean((b - fit) ** 2))) if math.isfinite(slope) else float("nan")
    nrmse = rmse / b_std if b_std > 0 else float("nan")
    corr = float(np.corrcoef(b, r)[0, 1]) if b_std > 0 and r_std > 0 else float("nan")
    return {
        "n": float(b.size),
        "corr": corr,
        "slope_bin_per_raw": slope,
        "intercept": intercept,
        "nrmse_after_fit": nrmse,
        "binary_mean": b_mean,
        "binary_std": b_std,
        "binary_min": float(np.min(b)),
        "binary_max": float(np.max(b)),
        "raw_mean": r_mean,
        "raw_std": r_std,
        "raw_min": float(np.min(r)),
        "raw_max": float(np.max(r)),
    }


def print_stats(title: str, result: Dict[str, float]) -> None:
    if result.get("n", 0.0) < 2:
        print(f"{title:24s} n={int(result.get('n', 0))} insufficient")
        return
    print(
        f"{title:24s} "
        f"n={int(result['n']):9d} "
        f"corr={result['corr']: .5f} "
        f"slope={result['slope_bin_per_raw']: .6e} "
        f"fit_nrmse={result['nrmse_after_fit']: .5f} "
        f"bin_std={result['binary_std']: .6e} "
        f"raw_std={result['raw_std']: .6e}"
    )


def main() -> None:
    args = parse_args()
    binary = Path(args.binary.format(date=args.date)).expanduser()
    raw_dir = find_raw_dir(args.date, args.raw_dir)
    ctm_path = raw_dir / f"GEOSIT.{args.date}.CTM_A1.C180.nc"
    dyn_path = raw_dir / f"GEOSIT.{args.date}.A3dyn.C180.nc"
    for path in (binary, ctm_path, dyn_path):
        if not path.exists():
            raise FileNotFoundError(path)

    header = read_binary_header(binary)
    offsets = section_offsets(header)
    ncells = int(header["Nc"])
    npanel = int(header.get("npanel", 6))
    bin_nz = int(header["nlevel"])
    bin_levels = parse_level_indices(args.levels, bin_nz)
    if bin_levels.size == 0:
        raise ValueError("--levels selected no binary levels")
    if args.window < 1 or args.window > int(header["nwindow"]):
        raise ValueError("--window out of binary range")

    with nc.Dataset(ctm_path) as ctm, nc.Dataset(dyn_path) as dyn:
        raw_nz = len(ctm.dimensions["lev"])
        raw_levels = raw_levels_for_binary(raw_nz, bin_nz, bin_levels, args.raw_level_offset)
        omega_index = args.omega_index
        if omega_index is None:
            omega_index = omega_index_for_window(args.window, len(dyn.dimensions["time"]))

        print("Paths")
        print(f"  binary: {binary}")
        print(f"  CTM_A1: {ctm_path}")
        print(f"  A3dyn:  {dyn_path}")
        print()
        print("Selection")
        print(
            f"  window={args.window}  omega_index={omega_index}  "
            f"panel={'all' if args.panel == 0 else args.panel}  "
            f"binary_levels={bin_levels[0]}..{bin_levels[-1]} ({bin_levels.size})  "
            f"raw_levels={raw_levels[0]}..{raw_levels[-1]}"
        )
        print(
            f"  binary C{ncells} L{bin_nz}, float{8 * int(header.get('float_bytes', 4))}, "
            f"mass_basis={header.get('mass_basis')}, sections={','.join(header['payload_sections'])}"
        )
        print(
            f"  balance={header.get('geos_horizontal_balance')}  "
            f"vertical_flux={header.get('geos_vertical_flux')}  "
            f"steps_this_window={header.get('steps_per_window_by_window', [header.get('steps_per_window')])[args.window - 1]}"
        )
        print()

        am = load_binary_section(binary, header, offsets, args.window, "am")[:, :, :, bin_levels]
        bm = load_binary_section(binary, header, offsets, args.window, "bm")[:, :, :, bin_levels]
        cm = load_binary_section(binary, header, offsets, args.window, "cm")[:, :, :, :]
        cm = cm[:, :, :, bin_levels] + cm[:, :, :, bin_levels + 1]
        cm *= 0.5

        # GEOS native convention stores MFXC/MFYC as the east/north face of
        # cell (i,j,k). The v4 binary mapping is am[i+1,j,k] and bm[i,j+1,k].
        bin_x_face = am[:, 1:, :, :]
        bin_y_face = bm[:, :, 1:, :]
        bin_x_center = 0.5 * (am[:, :-1, :, :] + am[:, 1:, :, :])
        bin_y_center = 0.5 * (bm[:, :, :-1, :] + bm[:, :, 1:, :])
        bin_z = cm

        raw_mfxc = raw_panel_xyz(ctm.variables["MFXC"], args.window - 1, raw_levels)
        raw_mfyc = raw_panel_xyz(ctm.variables["MFYC"], args.window - 1, raw_levels)
        raw_u = raw_panel_xyz(dyn.variables["U"], omega_index, raw_levels)
        raw_v = raw_panel_xyz(dyn.variables["V"], omega_index, raw_levels)
        raw_omega = raw_panel_xyz(dyn.variables["OMEGA"], omega_index, raw_levels)

        spatial_mask = np.ones((npanel, ncells, ncells), dtype=bool)
        lats = raw_lat_mask(ctm, ncells, npanel)
        if lats is not None:
            if args.lat_min is not None:
                spatial_mask &= lats >= args.lat_min
            if args.lat_max is not None:
                spatial_mask &= lats <= args.lat_max
        elif args.lat_min is not None or args.lat_max is not None:
            print("Warning: requested latitude mask, but no native lats variable was found.")
        if args.panel:
            keep = np.zeros_like(spatial_mask)
            keep[args.panel - 1] = spatial_mask[args.panel - 1]
            spatial_mask = keep
        mask4 = expand_spatial_mask(spatial_mask, bin_x_face.shape)

        print("Horizontal mass-flux pattern comparisons: binary canonical faces vs raw GEOS faces")
        print_stats("bin_x_face vs MFXC", stats(bin_x_face, raw_mfxc, mask4, args.sample))
        print_stats("bin_x_face vs MFYC", stats(bin_x_face, raw_mfyc, mask4, args.sample))
        print_stats("bin_y_face vs MFXC", stats(bin_y_face, raw_mfxc, mask4, args.sample))
        print_stats("bin_y_face vs MFYC", stats(bin_y_face, raw_mfyc, mask4, args.sample))
        print()
        print("Direct wind-speed pattern comparisons: binary cell-centered flux amount vs raw velocity")
        print_stats("bin_x_ctr vs raw U", stats(bin_x_center, raw_u, mask4, args.sample))
        print_stats("bin_x_ctr vs raw V", stats(bin_x_center, raw_v, mask4, args.sample))
        print_stats("bin_y_ctr vs raw U", stats(bin_y_center, raw_u, mask4, args.sample))
        print_stats("bin_y_ctr vs raw V", stats(bin_y_center, raw_v, mask4, args.sample))
        print()
        print("Vertical comparison: binary interface cm centered to layers vs raw OMEGA")
        print_stats("bin_z vs raw OMEGA", stats(bin_z, raw_omega, mask4, args.sample))


if __name__ == "__main__":
    main()
