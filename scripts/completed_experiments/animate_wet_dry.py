#!/usr/bin/env python3
"""Animate wet vs dry transport: surface CO2, 750 hPa CO2, column-mean VMR."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.animation as animation
from netCDF4 import Dataset
from datetime import datetime
import os

# --- Load data ---
wet_file = "/tmp/era5_anim_wet.nc"
dry_file = "/tmp/era5_anim_dry.nc"

def load(fname):
    ds = Dataset(fname, "r")
    # Python netCDF4 reads as (time, lat, lon) — no transpose needed
    d = dict(
        sfc = np.array(ds["co2_surface"][:]) * 1e6,   # (time, lat, lon)
        col = np.array(ds["co2_column_mean"][:]) * 1e6,
        lev = np.array(ds["co2_750hPa"][:]) * 1e6,
        lon = ds["lon"][:],
        lat = ds["lat"][:],
    )
    ds.close()
    return d

wet = load(wet_file)
dry = load(dry_file)

nt = min(wet["sfc"].shape[0], dry["sfc"].shape[0])
lon, lat = wet["lon"], wet["lat"]
print(f"Loaded: wet={wet['sfc'].shape}, dry={dry['sfc'].shape}, nt={nt}")

# Use every-other timestep for smoother GIF (1-hour data → 2h frames)
step = 2
frames = list(range(0, nt, step))

# --- Compute anomalies from 411 ppm ---
ref = 411.0

# --- Figure setup ---
fig, axes = plt.subplots(3, 2, figsize=(16, 10),
                          subplot_kw={"projection": None})
fig.subplots_adjust(hspace=0.30, wspace=0.08, left=0.06, right=0.94,
                     top=0.92, bottom=0.05)

titles_row = ["Surface CO$_2$", "750 hPa CO$_2$", "Column-mean VMR"]
titles_col = ["Moist transport", "Dry transport"]

# Shared color limits — will update dynamically
ims = []
cbs = []

def init_frame():
    """Set up the initial empty frame."""
    global ims, cbs
    ims = []
    cbs = []
    for row in range(3):
        for col_idx in range(2):
            ax = axes[row, col_idx]
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            if row == 2:
                ax.set_xlabel("Longitude")
            if col_idx == 0:
                ax.set_ylabel("Latitude")
            ax.set_xticks([-180, -90, 0, 90, 180])
            ax.set_yticks([-90, -45, 0, 45, 90])
            # Placeholder image
            im = ax.imshow(np.zeros((len(lat), len(lon))),
                           extent=[-180, 180, -90, 90], origin="lower",
                           cmap="RdBu_r", aspect="auto")
            ims.append(im)
    return ims

def update(frame_idx):
    t = list(frames)[frame_idx]
    # Get data for this time
    data = {
        "wet": [wet["sfc"][t], wet["lev"][t], wet["col"][t]],
        "dry": [dry["sfc"][t], dry["lev"][t], dry["col"][t]],
    }

    # Compute global extremes for symmetric color scale
    for row in range(3):
        w = data["wet"][row] - ref
        d = data["dry"][row] - ref
        vmax = max(np.nanmax(np.abs(w)), np.nanmax(np.abs(d)), 0.1)
        # Cap at 15 ppm for readability
        vmax = min(vmax, 15.0)

        for col_idx, key in enumerate(["wet", "dry"]):
            idx = row * 2 + col_idx
            field = data[key][row] - ref
            ims[idx].set_data(field)
            ims[idx].set_clim(-vmax, vmax)

    # Compute stats
    stats = {}
    for key in ["wet", "dry"]:
        for i, name in enumerate(["sfc", "750", "col"]):
            d = data[key][i]
            stats[f"{key}_{name}"] = f"std={np.nanstd(d):.3f}"

    # Titles
    hours = t  # 1-hour intervals
    for row in range(3):
        for col_idx, key in enumerate(["wet", "dry"]):
            ax = axes[row, col_idx]
            name = ["sfc", "750", "col"][row]
            ax.set_title(f"{titles_col[col_idx]} — {titles_row[row]}\n"
                         f"{stats[f'{key}_{name}']} ppm",
                         fontsize=10)

    fig.suptitle(f"CO$_2$ anomaly from 411 ppm — Hour {hours:02d} / 48\n"
                 f"18L balanced, uniform IC, Poisson-balanced fluxes",
                 fontsize=13, fontweight="bold")
    return ims

# Initialize
init_frame()

# Create animation
print(f"Creating animation with {len(frames)} frames...")
ani = animation.FuncAnimation(fig, update, frames=len(frames),
                                blit=False, interval=200)

# Add colorbars (one per row)
for row in range(3):
    cb = fig.colorbar(ims[row * 2], ax=axes[row, :].tolist(),
                       shrink=0.8, pad=0.02, label="ppm anomaly")
    cbs.append(cb)

outdir = os.path.expanduser("~/www/catrina")
os.makedirs(outdir, exist_ok=True)
outpath = os.path.join(outdir, "wet_vs_dry_transport.gif")
ani.save(outpath, writer="pillow", fps=5, dpi=100)
print(f"Saved: {outpath}")

# Also save a static comparison at hour 24
fig2, axes2 = plt.subplots(3, 2, figsize=(16, 10))
fig2.subplots_adjust(hspace=0.35, wspace=0.15, left=0.06, right=0.94,
                      top=0.90, bottom=0.05)
t24 = min(24, nt - 1)
for row in range(3):
    for col_idx, (key, label) in enumerate([("wet", "Moist"), ("dry", "Dry")]):
        ax = axes2[row, col_idx]
        src = [wet, dry][col_idx]
        fields = [src["sfc"], src["lev"], src["col"]]
        field = fields[row][t24] - ref
        vmax = min(max(np.nanmax(np.abs(field)), 0.1), 15.0)
        im = ax.imshow(field, extent=[-180, 180, -90, 90], origin="lower",
                       cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
        fig2.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        std = np.nanstd(fields[row][t24])
        ax.set_title(f"{label} — {titles_row[row]} (std={std:.3f} ppm)", fontsize=10)
        if row == 2: ax.set_xlabel("Longitude")
        if col_idx == 0: ax.set_ylabel("Latitude")

fig2.suptitle(f"CO$_2$ at hour {t24} — Wet vs Dry transport (uniform 411 ppm IC)",
              fontsize=13, fontweight="bold")
static_path = os.path.join(outdir, "wet_vs_dry_hour24.png")
fig2.savefig(static_path, dpi=150)
print(f"Saved: {static_path}")
