#!/usr/bin/env python3
"""Detailed visualization of noise growth from flat dry VMR IC on moist transport.

Creates:
1. GIF: 6-panel (sfc, 750, col) x (field, timestep Δ) with fixed color scale
2. GIF: Zonal-mean cross section (lat vs time) for sfc, 750, col
3. PNG: Zonal std vs time for each level
4. GIF: Zoomed early hours (0-12) with tight color scale to see where noise starts
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from netCDF4 import Dataset
import os

outdir = os.path.expanduser("~/www/catrina")
os.makedirs(outdir, exist_ok=True)

# --- Load data ---
ds = Dataset("/tmp/era5_anim_wet.nc")
sfc = np.array(ds["co2_surface"][:]) * 1e6     # (time, lat, lon)
lev = np.array(ds["co2_750hPa"][:]) * 1e6
col = np.array(ds["co2_column_mean"][:]) * 1e6
lon = np.array(ds["lon"][:])
lat = np.array(ds["lat"][:])
ds.close()

nt, nlat, nlon = sfc.shape
ref = 411.0
print(f"Shape: ({nt}, {nlat}, {nlon}), hours 0..{nt-1}")

# Anomalies
sfc_a = sfc - ref
lev_a = lev - ref
col_a = col - ref

# =====================================================================
# PLOT 1: Zonal std time series — where and when does noise grow?
# =====================================================================
fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
fig1.subplots_adjust(wspace=0.3, bottom=0.15)

hours = np.arange(nt)
for ax, data, title in zip(axes1, [sfc, lev, col],
                            ["Surface", "750 hPa", "Column mean"]):
    # Global std
    gl_std = [np.std(data[t]) for t in range(nt)]
    ax.plot(hours, gl_std, "k-", lw=2, label="Global")

    # Latitude bands
    bands = [
        (0, 20, "Polar S", "C0"),
        (20, 70, "Mid-lat S", "C1"),
        (70, 110, "Deep tropics", "C3"),
        (110, 180, "Subtropics S", "C2"),
        (180, 250, "Subtropics N", "C4"),
        (250, 290, "Deep tropics N", "C5"),
        (290, 341, "Mid-lat N", "C6"),
        (341, 361, "Polar N", "C7"),
    ]
    for j1, j2, name, color in bands:
        band_std = [np.std(data[t, j1:j2, :]) for t in range(nt)]
        ax.plot(hours, band_std, color=color, lw=1, alpha=0.7, label=name)

    ax.set_xlabel("Hour")
    ax.set_ylabel("Std (ppm)")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.set_ylim(0.005, 50)
    ax.grid(True, alpha=0.3)

axes1[0].legend(fontsize=6, ncol=2, loc="upper left")
fig1.suptitle("Noise growth by latitude band — Moist transport, flat dry VMR IC",
              fontsize=12, fontweight="bold")
fig1.savefig(os.path.join(outdir, "noise_growth_zonal_std.png"), dpi=150)
print("Saved noise_growth_zonal_std.png")

# =====================================================================
# PLOT 2: Hovmoller (zonal-mean anomaly vs lat and time)
# =====================================================================
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
fig2.subplots_adjust(wspace=0.3, bottom=0.15)

for ax, data_a, title in zip(axes2, [sfc_a, lev_a, col_a],
                               ["Surface", "750 hPa", "Column mean"]):
    # Zonal mean anomaly: (time, lat)
    zm = np.mean(data_a, axis=2)
    vmax = min(np.max(np.abs(zm)), 5.0)
    im = ax.imshow(zm.T, aspect="auto", origin="lower",
                   extent=[0, nt-1, -90, 90],
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{title} — zonal-mean anomaly")
    fig2.colorbar(im, ax=ax, shrink=0.8, label="ppm")

fig2.suptitle("Hovmoller: zonal-mean CO$_2$ anomaly — Moist, flat dry IC",
              fontsize=12, fontweight="bold")
fig2.savefig(os.path.join(outdir, "noise_hovmoller.png"), dpi=150)
print("Saved noise_hovmoller.png")

# =====================================================================
# PLOT 3: Timestep differences — where does noise APPEAR first?
# =====================================================================
fig3, axes3 = plt.subplots(3, 4, figsize=(18, 11))
fig3.subplots_adjust(hspace=0.35, wspace=0.15)

# Show Δ for hours 0→1, 1→2, 2→3, 5→6 at each level
dt_pairs = [(0, 1), (1, 2), (2, 3), (5, 6)]
levels = [("Surface", sfc), ("750 hPa", lev), ("Column mean", col)]

for row, (lname, data) in enumerate(levels):
    for ci, (t1, t2) in enumerate(dt_pairs):
        ax = axes3[row, ci]
        if t2 >= nt:
            ax.set_visible(False)
            continue
        delta = data[t2] - data[t1]
        vmax = max(np.max(np.abs(delta)), 0.001)
        vmax = min(vmax, 2.0)  # cap for readability
        im = ax.imshow(delta, extent=[-180, 180, -90, 90], origin="lower",
                       cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
        std_d = np.std(delta)
        ax.set_title(f"{lname} Δ(h{t1}→h{t2})\nstd={std_d:.4f} ppm", fontsize=9)
        fig3.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        if row == 2: ax.set_xlabel("Lon")
        if ci == 0: ax.set_ylabel("Lat")

fig3.suptitle("Hourly differences — where does noise first appear?",
              fontsize=13, fontweight="bold")
fig3.savefig(os.path.join(outdir, "noise_hourly_deltas.png"), dpi=150)
print("Saved noise_hourly_deltas.png")

# =====================================================================
# PLOT 4: Early-hour zoom animation (hours 0-12, tight color scale)
# =====================================================================
n_early = min(13, nt)
fig4, axes4 = plt.subplots(2, 3, figsize=(16, 8))
fig4.subplots_adjust(hspace=0.30, wspace=0.15, left=0.06, right=0.92,
                      top=0.90, bottom=0.08)

ims4 = []
for row in range(2):
    for ci in range(3):
        ax = axes4[row, ci]
        im = ax.imshow(np.zeros((nlat, nlon)),
                       extent=[-180, 180, -90, 90], origin="lower",
                       cmap="RdBu_r", aspect="auto")
        ims4.append(im)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_yticks([-90, -45, 0, 45, 90])

# Row 0: anomaly fields (sfc, 750, col)
# Row 1: Δ from previous hour (sfc, 750, col)
labels_top = ["Surface anomaly", "750 hPa anomaly", "Column-mean anomaly"]
labels_bot = ["Surface Δ", "750 hPa Δ", "Column-mean Δ"]

def update_early(t):
    data_list = [sfc_a[t], lev_a[t], col_a[t]]
    if t > 0:
        delta_list = [sfc[t]-sfc[t-1], lev[t]-lev[t-1], col[t]-col[t-1]]
    else:
        delta_list = [np.zeros_like(sfc[0])]*3

    for ci in range(3):
        # Top row: anomaly
        field = data_list[ci]
        vmax = max(np.max(np.abs(field)), 0.01)
        vmax = min(vmax, 3.0)
        ims4[ci].set_data(field)
        ims4[ci].set_clim(-vmax, vmax)
        std = np.std(data_list[ci] + ref)
        axes4[0, ci].set_title(f"{labels_top[ci]}\nstd={std:.4f} ppm", fontsize=10)

        # Bottom row: delta
        delta = delta_list[ci]
        dvmax = max(np.max(np.abs(delta)), 0.001)
        dvmax = min(dvmax, 1.0)
        ims4[3+ci].set_data(delta)
        ims4[3+ci].set_clim(-dvmax, dvmax)
        dstd = np.std(delta)
        axes4[1, ci].set_title(f"{labels_bot[ci]}\nΔ_std={dstd:.4f} ppm", fontsize=10)

    fig4.suptitle(f"Noise growth — Hour {t:02d} / {n_early-1}\n"
                  f"Moist transport, flat dry VMR IC (411 ppm)",
                  fontsize=13, fontweight="bold")
    return ims4

ani4 = animation.FuncAnimation(fig4, update_early, frames=n_early,
                                blit=False, interval=500)

# Add colorbars
for ci in range(3):
    fig4.colorbar(ims4[ci], ax=axes4[0, ci], shrink=0.7, pad=0.02, label="ppm")
    fig4.colorbar(ims4[3+ci], ax=axes4[1, ci], shrink=0.7, pad=0.02, label="ppm")

ani4.save(os.path.join(outdir, "noise_early_hours.gif"),
          writer="pillow", fps=2, dpi=100)
print("Saved noise_early_hours.gif")

# =====================================================================
# PLOT 5: Full 48-hour animation with consistent color scale
# =====================================================================
fig5, axes5 = plt.subplots(1, 3, figsize=(16, 4.5))
fig5.subplots_adjust(wspace=0.15, left=0.05, right=0.92, top=0.85, bottom=0.12)

ims5 = []
for ci in range(3):
    ax = axes5[ci]
    im = ax.imshow(np.zeros((nlat, nlon)),
                   extent=[-180, 180, -90, 90], origin="lower",
                   cmap="RdBu_r", aspect="auto")
    ims5.append(im)
    ax.set_xlabel("Longitude")
    if ci == 0: ax.set_ylabel("Latitude")

labels5 = ["Surface", "750 hPa", "Column mean"]

def update_full(t):
    data_list = [sfc_a[t], lev_a[t], col_a[t]]
    for ci in range(3):
        field = data_list[ci]
        # Use consistent max across all times for this level
        vmax = min(max(np.max(np.abs(field)), 0.01), 15.0)
        ims5[ci].set_data(field)
        ims5[ci].set_clim(-vmax, vmax)
        std = np.std(field + ref)
        axes5[ci].set_title(f"{labels5[ci]} (std={std:.3f} ppm)", fontsize=10)

    fig5.suptitle(f"CO$_2$ anomaly from 411 ppm — Hour {t:02d}/48  "
                  f"(moist transport, flat dry IC)",
                  fontsize=12, fontweight="bold")
    return ims5

for ci in range(3):
    fig5.colorbar(ims5[ci], ax=axes5[ci], shrink=0.8, pad=0.02, label="ppm")

ani5 = animation.FuncAnimation(fig5, update_full, frames=nt,
                                blit=False, interval=150)
ani5.save(os.path.join(outdir, "noise_growth_full.gif"),
          writer="pillow", fps=4, dpi=100)
print("Saved noise_growth_full.gif")

print("\nAll outputs saved to", outdir)
