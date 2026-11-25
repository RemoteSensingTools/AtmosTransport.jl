"""Compare latitude–pressure sections for the July 2022 Gaussian surface release after 3 days."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm

from plot_zonal import load, section


def plot(zonal, output):
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
    fig, axes = plt.subplots(3, 3, figsize=(8.2, 5.6), sharex=True, sharey=True,
                             layout="constrained")
    absolute = [0, .125, .25, .5, 1, 2, 4, 8, 16, 32]
    signed = [-16, -8, -4, -2, -1, -.5, -.25, .25, .5, 1, 2, 4, 8, 16]
    panels = [("AP", "Advection + mixing"), ("C", "Convection alone"), ("APC", "All processes")]
    for row, (arm, label) in enumerate(panels):
        fields = []
        for driver in ("era5", "merra2"):
            run = load(zonal, driver, "c90", "202207", arm, "32")
            fields.append(section(run, "sfc", 72))
        for col, field in enumerate([*fields, fields[0] - fields[1]]):
            ax = axes[row, col]
            cmap = plt.get_cmap("RdBu_r" if col == 2 else "Blues").copy()
            cmap.set_bad("#d4d4d4")
            bounds = signed if col == 2 else absolute
            mesh = ax.pcolormesh(run["lat_edges"], run["p_edges"], field, cmap=cmap,
                                 norm=BoundaryNorm(bounds, cmap.N), rasterized=True)
            ax.set(ylim=(1000, 100), xlim=(-90, 90), yticks=[1000, 700, 400, 100],
                   xticks=[-60, 0, 60], xticklabels=["60°S", "0°", "60°N"])
            if row == 0:
                ax.set_title(["ERA5", "MERRA-2", "ERA5 − MERRA-2"][col])
            if col == 0:
                ax.set_ylabel(f"{label}\nDry pressure (hPa)")
            if row == 2 and col in (1, 2):
                parents = axes[:, :2].ravel().tolist() if col == 1 else axes[:, 2].tolist()
                bar = fig.colorbar(mesh, ax=parents, orientation="horizontal",
                                   extend="max" if col == 1 else "both", fraction=.055,
                                   ticks=[0, .25, 1, 4, 16, 32] if col == 1 else [-16, -2, 0, 2, 16])
                bar.set_label("Mole fraction / well-mixed value" if col == 1 else "Difference (same units)")
    fig.suptitle("Surface release after three days · July 2022")
    fig.savefig(output)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zonal", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.zonal, args.output)
