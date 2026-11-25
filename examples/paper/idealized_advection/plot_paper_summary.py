"""Regenerate the two Section 4 figures from the model's idealized-advection directory.

Input data are the archived case results (paper data revision 0b13756fd).
The original deformation PDF stores each map's cell colors as a lossless
bitmap, separately from labels. Reuse those fields without resampling.
"""
import argparse
import csv
import subprocess
import tempfile
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from PIL import Image

SOURCE = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
SCHEMES = [
    ("upwind", "Upwind", "#303030", "o"),
    ("slopes", "Slopes", "#0072B2", "^"),
    ("ppm", "PPM", "#D55E00", "s"),
    ("ppm_cw84", "PPM (CW84)", "#009E73", "v"),
    ("linrood5", "Lin–Rood 5", "#AA4477", "D"),
    ("linrood7", "Lin–Rood 7", "#6655AA", "x"),
]


def convergence(source, output):
    with (source / "results/norms.csv").open() as f:
        data = [r for r in csv.DictReader(f) if not r["tag"] and r["family"] == "cs"
                and r["precision"] == "Float64" and r["time"] == "T"
                and r["tracer"] in ("gaussian_hill", "gaussian_hills")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.85), sharey=True)
    fig.subplots_adjust(left=.10, right=.98, bottom=.28, top=.86, wspace=.15)
    for ax, (flow, title) in zip(axes, [("A", "Solid-body rotation"),
                                        ("B", "Nondivergent deformation"),
                                        ("C", "Divergent deformation")]):
        for scheme, name, color, marker in SCHEMES:
            points = sorted((int(r["grid"][1:]), float(r["l2"])) for r in data
                            if r["flow"] == flow and r["scheme"] == scheme)
            if [n for n, value in points] != [48, 96, 192]:
                raise ValueError(f"Missing resolutions for {flow} {scheme}")
            ax.plot([n for n, value in points], [value for n, value in points],
                    label=name, marker=marker, markersize=4, lw=1.2, color=color)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([48, 96, 192], ["C48", "C96", "C192"])
        ax.set_title(title, fontsize=8.7)
        ax.grid(axis="y", alpha=.22)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(r"Normalized $\ell_2$ error")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=6,
               frameon=False, columnspacing=1.1, handlelength=1.5)
    fig.savefig(output / "verification_convergence.pdf")
    plt.close(fig)


def deformation(source, output):
    # In the published panel order the rows are exact, C96 upwind, N96
    # upwind, C96 PPM, lat-lon PPM, and C96 Lin–Rood. Each row starts
    # with the day 6 and day 12 slotted-cylinder fields.
    with tempfile.TemporaryDirectory() as temp:
        prefix = Path(temp) / "panel"
        original = str(source / "figures/deformation_maps.pdf")
        listing = subprocess.check_output(["pdfimages", "-list", original], text=True)
        types = dict((int(number), kind) for number, kind in
                     re.findall(r"^\s*1\s+(\d+)\s+(image|smask)\s+", listing, re.M))
        maps = [number for number, kind in types.items() if kind == "image"]
        if len(maps) != 25:  # Six rows of four maps, followed by the colorbar.
            raise ValueError("Unexpected layout of the model's deformation figure")
        subprocess.run(["pdfimages", "-png", original,
                        str(prefix)], check=True)
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.0))
        fig.subplots_adjust(left=.11, right=.985, bottom=.205, top=.885, wspace=.065, hspace=.08)
        for col, (name, source_row) in enumerate([("Exact solution", 0), ("Upwind", 1),
                                                   ("Lin–Rood 7", 5)]):
            axes[0, col].set_title(name, fontsize=9)
            for row in range(2):
                index = maps[4 * source_row + row]
                field = np.asarray(Image.open(f"{prefix}-{index:03d}.png").convert("RGB"))
                if types.get(index + 1) != "smask":
                    raise ValueError("Expected a transparency mask for this cubed-sphere map")
                alpha = np.asarray(Image.open(f"{prefix}-{index+1:03d}.png").convert("L"))
                if field.shape[:2] != alpha.shape:
                    raise ValueError("Unexpected map or mask dimensions")
                ax = axes[row, col]
                ax.imshow(np.dstack([field, alpha]), extent=(0, 360, -90, 90),
                          interpolation="none", aspect="equal")
                ax.set(xticks=[], yticks=[])
                for spine in ax.spines.values():
                    spine.set_color("#888888")
                    spine.set_linewidth(.5)
        for ax, day in zip(axes[:, 0], ("Day 6", "Day 12")):
            ax.set_ylabel(day, rotation=0, ha="right", va="center")
        colors = ["#f3f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                  "#2a78d6", "#256abf", "#1d4f8f", "#143766"]
        cax = fig.add_axes([.31, .135, .43, .024])
        cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1.05),
                          cmap=LinearSegmentedColormap.from_list("blue", colors)),
                          cax=cax, orientation="horizontal", extend="both")
        cb.set_ticks([0, .25, .5, .75, 1])
        cb.set_label("Tracer mole fraction", labelpad=1)
        fig.savefig(output / "verification_deformation.pdf")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SOURCE / "figures")
    settings = parser.parse_args()
    settings.output.mkdir(parents=True, exist_ok=True)
    convergence(SOURCE, settings.output)
    deformation(SOURCE, settings.output)
