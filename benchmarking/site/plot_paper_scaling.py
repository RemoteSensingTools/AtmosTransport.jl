"""Plot Section 5 from the dashboard records. Pass the output PDF as the argument."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

from build_site import Bench, FLOAT_TYPE, read_records


DEVICES = [
    ("L40S", "NVIDIA L40S", "#0072B2", "o"),
    ("A100", "NVIDIA A100", "#D55E00", "s"),
    ("M5Pro", "Apple M5 Pro", "#009E73", "^"),
    ("CPU-wurst", "CPU (32 threads)", "#484848", "D"),
]
GRIDS = [24, 48, 90, 180, 360, 720]
TRACERS = [1, 4, 12, 50, 100, 200]


def plot(output):
    rows, problems = read_records(Path(__file__).resolve().parent / "data")
    if problems:
        raise ValueError("\n".join(problems))
    bench = Bench([row for row in rows if row["float_type"] == FLOAT_TYPE])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.95), sharey=True)
    fig.subplots_adjust(left=.093, right=.983, top=.88, bottom=.31, wspace=.16)

    for ax, grid, title in zip(axes, [None, 90, 180],
                               ["(a) Four tracers", "(b) C90", "(c) C180"]):
        ticks = TRACERS if grid else GRIDS
        for device, label, color, marker in DEVICES:
            points = []
            for x in ticks:
                config_grid, count = (grid, x) if grid else (x, 4)
                seconds = bench.s(device, "main", f"C{config_grid}", count, "all")
                if seconds is None:
                    continue
                if config_grid == 720:  # Different meteorology
                    ax.plot(x, 365 * seconds, marker=marker, color=color,
                            mfc="white", ms=5, ls="none")
                else:
                    points.append((x, 365 * seconds))
            if points:
                xs, ys = zip(*points)
                ax.plot(xs, ys, label=label, color=color, marker=marker,
                        lw=1.15, ms=4, ls="none" if device == "CPU-wurst" else "-")
        if grid:
            single = bench.s("L40S", "main", f"C{grid}", 1, "all")
            ax.plot(TRACERS, [365 * single * n for n in TRACERS],
                    color="#595959", lw=1.15, ls="--", zorder=1,
                    label="Linear scaling")
        ax.set(xscale="log", yscale="log", title=title,
               xlabel="Tracer count" if grid else "Cubed-sphere resolution")
        ax.set_xticks(ticks, [str(x) if grid else f"C{x}" for x in ticks], fontsize=7)
        ax.set_xlim(ticks[0] / 1.28, ticks[-1] * 1.28)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.grid(axis="y", alpha=.22)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(34, 520000)
    axes[0].set_yticks([60, 600, 3600, 21600, 86400, 345600],
                       ["1 min", "10 min", "1 h", "6 h", "1 d", "4 d"])
    axes[0].set_ylabel("Time per simulated year")
    fig.legend(*axes[1].get_legend_handles_labels(), loc="lower center", ncol=5,
               frameon=False, handlelength=1.55, columnspacing=1.2)
    fig.savefig(output)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    plot(parser.parse_args().output)
