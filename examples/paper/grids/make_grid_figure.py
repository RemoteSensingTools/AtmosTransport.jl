"""Figure: the three transport grids (latitude-longitude, reduced Gaussian, cubed sphere) at a
comparable resolution, coloured by cell area (km2, Earth radius 6371 km).

Inputs: cs_c<nc>_corners.csv (cell corners of the model's GEOS-convention cubed sphere, written by
export_cs.jl) and n<nc>_pl.txt (ECMWF's ring sizes of the reduced-Gaussian grid, from the eccodes
sample reduced_gg_pl_<nc>; the octahedral rule nlon = 4k + 16 is used when the table is absent).
Ring boundaries lie midway between the Gauss-Legendre latitudes, as in the model.
Usage: python3 make_grid_figure.py <outdir> [nc]     (nc: cubed-sphere resolution, e.g. 48 or 12)
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
import cartopy.io.shapereader as shpreader

HERE = Path(__file__).parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE

# Reference palette: one sequential hue (blue, light -> dark), neutral inks, light surface.
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
         "#2a78d6", "#256abf"]                   # sequential blue, steps 100-500 (fills stay
MESH = "#0b0b0b"                                 # lighter than the mesh lines, step 700)
EARTH_RADIUS_KM = 6371.0
CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUES)

LON0, LAT0 = -40.0, 45.0          # view centre: North Atlantic, with the pole in view
EDGE_POINTS = 1                   # points per cell edge (set in main: more for coarse grids)


def xyz(lon, lat):
    lon, lat = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def view(p):
    """Rotate unit vectors so the view centre is +x; return (depth, right, up)."""
    l0, f0 = np.radians(LON0), np.radians(LAT0)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    x1 = x * np.cos(l0) + y * np.sin(l0)
    y1 = -x * np.sin(l0) + y * np.cos(l0)
    depth = x1 * np.cos(f0) + z * np.sin(f0)
    up = -x1 * np.sin(f0) + z * np.cos(f0)
    return depth, y1, up


def project_polygon(p):
    """Orthographic outline of a cell given as unit vectors; None if entirely on the far side.
    Vertices behind the limb are moved onto it, so cells at the edge are cut by the disc."""
    d, r, u = view(p)
    if np.all(d <= 0):
        return None
    hidden = d < 0
    if hidden.any():
        n = np.hypot(r, u)
        r = np.where(hidden, r / n, r)
        u = np.where(hidden, u / n, u)
    return np.column_stack([r, u])


def lonlat_cell(lon_w, lon_e, lat_s, lat_n):
    """Outline of a cell bounded by meridians and parallels (edges sampled in lon/lat)."""
    n = EDGE_POINTS
    t = np.linspace(0, 1, n, endpoint=False)
    lon = np.concatenate([lon_w + t * (lon_e - lon_w), np.full(n, lon_e),
                          lon_e + t * (lon_w - lon_e), np.full(n, lon_w)])
    lat = np.concatenate([np.full(n, lat_s), lat_s + t * (lat_n - lat_s),
                          np.full(n, lat_n), lat_n + t * (lat_s - lat_n)])
    return xyz(lon, lat)


def great_circle_cell(c):
    """Outline of a cell with great-circle edges between its four corners (unit vectors)."""
    pts = []
    for a, b in zip(c, np.roll(c, -1, axis=0)):
        for t in np.linspace(0, 1, EDGE_POINTS, endpoint=False):
            v = (1 - t) * a + t * b
            pts.append(v / np.linalg.norm(v))
    return np.array(pts)


def band_area(lon_w, lon_e, lat_s, lat_n):
    return np.radians(lon_e - lon_w) * (np.sin(np.radians(lat_n)) - np.sin(np.radians(lat_s)))


def quad_area(c):
    """Area of a spherical quadrilateral (unit sphere) from its corners, as two triangles."""
    def tri(a, b, cc):
        num = abs(np.dot(a, np.cross(b, cc)))
        den = 1 + np.dot(a, b) + np.dot(b, cc) + np.dot(cc, a)
        return 2 * np.arctan2(num, den)
    return tri(c[0], c[1], c[2]) + tri(c[0], c[2], c[3])


def latlon_grid(dlon=7.5, dlat=7.5):
    cells, areas = [], []
    for lat_s in np.arange(-90, 90, dlat):
        for lon_w in np.arange(0, 360, dlon):
            cells.append(lonlat_cell(lon_w, lon_w + dlon, lat_s, lat_s + dlat))
            areas.append(band_area(lon_w, lon_w + dlon, lat_s, lat_s + dlat))
    return cells, np.array(areas), []


def reduced_gaussian_grid(N=12, table=None):
    """Reduced-Gaussian cells: ring sizes from an ECMWF table (the layout of ERA5's N320 grid;
    `n<N>_pl.txt`, read from the eccodes sample `reduced_gg_pl_<N>`), else the octahedral rule."""
    nodes, _ = np.polynomial.legendre.leggauss(2 * N)
    lat = np.degrees(np.arcsin(nodes))                       # south -> north
    faces = np.concatenate([[-90.0], 0.5 * (lat[:-1] + lat[1:]), [90.0]])
    if table is not None:
        nlon = [int(n) for n in np.loadtxt(table)]
        assert len(nlon) == 2 * N and nlon == nlon[::-1]
    else:
        hemi = [4 * k + 16 for k in range(1, N + 1)]
        nlon = hemi + hemi[::-1]
    cells, areas = [], []
    for j, n in enumerate(nlon):
        dlon = 360.0 / n
        for i in range(n):
            cells.append(lonlat_cell(i * dlon, (i + 1) * dlon, faces[j], faces[j + 1]))
            areas.append(band_area(i * dlon, (i + 1) * dlon, faces[j], faces[j + 1]))
    return cells, np.array(areas), []


def cubed_sphere_grid(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    nc = int(round(np.sqrt(len(data) / 6))) - 1
    corners = xyz(data[:, 3], data[:, 4]).reshape(6, nc + 1, nc + 1, 3)    # [panel, j, i]
    cells, areas, edges = [], [], []
    for p in range(6):
        for j in range(nc):
            for i in range(nc):
                c = np.array([corners[p, j, i], corners[p, j, i + 1],
                              corners[p, j + 1, i + 1], corners[p, j + 1, i]])
                cells.append(great_circle_cell(c))
                areas.append(quad_area(c))
        # the four panel edges, as sampled great-circle polylines
        for line in (corners[p, 0, :], corners[p, nc, :], corners[p, :, 0], corners[p, :, nc]):
            pts = []
            for a, b in zip(line[:-1], line[1:]):
                for t in np.linspace(0, 1, 4, endpoint=False):
                    v = (1 - t) * a + t * b
                    pts.append(v / np.linalg.norm(v))
            pts.append(line[-1])
            edges.append(np.array(pts))
    return cells, np.array(areas), edges


def visible_segments(p):
    """Split a polyline of unit vectors into the pieces on the near side."""
    d, r, u = view(p)
    out, cur = [], []
    for k in range(len(d)):
        if d[k] > 0:
            cur.append((r[k], u[k]))
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return [np.array(s) for s in out if len(s) > 1]


def coastlines():
    segs = []
    reader = shpreader.Reader(shpreader.natural_earth("110m", "physical", "coastline"))
    for geom in reader.geometries():
        for line in getattr(geom, "geoms", [geom]):
            ll = np.asarray(line.coords)
            segs += visible_segments(xyz(ll[:, 0], ll[:, 1]))
    return segs


def thousands(x):
    """Three significant figures with thin-space thousands separators."""
    v = float(f"{x:.3g}")
    return f"{v:,.0f}".replace(",", "\u2009")


def main():
    global EDGE_POINTS
    nc = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    EDGE_POINTS = 6 if nc <= 16 else 1
    dll = 360.0 / (4 * nc)                       # the cubed sphere's spacing at the equator
    table = HERE / f"n{nc}_pl.txt"               # ECMWF ring sizes, if available for this N
    grids = [
        ("Latitude–longitude", f"{dll:g}° × {dll:g}°", latlon_grid(dll, dll)),
        ("Reduced Gaussian", f"N{nc}" if table.exists() else f"octahedral, N = {nc}",
         reduced_gaussian_grid(nc, table if table.exists() else None)),
        ("Cubed sphere", f"C{nc}", cubed_sphere_grid(HERE / f"cs_c{nc}_corners.csv")),
    ]
    lw = 0.3 if nc <= 16 else 0.09               # cell outlines (dark, so they differ from every fill)
    km2 = [areas * EARTH_RADIUS_KM**2 for _, _, (_, areas, _) in grids]
    for (name, _, (cells, areas, _)), a in zip(grids, km2):
        print(f"{name:20s} cells {len(cells):6d}  area sum/4pi {areas.sum() / (4 * np.pi):.6f}  "
              f"cell area {a.min():8.0f} .. {a.max():8.0f} km2 (mean {a.mean():8.0f})  max/min {a.max() / a.min():.2f}")
    vmax = 5e3 * np.ceil(max(a.max() for a in km2) / 5e3)
    norm = Normalize(vmin=0.0, vmax=vmax)        # linear, shared: the cubed sphere's range stays visible
    a_ref = 4 * np.pi * EARTH_RADIUS_KM**2 / (6 * nc * nc)      # mean cubed-sphere cell area
    coast = coastlines()

    plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.1, 3.4), facecolor=SURFACE)
    axes = [fig.add_axes([0.012 + 0.33 * k, 0.19, 0.316, 0.66]) for k in range(3)]
    for ax, (name, detail, (cells, areas, edges)), r in zip(axes, grids, km2):
        polys, vals = [], []
        for cell, value in zip(cells, r):
            poly = project_polygon(cell)
            if poly is not None:
                polys.append(poly)
                vals.append(value)
        vals = np.array(vals)
        # Outline width follows the cell size, so the outlines of very small cells (the
        # latitude-longitude poles) do not cover their fill.
        widths = lw * np.clip(vals / a_ref, 0.03, 1.0)
        pc = PolyCollection(polys, array=vals, cmap=CMAP, norm=norm,
                            edgecolors=MESH, linewidths=widths)
        ax.add_collection(pc)
        ax.add_collection(LineCollection(coast, colors=INK, linewidths=0.7))
        if edges:
            segs = [s for e in edges for s in visible_segments(e)]
            ax.add_collection(LineCollection(segs, colors=INK, linewidths=1.4, capstyle="round"))
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, ec=INK2, lw=0.5))
        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(-1.02, 1.02)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(0.5, 1.15, name, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=9, color=INK, fontweight="bold")
        ax.text(0.5, 1.08, f"{detail} · " + f"{len(cells):,}".replace(",", "\u2009") + " cells",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5, color=INK2)
        ax.text(0.5, 1.012, f"cell area {thousands(r.min())}–{thousands(r.max())} km²",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5, color=INK2)
    cax = fig.add_axes([0.30, 0.11, 0.40, 0.026])
    ticks = np.arange(0, vmax + 1, 10e3)
    cb = fig.colorbar(pc, cax=cax, orientation="horizontal", ticks=ticks)
    cb.ax.set_xticklabels([f"{t / 1e3:g}" for t in ticks])
    cb.ax.tick_params(length=2, width=0.4, colors=INK2, labelsize=7)
    cb.ax.minorticks_off()
    cb.outline.set_linewidth(0.4)
    cb.outline.set_edgecolor(INK2)
    cb.set_label("Cell area (10³ km²)", fontsize=8, color=INK)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"grids_c{nc}.{ext}", dpi=300, facecolor=SURFACE)
    print("wrote", OUT / f"grids_c{nc}.pdf")


if __name__ == "__main__":
    main()
