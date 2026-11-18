"""Step 1: dataset exploration.

    python experiments/eda.py

Outputs (results/eda/):
    inventory.csv           one row per trip: files, sizes, counts, gaps, extent, connectivity...
    summary.json            partition-level totals and data-quality findings
    figures/*.png           distributions, trajectory plots, road-network plots
Only the train and student_test partitions are touched.
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from common import RESULTS, SPLITS, save_json
from pathsnap.geo import LocalProjection, project_point_on_segments
from pathsnap.io_utils import load_case, list_cases

OUT = os.path.join(RESULTS, "eda")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371000 * np.arcsin(np.sqrt(a))


def raw_checks(folder, cid):
    """Checks done on the raw text before numpy parsing."""
    info = {}
    for ext in ("nodes", "arcs", "track", "route"):
        p = os.path.join(folder, f"{cid}.{ext}")
        info[f"has_{ext}"] = os.path.exists(p)
        info[f"size_kb_{ext}"] = os.path.getsize(p) / 1024 if os.path.exists(p) else np.nan
        if os.path.exists(p):
            with open(p, "rb") as f:
                raw = f.read()
            info[f"crlf_{ext}"] = b"\r\n" in raw
            lines = [l for l in raw.decode().splitlines() if l.strip()]
            cols = {len(l.split()) for l in lines}
            info[f"cols_{ext}"] = "/".join(map(str, sorted(cols)))
    return info


def trip_stats(split, folder):
    c = load_case(folder)
    cid = c["id"]
    tr, nodes, arcs, route = c["track"], c["nodes"], c["arcs"], c["route"]
    row = {"split": split, "trip": cid, **raw_checks(folder, cid)}

    # GPS observations
    row["n_fixes"] = len(tr)
    row["nan_fixes"] = int(np.isnan(tr).any(axis=1).sum())
    row["bad_coord"] = int(((np.abs(tr[:, 0]) > 180) | (np.abs(tr[:, 1]) > 90)).sum())
    dt = np.diff(tr[:, 2])
    row["time_non_increasing"] = int((dt <= 0).sum())
    row["dt_eq_1s_share"] = float(np.mean(np.isclose(dt, 1.0)))
    row["dt_max"] = float(dt.max())
    row["gaps_over_10s"] = int((dt > 10).sum())
    row["duration_min"] = float((tr[-1, 2] - tr[0, 2]) / 60)
    step = haversine(tr[:-1, 0], tr[:-1, 1], tr[1:, 0], tr[1:, 1])
    row["track_km"] = float(step.sum() / 1000)
    row["speed_median_kmh"] = float(np.median(step / np.maximum(dt, 1e-6)) * 3.6)
    row["jumps_over_60ms"] = int((step / np.maximum(dt, 1e-6) > 60).sum())
    row["stationary_share"] = float(np.mean(step < 0.5))
    row["lon_min"], row["lon_max"] = float(tr[:, 0].min()), float(tr[:, 0].max())
    row["lat_min"], row["lat_max"] = float(tr[:, 1].min()), float(tr[:, 1].max())
    row["lon_center"], row["lat_center"] = float(tr[:, 0].mean()), float(tr[:, 1].mean())
    row["extent_km_x"] = float(haversine(row["lon_min"], row["lat_center"], row["lon_max"], row["lat_center"]) / 1000)
    row["extent_km_y"] = float(haversine(row["lon_center"], row["lat_min"], row["lon_center"], row["lat_max"]) / 1000)

    # road network
    n = len(nodes)
    row["n_nodes"], row["n_arcs"] = n, len(arcs)
    row["arcs_bad_index"] = int(((arcs < 0) | (arcs >= n)).any(axis=1).sum())
    row["self_loops"] = int((arcs[:, 0] == arcs[:, 1]).sum())
    pairs = set(map(tuple, arcs))
    row["two_way_share"] = float(np.mean([(v, u) in pairs for u, v in arcs]))
    row["dup_arcs"] = len(arcs) - len(pairs)
    deg = np.bincount(arcs.ravel(), minlength=n)
    row["isolated_nodes"] = int((deg == 0).sum())
    g = coo_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(n, n)).tocsr()
    nw, lw = connected_components(g, directed=True, connection="weak")
    ns, ls = connected_components(g, directed=True, connection="strong")
    row["weak_components"] = int(nw)
    row["largest_weak_share"] = float(np.bincount(lw).max() / n)
    row["strong_components"] = int(ns)
    row["largest_strong_share"] = float(np.bincount(ls).max() / n)

    proj = LocalProjection(row["lon_center"], row["lat_center"])
    P, T = proj(nodes), proj(tr[:, :2])
    alen = np.linalg.norm(P[arcs[:, 1]] - P[arcs[:, 0]], axis=1)
    row["arc_len_median_m"] = float(np.median(alen))
    row["network_km"] = float(alen.sum() / 1000)

    # reference route
    if route is not None:
        row["route_arcs"] = len(route)
        row["route_bad_index"] = int(((route < 0) | (route >= len(arcs))).sum())
        cont = arcs[route[:-1], 1] == arcs[route[1:], 0]
        row["route_breaks"] = int((~cont).sum())
        row["route_repeat_arcs"] = int(len(route) - len(set(route.tolist())))
        row["route_uturns"] = int(sum(1 for a, b in zip(route[:-1], route[1:])
                                      if arcs[a, 0] == arcs[b, 1] and arcs[a, 1] == arcs[b, 0]))
        row["route_km"] = float(alen[route].sum() / 1000)
        A, B = P[arcs[route, 0]], P[arcs[route, 1]]
        sub = T[:: max(1, len(T) // 1500)]
        dev = np.array([project_point_on_segments(p[None], A, B)[0].min() for p in sub])
        row["gps_dev_p50_m"] = float(np.median(dev))
        row["gps_dev_p90_m"] = float(np.percentile(dev, 90))
        row["gps_dev_max_m"] = float(dev.max())
        row["share_fixes_over_50m"] = float(np.mean(dev > 50))
        d0 = project_point_on_segments(T[:1], A[:1], B[:1])[0][0]
        d1 = project_point_on_segments(T[-1:], A[-1:], B[-1:])[0][0]
        row["start_offset_m"], row["end_offset_m"] = float(d0), float(d1)
        row["_dev"] = dev
    return row


def plots(df):
    # distributions
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for a, col, title in zip(ax.ravel(),
                             ["n_fixes", "track_km", "dt_max", "speed_median_kmh", "n_arcs", "gps_dev_p90_m"],
                             ["GPS fixes per trip", "Track length (km)", "Largest time gap (s)",
                              "Median speed (km/h)", "Road-network arcs per trip", "90th pct GPS-to-route distance (m)"]):
        for split, g in df.groupby("split"):
            a.hist(g[col], bins=25, alpha=0.6, label=split)
        a.set_title(title)
        a.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "distributions.png"), dpi=120)
    plt.close(fig)

    # GPS deviation from reference route (all fixes pooled)
    fig, ax = plt.subplots(figsize=(8, 4))
    for split, g in df.groupby("split"):
        dev = np.concatenate(g["_dev"].to_list())
        ax.hist(np.minimum(dev, 200), bins=100, alpha=0.6, density=True, label=f"{split} (median {np.median(dev):.1f} m)")
    ax.set_xlabel("distance from GPS fix to reference route (m, capped at 200)")
    ax.set_ylabel("density")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("GPS noise relative to the reference route")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "gps_deviation.png"), dpi=120)
    plt.close(fig)

    # where the trips are
    fig, ax = plt.subplots(figsize=(10, 5))
    for split, g in df.groupby("split"):
        ax.scatter(g.lon_center, g.lat_center, s=25, label=split)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Trip locations (centre of each GPS track)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "trip_locations.png"), dpi=120)
    plt.close(fig)


def trajectory_grid(split, ids):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, cid in zip(axes.ravel(), ids):
        c = load_case(os.path.join(SPLITS[split], cid))
        proj = LocalProjection(*c["track"][:, :2].mean(0))
        P, T = proj(c["nodes"]), proj(c["track"][:, :2])
        lo, hi = T.min(0) - 200, T.max(0) + 200
        arcs = c["arcs"]
        inside = ((P[arcs[:, 0]] >= lo) & (P[arcs[:, 0]] <= hi)).all(1)
        seg = arcs[inside]
        xs = np.stack([P[seg[:, 0], 0], P[seg[:, 1], 0], np.full(len(seg), np.nan)], 1).ravel()
        ys = np.stack([P[seg[:, 0], 1], P[seg[:, 1], 1], np.full(len(seg), np.nan)], 1).ravel()
        ax.plot(xs, ys, color="#cccccc", lw=0.5, label="road network")
        r = c["route"]
        xs = np.stack([P[arcs[r, 0], 0], P[arcs[r, 1], 0], np.full(len(r), np.nan)], 1).ravel()
        ys = np.stack([P[arcs[r, 0], 1], P[arcs[r, 1], 1], np.full(len(r), np.nan)], 1).ravel()
        ax.plot(xs, ys, color="#2a9d8f", lw=3, alpha=0.6, label="reference route")
        ax.plot(T[:, 0], T[:, 1], ".", ms=1.5, color="#e76f51", label="GPS fixes")
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
        ax.set_title(f"{split} {cid}")
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"trajectories_{split}.png"), dpi=110)
    plt.close(fig)


def main():
    rows = []
    for split, folder in SPLITS.items():
        for f in list_cases(folder):
            rows.append(trip_stats(split, f))
            print(split, rows[-1]["trip"], flush=True)
    df = pd.DataFrame(rows)
    df.drop(columns=["_dev"]).to_csv(os.path.join(OUT, "inventory.csv"), index=False)
    plots(df)
    for split in SPLITS:
        g = df[df.split == split]
        ids = list(g.sort_values("gps_dev_p90_m")["trip"].iloc[[0, len(g) // 4, len(g) // 2, 3 * len(g) // 4, -2, -1]])
        trajectory_grid(split, ids)

    summary = {}
    for split, g in df.groupby("split"):
        dev = np.concatenate(g["_dev"].to_list())
        summary[split] = {
            "trips": len(g), "gps_fixes": int(g.n_fixes.sum()),
            "fixes_per_trip_min_median_max": [int(g.n_fixes.min()), int(g.n_fixes.median()), int(g.n_fixes.max())],
            "nan_fixes": int(g.nan_fixes.sum()), "bad_coords": int(g.bad_coord.sum()),
            "non_increasing_timestamps": int(g.time_non_increasing.sum()),
            "share_steps_exactly_1s": float(np.average(g.dt_eq_1s_share, weights=g.n_fixes)),
            "trips_with_gap_over_60s": int((g.dt_max > 60).sum()),
            "largest_gap_s": float(g.dt_max.max()),
            "road_arcs_total": int(g.n_arcs.sum()), "road_nodes_total": int(g.n_nodes.sum()),
            "isolated_nodes_total": int(g.isolated_nodes.sum()),
            "trips_with_multiple_weak_components": int((g.weak_components > 1).sum()),
            "median_largest_strong_component_share": float(g.largest_strong_share.median()),
            "route_breaks_total": int(g.route_breaks.sum()),
            "route_uturns_total": int(g.route_uturns.sum()),
            "gps_dev_median_m": float(np.median(dev)), "gps_dev_p90_m": float(np.percentile(dev, 90)),
            "gps_dev_p99_m": float(np.percentile(dev, 99)),
            "trips_with_median_dev_over_50m": g.loc[g.gps_dev_p50_m > 50, "trip"].tolist(),
            "trips_end_offset_over_200m": g.loc[g.end_offset_m > 200, "trip"].tolist(),
            "crlf_track_files": int(g.crlf_track.sum()),
        }
    save_json(summary, os.path.join(OUT, "summary.json"))
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()
