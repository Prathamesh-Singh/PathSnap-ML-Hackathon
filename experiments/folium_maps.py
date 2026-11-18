"""Interactive Folium maps: raw GPS, nearby road network, reference route,
predicted route (classifier-only and classifier+HMM), each as its own layer.

    python experiments/folium_maps.py --split student_test --trips 00000052 00000056
    python experiments/folium_maps.py --split train --trips 00000003 --eda   (EDA map, no prediction)

Maps are written to results/maps/<split>_<trip>.html
"""
import argparse
import os

import folium
import numpy as np

from common import RESULTS, SPLITS
from pathsnap.io_utils import load_case

OUT = os.path.join(RESULTS, "maps")
os.makedirs(OUT, exist_ok=True)


def arcs_to_lines(nodes, arcs, ids):
    """Merge consecutive connected arcs into polylines of (lat, lon)."""
    lines, cur = [], []
    for a in ids:
        u, v = arcs[a]
        if cur and cur[-1] == tuple(nodes[u][::-1]):
            cur.append(tuple(nodes[v][::-1]))
        else:
            if len(cur) > 1:
                lines.append(cur)
            cur = [tuple(nodes[u][::-1]), tuple(nodes[v][::-1])]
    if len(cur) > 1:
        lines.append(cur)
    return lines


def add_legend(m, items):
    html = "".join(f'<div><span style="display:inline-block;width:18px;height:4px;background:{c};'
                   f'margin-right:6px;vertical-align:middle"></span>{t}</div>' for t, c in items)
    m.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;'
        f'padding:8px 10px;border:1px solid #999;border-radius:4px;font:13px sans-serif">{html}</div>'))


def make_map(case, predictions=None, network_pad_deg=0.003, title=""):
    tr, nodes, arcs = case["track"], case["nodes"], case["arcs"]
    lat, lon = tr[:, 1], tr[:, 0]
    m = folium.Map(location=[lat.mean(), lon.mean()], tiles="cartodbpositron", control_scale=True)
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])

    # road network near the track
    lo = tr[:, :2].min(0) - network_pad_deg
    hi = tr[:, :2].max(0) + network_pad_deg
    inside = np.where(((nodes[arcs[:, 0]] >= lo) & (nodes[arcs[:, 0]] <= hi)).all(1))[0]
    net = folium.FeatureGroup(name=f"Road network ({len(inside)} arcs)", show=True)
    for a in inside:
        u, v = arcs[a]
        folium.PolyLine([nodes[u][::-1], nodes[v][::-1]], color="#9a9a9a", weight=1, opacity=0.6).add_to(net)
    net.add_to(m)

    items = [("Road network", "#9a9a9a")]
    if case.get("route") is not None:
        ref = folium.FeatureGroup(name="Reference route", show=True)
        for line in arcs_to_lines(nodes, arcs, case["route"]):
            folium.PolyLine(line, color="#2a9d8f", weight=8, opacity=0.5).add_to(ref)
        ref.add_to(m)
        items.append(("Reference route", "#2a9d8f"))

    colors = ["#e76f51", "#6a4c93", "#1d3557"]
    for (name, route), col in zip((predictions or {}).items(), colors):
        g = folium.FeatureGroup(name=name, show=True)
        for line in arcs_to_lines(nodes, arcs, route):
            folium.PolyLine(line, color=col, weight=3, opacity=0.9).add_to(g)
        g.add_to(m)
        items.append((name, col))

    gps = folium.FeatureGroup(name=f"GPS fixes ({len(tr)})", show=True)
    step = max(1, len(tr) // 3000)
    for x, y, t in tr[::step]:
        folium.CircleMarker([y, x], radius=2, color="#264653", fill=True, fill_opacity=0.8,
                            weight=0, tooltip=f"t={t:.0f}s").add_to(gps)
    gps.add_to(m)
    items.append(("GPS fixes", "#264653"))
    folium.Marker([lat[0], lon[0]], tooltip="start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([lat[-1], lon[-1]], tooltip="end", icon=folium.Icon(color="red")).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    add_legend(m, items)
    if title:
        m.get_root().html.add_child(folium.Element(
            f'<div style="position:fixed;top:12px;left:60px;z-index:9999;background:white;padding:6px 10px;'
            f'border:1px solid #999;border-radius:4px;font:14px sans-serif"><b>{title}</b></div>'))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="student_test")
    ap.add_argument("--trips", nargs="+", required=True)
    ap.add_argument("--eda", action="store_true", help="no predictions, just data")
    a = ap.parse_args()

    if not a.eda:
        from pathsnap.pipeline import PathSnap
        snap = PathSnap.load()
    for tid in a.trips:
        case = load_case(os.path.join(SPLITS[a.split], tid))
        preds, title = None, f"{a.split} trip {tid}"
        if not a.eda:
            res = snap.predict_case(case, with_classifier_only=True)
            preds = {"Predicted route (classifier + HMM)": res["route"],
                     "Classifier-only (argmax per fix)": res["classifier_only_route"]}
        m = make_map(case, preds, title=title)
        out = os.path.join(OUT, f"{a.split}_{tid}{'_eda' if a.eda else ''}.html")
        m.save(out)
        print("saved", out)


if __name__ == "__main__":
    main()
