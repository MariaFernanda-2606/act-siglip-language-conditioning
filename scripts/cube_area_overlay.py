#!/usr/bin/env python
"""Live cam_high view with the cube placement area of the training demos drawn on top.

Run in the `lerobot` env (OpenCV with GUI + pyrealsense2), with the policy runner NOT running
(the camera can only be opened by one process):

    conda activate lerobot
    python scripts/cube_area_overlay.py              # area only
    python scripts/cube_area_overlay.py --layout 1   # also draws fixed test layout 1

Shows:
  * solid blue rectangle  = area holding 96% of the cube positions in the 300 demos;
    dashed                = 81% core;
  * detected cubes: circle green if inside the area, red + "FORA" if outside;
  * target squares of the chosen test layout (keys 1-6, 0 = none);
  * camera shift against the recording (phase correlation on the table); if it is more than a few
    pixels the camera moved and the area on the table no longer matches the demos.
Keys: 1-6 layout, 0 no layout, r toggle reference edges, s save snapshot, q / Esc quit.

Data: docs/teste_robo/area_cubos.json and cam_high_referencia.png (median first frame of the demos).
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
AREA_DIR = REPO / "docs" / "teste_robo"
CAM_HIGH_SERIAL = "218622275519"
TABLE = (slice(int(0.36 * 480), int(0.92 * 480)), slice(int(0.20 * 640), int(0.80 * 640)))  # cube search box
COLOURS_BGR = {"red": (40, 40, 225), "yellow": (0, 200, 240), "green": (70, 160, 30)}
NAMES_PT = {"red": "vermelho", "yellow": "amarelo", "green": "verde"}


def detect_cubes(rgb: np.ndarray) -> dict:
    """Same HSV thresholds used to map the cube positions in the demos. Returns {colour: (x, y)}."""
    ys, xs = TABLE
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb[ys, xs]), cv2.COLOR_RGB2HSV)
    h, s, v = (hsv[..., i].astype(int) for i in range(3))
    masks = {
        "red": ((h < 8) | (h > 170)) & (s > 110) & (v > 70),
        "yellow": (h >= 18) & (h <= 35) & (s > 110) & (v > 110),
        "green": (h >= 40) & (h <= 95) & (s > 40) & (v > 25) & (v < 150),
    }
    found = {}
    for c, m in masks.items():
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, st, cen = cv2.connectedComponentsWithStats(m)
        best = None
        for i in range(1, n):
            x, y = cen[i][0] + xs.start, cen[i][1] + ys.start
            # x/y limits drop the drawer stripes and the arms (false positives seen in the demos)
            if 40 <= st[i, 4] <= 900 and y >= 265 and 180 <= x <= 470 and (best is None or st[i, 4] > best[2]):
                best = (x, y, st[i, 4])
        if best:
            found[c] = best[:2]
    return found


def dashed_rect(img, r, colour, thickness=1, dash=8):
    x0, y0, x1, y1 = r
    for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        a, b = np.array(a, float), np.array(b, float)
        n = max(1, int(np.linalg.norm(b - a) // dash))
        for i in range(0, n, 2):
            p, q = a + (b - a) * i / n, a + (b - a) * min(i + 1, n) / n
            cv2.line(img, tuple(p.astype(int)), tuple(q.astype(int)), colour, thickness, cv2.LINE_AA)


def camera_shift(ref_gray: np.ndarray, gray: np.ndarray) -> tuple[float, float]:
    """(dx, dy) of the live image against the reference, on the table region (people move above it)."""
    sl = (slice(150, 470), slice(0, 640))
    a, b = np.float32(ref_gray[sl]), np.float32(gray[sl])
    win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(a, b, win)
    return dx, dy


def render(rgb, area, ref_gray, ref_edges, layout, show_ref):
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if show_ref:
        img[ref_edges > 0] = (img[ref_edges > 0] * 0.4 + np.array([200, 0, 200]) * 0.6).astype(np.uint8)
    blue = (191, 95, 31)
    x0, y0, x1, y1 = area["rect_96"]
    cv2.rectangle(img, (x0, y0), (x1, y1), blue, 2, cv2.LINE_AA)
    dashed_rect(img, area["rect_81"], blue, 1)
    s = area["cube_side_px"]
    if layout:
        for c, (x, y) in area["layouts"][layout - 1].items():
            p, q = (int(x - s / 2), int(y - s / 2)), (int(x + s / 2), int(y + s / 2))
            cv2.rectangle(img, p, q, COLOURS_BGR[c], 2, cv2.LINE_AA)
    cubes = detect_cubes(rgb)
    lines = []
    for c, (x, y) in cubes.items():
        inside = x0 <= x <= x1 and y0 <= y <= y1
        cv2.circle(img, (int(x), int(y)), int(s * 0.75), (60, 200, 60) if inside else (0, 0, 255), 2, cv2.LINE_AA)
        if not inside:
            cv2.putText(img, "FORA", (int(x) - 18, int(y) - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        if layout:
            tx, ty = area["layouts"][layout - 1][c]
            err = np.hypot(x - tx, y - ty) / s
            lines.append(f"{NAMES_PT[c]}: {err:.1f} cubo do alvo")
    missing = [NAMES_PT[c] for c in COLOURS_BGR if c not in cubes]
    if missing:
        lines.append("nao detectado: " + ", ".join(missing))
    dx, dy = camera_shift(ref_gray, cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    moved = max(abs(dx), abs(dy)) > 4
    lines.insert(0, f"camera vs gravacao: dx {dx:+.1f} dy {dy:+.1f} px" + ("  <- CAMERA MEXEU" if moved else "  ok"))
    if layout:
        lines.insert(1, f"layout {layout}")
    for i, t in enumerate(lines):
        col = (0, 0, 255) if (i == 0 and moved) else (255, 255, 255)
        cv2.putText(img, t, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, t, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", type=int, default=0, help="test layout 1-6 to draw (0 = none)")
    ap.add_argument("--image", type=Path, help="use this RGB image instead of the camera (offline check)")
    ap.add_argument("--out", type=Path, help="with --image: write the annotated image here, no window")
    ap.add_argument("--snap-dir", type=Path, default=REPO / "outputs" / "robot")
    args = ap.parse_args()

    area = json.loads((AREA_DIR / "area_cubos.json").read_text())
    ref = cv2.imread(str(AREA_DIR / "cam_high_referencia.png"))
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    ref_edges = cv2.Canny(ref_gray, 60, 140)
    layout, show_ref = args.layout, False

    if args.image:
        rgb = cv2.cvtColor(cv2.imread(str(args.image)), cv2.COLOR_BGR2RGB)
        img = render(rgb, area, ref_gray, ref_edges, layout, show_ref)
        if args.out:
            cv2.imwrite(str(args.out), img)
            return
        cv2.imshow("area dos cubos", img)
        cv2.waitKey(0)
        return

    import pyrealsense2 as rs

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_device(CAM_HIGH_SERIAL)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    pipe.start(cfg)
    try:
        for _ in range(30):  # let auto-exposure settle
            pipe.wait_for_frames()
        while True:
            rgb = np.asanyarray(pipe.wait_for_frames().get_color_frame().get_data()).copy()
            img = render(rgb, area, ref_gray, ref_edges, layout, show_ref)
            cv2.imshow("area dos cubos (q sai, 1-6 layout, r referencia, s salva)", cv2.resize(img, (960, 720)))
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if ord("0") <= k <= ord("6"):
                layout = k - ord("0")
            elif k == ord("r"):
                show_ref = not show_ref
            elif k == ord("s"):
                args.snap_dir.mkdir(parents=True, exist_ok=True)
                p = args.snap_dir / f"area_check_{time.strftime('%Y%m%d_%H%M%S')}_layout{layout}.png"
                cv2.imwrite(str(p), img)
                print("saved", p)
    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
