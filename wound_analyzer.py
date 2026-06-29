"""
wound_analyzer.py
──────────────────────────────────────────────────────────────────────────────
Intraoperative Wound Assessment System
Surgical ABCDE Framework (adapted from dermatology ABCDE rule)

  A – Asymmetry    → irregular wound shape  (contour solidity & hu moments)
  B – Border       → wound edge clarity     (border gradient sharpness)
  C – Color        → tissue health status   (HSV color zone analysis)
  D – Diameter     → wound size vs target   (contour area / bounding box)
  E – Evolution    → change between frames  (frame-diff or first-vs-current)

Pipeline (from Project 2 IPO architecture):
  INPUT   → Capture & Clean  (grayscale, Gaussian blur, threshold)
  PROCESS → Extract & Measure (contours, convex hull, convexity defects)
  OUTPUT  → Decide & Act     (ABCDE scores → PASS / CAUTION / CRITICAL)
──────────────────────────────────────────────────────────────────────────────
"""

import cv2
import numpy as np
import os
import json
from dataclasses import dataclass, field
from typing import Optional

# ─── Configuration ────────────────────────────────────────────────────────────
TARGET_DIAMETER_PX   = 180          # expected wound diameter (pixels)
DIAMETER_TOLERANCE   = 0.30         # ±30 % acceptable
ASYMMETRY_THRESHOLD  = 0.82         # solidity below this → irregular
BORDER_SHARP_MIN     = 18.0         # Laplacian variance; below → blurry border
EVOLUTION_ALERT_PCT  = 0.25         # >25% area change between frames → flag

OUTPUT_DIR           = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Data container ───────────────────────────────────────────────────────────
@dataclass
class ABCDEReport:
    filename:       str
    # Raw measurements
    contour_area:   float  = 0.0
    hull_area:      float  = 0.0
    solidity:       float  = 0.0     # contour_area / hull_area  (A)
    border_sharpness: float = 0.0    # Laplacian variance on border strip (B)
    color_score:    float  = 0.0     # 0-1, healthy pink ratio  (C)
    diameter_px:    float  = 0.0     # max bounding dimension    (D)
    evolution_pct:  float  = 0.0     # % area change vs previous (E)
    # ABCDE sub-scores (0 = normal, 1 = mild, 2 = severe)
    A_score: int = 0
    B_score: int = 0
    C_score: int = 0
    D_score: int = 0
    E_score: int = 0
    # Final
    total_score: int  = 0
    verdict:     str  = "UNKNOWN"
    verdict_color: tuple = field(default_factory=lambda: (0, 0, 0))
    defect_points: list = field(default_factory=list)   # convexity defect coords


# ─── Phase 1: IPO INPUT – Capture & Clean ────────────────────────────────────
def preprocess(img: np.ndarray):
    """
    Grayscale → Gaussian Blur → Adaptive Threshold
    Returns (gray, blurred, thresh, wound_mask)
    """
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 2)

    # Adaptive threshold handles non-uniform OR lighting better than global
    thresh  = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31, C=6
    )

    # Morphological cleanup – remove tiny noise blobs
    kernel      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    wound_mask  = cv2.morphologyEx(thresh,  cv2.MORPH_CLOSE, kernel, iterations=2)
    wound_mask  = cv2.morphologyEx(wound_mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    return gray, blurred, thresh, wound_mask


def extract_primary_contour(wound_mask: np.ndarray):
    """Return the largest contour (primary wound boundary)."""
    contours, _ = cv2.findContours(
        wound_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


# ─── Phase 2: IPO PROCESS – Extract & Measure (ABCDE) ────────────────────────

# A – Asymmetry
def score_asymmetry(contour) -> tuple[int, float, float]:
    """
    Solidity = contour_area / convex_hull_area.
    Low solidity → irregular shape → asymmetric wound.
    """
    c_area = cv2.contourArea(contour)
    hull   = cv2.convexHull(contour)
    h_area = cv2.contourArea(hull)
    if h_area == 0:
        return 2, 0.0, 0.0
    solidity = c_area / h_area
    if   solidity >= ASYMMETRY_THRESHOLD:          score = 0   # symmetric
    elif solidity >= ASYMMETRY_THRESHOLD - 0.15:   score = 1   # mildly irregular
    else:                                           score = 2   # severely irregular
    return score, float(c_area), float(h_area)


# B – Border
def score_border(img_gray: np.ndarray, contour) -> tuple[int, float]:
    """
    Measure Laplacian variance along a thin strip around the wound border.
    High variance = sharp, well-defined border (healthy healing).
    Low variance  = blurry / diffuse border (concerning).
    """
    border_mask = np.zeros_like(img_gray)
    cv2.drawContours(border_mask, [contour], -1, 255, thickness=8)
    border_region = cv2.bitwise_and(img_gray, img_gray, mask=border_mask)
    lap   = cv2.Laplacian(border_region, cv2.CV_64F)
    var   = float(lap.var())
    if   var >= BORDER_SHARP_MIN * 2:  score = 0
    elif var >= BORDER_SHARP_MIN:      score = 1
    else:                              score = 2
    return score, var


# C – Color (tissue health)
def score_color(img_bgr: np.ndarray, contour) -> tuple[int, float]:
    """
    Analyse tissue color in HSV space inside wound boundary.
    Healthy granulation tissue = warm pink/red hue (H: 0-20, 160-180 in HSV).
    Pale ischemic tissue       = low saturation.
    Necrotic tissue            = very dark (low V).
    Returns ratio of healthy-pink pixels.
    """
    mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    hsv     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    total   = int(np.sum(mask > 0))
    if total == 0:
        return 2, 0.0

    # Healthy pink/red ranges in HSV
    pink_low1  = cv2.inRange(hsv, (0,   60,  80), (20,  255, 255))
    pink_low2  = cv2.inRange(hsv, (160, 60,  80), (180, 255, 255))
    healthy    = cv2.bitwise_or(pink_low1, pink_low2)

    # Necrotic: very dark
    necrotic   = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))

    # Pale: low saturation + mid value
    pale       = cv2.inRange(hsv, (0, 0, 100), (180, 40, 200))

    healthy_px  = int(np.sum(cv2.bitwise_and(healthy,  mask) > 0))
    necrotic_px = int(np.sum(cv2.bitwise_and(necrotic, mask) > 0))
    pale_px     = int(np.sum(cv2.bitwise_and(pale,     mask) > 0))

    healthy_ratio  = healthy_px  / total
    necrotic_ratio = necrotic_px / total

    if   necrotic_ratio > 0.15:              score = 2   # necrotic tissue present
    elif healthy_ratio  < 0.25:              score = 1   # insufficient perfusion
    else:                                    score = 0   # good pink tissue
    return score, float(healthy_ratio)


# D – Diameter
def score_diameter(contour) -> tuple[int, float]:
    """
    Bounding circle diameter vs TARGET_DIAMETER_PX.
    """
    _, radius = cv2.minEnclosingCircle(contour)
    diameter  = radius * 2
    ratio     = diameter / TARGET_DIAMETER_PX
    deviation = abs(ratio - 1.0)
    if   deviation <= DIAMETER_TOLERANCE:              score = 0
    elif deviation <= DIAMETER_TOLERANCE * 2:          score = 1
    else:                                              score = 2
    return score, float(diameter)


# E – Evolution (frame comparison)
def score_evolution(current_area: float,
                    previous_area: Optional[float]) -> tuple[int, float]:
    """
    Compares wound area to previous frame (or first image in batch).
    """
    if previous_area is None or previous_area == 0:
        return 0, 0.0
    change = abs(current_area - previous_area) / previous_area
    if   change <= EVOLUTION_ALERT_PCT * 0.5:   score = 0
    elif change <= EVOLUTION_ALERT_PCT:          score = 1
    else:                                        score = 2
    return score, float(change)


# Convexity defects → structural breakdown points
def find_defect_points(contour, min_depth=10.0) -> list:
    """
    Returns (x,y) coordinates of significant convexity defect points.
    These mark locations of wound border collapse / dehiscence.
    """
    if len(contour) < 5:
        return []
    hull    = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 3:
        return []
    try:
        defects = cv2.convexityDefects(contour, hull)
    except cv2.error:
        return []
    if defects is None:
        return []
    points = []
    for d in defects:
        s, e, f, depth_raw = d[0]
        depth = depth_raw / 256.0
        if depth > min_depth:
            fx, fy = contour[f][0]
            points.append((int(fx), int(fy)))
    return points


# ─── Phase 3: IPO OUTPUT – Decide & Act ─────────────────────────────────────

VERDICT_TABLE = {
    (0, 1): ("PASS — HEALING NORMAL",  (0, 200,  50)),
    (2, 4): ("CAUTION — MONITOR",      (0, 180, 255)),
    (5, 8): ("FAIL — CRITICAL WOUND",  (0,  30, 220)),
    (9,10): ("FAIL — CRITICAL WOUND",  (0,  30, 220)),
}

def determine_verdict(total: int) -> tuple[str, tuple]:
    if   total <= 1: return "PASS — HEALING NORMAL",  (0, 200,  50)
    elif total <= 4: return "CAUTION — MONITOR",       (0, 180, 255)
    else:            return "FAIL — CRITICAL WOUND",   (0,  30, 220)


def draw_annotated_output(img_bgr: np.ndarray,
                          contour,
                          report: ABCDEReport) -> np.ndarray:
    """
    Draw clinical overlay on the image:
    - Wound boundary contour
    - Convexity defect points (dehiscence markers)
    - Bounding box
    - ABCDE score panel
    - Verdict banner
    """
    out = img_bgr.copy()
    vc  = report.verdict_color

    # 1. Wound contour
    cv2.drawContours(out, [contour], -1, vc, 2, cv2.LINE_AA)

    # 2. Convex hull overlay
    hull_pts = cv2.convexHull(contour)
    cv2.polylines(out, [hull_pts], True, (200, 200, 0), 1, cv2.LINE_AA)

    # 3. Dehiscence / defect markers
    for (dx, dy) in report.defect_points:
        cv2.circle(out, (dx, dy), 6, (0, 0, 255), -1)
        cv2.circle(out, (dx, dy), 8, (255, 255, 255), 1)

    # 4. Bounding box
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(out, (x, y), (x+w, y+h), (200, 200, 200), 1)

    # 5. Score panel (semi-transparent background)
    panel_h = 220
    panel   = out.copy()
    cv2.rectangle(panel, (0, 0), (260, panel_h), (10, 10, 10), -1)
    cv2.addWeighted(panel, 0.65, out, 0.35, 0, out)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    lh    = 28
    items = [
        (f"A Asymmetry   [{report.A_score}/2]  sol={report.solidity:.2f}",   report.A_score),
        (f"B Border      [{report.B_score}/2]  shp={report.border_sharpness:.1f}", report.B_score),
        (f"C Color       [{report.C_score}/2]  hlth={report.color_score:.2f}", report.C_score),
        (f"D Diameter    [{report.D_score}/2]  {report.diameter_px:.0f}px",   report.D_score),
        (f"E Evolution   [{report.E_score}/2]  Δ={report.evolution_pct*100:.1f}%", report.E_score),
        (f"TOTAL SCORE   [{report.total_score}/10]", -1),
    ]
    score_colors = {0: (0, 220, 80), 1: (0, 200, 255), 2: (0, 60, 255)}
    for i, (text, sc) in enumerate(items):
        col = score_colors.get(sc, (200, 200, 200))
        cv2.putText(out, text, (8, 24 + i * lh),
                    font, 0.45, col, 1, cv2.LINE_AA)

    # 6. Verdict banner
    banner_y = out.shape[0] - 46
    cv2.rectangle(out, (0, banner_y), (out.shape[1], out.shape[0]), (10, 10, 10), -1)
    cv2.putText(out, report.verdict,
                (12, banner_y + 30), font, 0.75, vc, 2, cv2.LINE_AA)

    # 7. Filename watermark
    cv2.putText(out, os.path.basename(report.filename),
                (out.shape[1] - 220, out.shape[0] - 12),
                font, 0.38, (160, 160, 160), 1, cv2.LINE_AA)

    return out


# ─── Main analysis function ───────────────────────────────────────────────────
def analyze_wound(img_path: str,
                  previous_area: Optional[float] = None) -> Optional[ABCDEReport]:
    img = cv2.imread(img_path)
    if img is None:
        print(f"  ✗  Cannot read: {img_path}")
        return None

    report           = ABCDEReport(filename=img_path)
    gray, _, _, mask = preprocess(img)
    contour          = extract_primary_contour(mask)

    if contour is None or cv2.contourArea(contour) < 200:
        print(f"  ✗  No wound contour found: {img_path}")
        return None

    # ABCDE scoring
    report.A_score, report.contour_area, report.hull_area = score_asymmetry(contour)
    report.solidity = report.contour_area / report.hull_area if report.hull_area else 0

    report.B_score, report.border_sharpness = score_border(gray, contour)

    report.C_score, report.color_score = score_color(img, contour)

    report.D_score, report.diameter_px = score_diameter(contour)

    report.E_score, report.evolution_pct = score_evolution(
        report.contour_area, previous_area
    )

    report.defect_points = find_defect_points(contour)

    report.total_score = (report.A_score + report.B_score +
                          report.C_score + report.D_score + report.E_score)
    report.verdict, report.verdict_color = determine_verdict(report.total_score)

    # Save annotated image
    annotated = draw_annotated_output(img, contour, report)
    out_name  = os.path.splitext(os.path.basename(img_path))[0] + "_analyzed.jpg"
    out_path  = os.path.join(OUTPUT_DIR, out_name)
    cv2.imwrite(out_path, annotated)

    return report


# ─── Batch runner ─────────────────────────────────────────────────────────────
def run_batch(image_dir: str = "sample_images"):
    paths = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    if not paths:
        print("No images found. Run generate_wounds.py first.")
        return

    print(f"\n{'='*68}")
    print(f"  INTRAOPERATIVE WOUND ASSESSMENT SYSTEM  |  {len(paths)} images")
    print(f"  Surgical ABCDE Framework")
    print(f"{'='*68}\n")

    previous_area = None
    results       = []

    for path in paths:
        report = analyze_wound(path, previous_area)
        if report is None:
            continue
        previous_area = report.contour_area
        results.append(report)

        flag_A = "⚠" if report.A_score > 0 else "✓"
        flag_B = "⚠" if report.B_score > 0 else "✓"
        flag_C = "⚠" if report.C_score > 0 else "✓"
        flag_D = "⚠" if report.D_score > 0 else "✓"
        flag_E = "⚠" if report.E_score > 0 else "✓"

        print(f"  {os.path.basename(path):<30}  Score: {report.total_score:>2}/10  →  {report.verdict}")
        print(f"    A{flag_A} sol={report.solidity:.2f}  "
              f"B{flag_B} shp={report.border_sharpness:.1f}  "
              f"C{flag_C} hlth={report.color_score:.2f}  "
              f"D{flag_D} {report.diameter_px:.0f}px  "
              f"E{flag_E} Δ{report.evolution_pct*100:.1f}%  "
              f"| defects={len(report.defect_points)}")
        print()

    # Summary
    verdicts = [r.verdict for r in results]
    pass_n   = sum(1 for v in verdicts if "PASS"    in v)
    caut_n   = sum(1 for v in verdicts if "CAUTION" in v)
    fail_n   = sum(1 for v in verdicts if "FAIL"    in v)

    print(f"{'─'*68}")
    print(f"  SUMMARY   PASS: {pass_n}  |  CAUTION: {caut_n}  |  CRITICAL: {fail_n}")
    print(f"  Annotated images saved → ./{OUTPUT_DIR}/")
    print(f"{'='*68}\n")

    # Save JSON report
    json_data = [
        {
            "file":          os.path.basename(r.filename),
            "verdict":       r.verdict,
            "total_score":   r.total_score,
            "A_asymmetry":   r.A_score,
            "B_border":      r.B_score,
            "C_color":       r.C_score,
            "D_diameter":    r.D_score,
            "E_evolution":   r.E_score,
            "solidity":      round(r.solidity, 4),
            "border_sharp":  round(r.border_sharpness, 2),
            "color_ratio":   round(r.color_score, 4),
            "diameter_px":   round(r.diameter_px, 1),
            "evolution_pct": round(r.evolution_pct * 100, 2),
            "defect_count":  len(r.defect_points),
        }
        for r in results
    ]
    json_path = os.path.join(OUTPUT_DIR, "abcde_report.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  JSON report → {json_path}\n")

    return results


if __name__ == "__main__":
    run_batch("sample_images")
