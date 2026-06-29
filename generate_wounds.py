"""
generate_wounds.py
──────────────────────────────────────────────────────────────────────────────
Synthetic intraoperative wound image generator.
Produces 3 wound grades:
  • HEALTHY   – symmetric, clear border, pink tissue, target diameter
  • CONCERNING – mild asymmetry, blurred border, pale/dark patches
  • CRITICAL   – severe irregularity, necrotic dark tissue, oversized, satellite zones

Each image simulates an overhead surgical camera view under OR lighting.
──────────────────────────────────────────────────────────────────────────────
"""

import cv2
import numpy as np
import os

OUTPUT_DIR = "sample_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RNG = np.random.default_rng(2024)

IMG_SIZE   = 512
CENTER     = (IMG_SIZE // 2, IMG_SIZE // 2)

# ── Tissue color palette (BGR) ────────────────────────────────────────────────
SKIN_BASE      = (140, 160, 190)   # surgical drape / surrounding skin tone
HEALTHY_PINK   = (100, 130, 210)   # well-perfused granulation tissue
PALE_TISSUE    = (160, 170, 200)   # ischemic / poorly perfused
NECROTIC_DARK  = ( 30,  35,  55)   # necrotic / eschar
BLOOD_RED      = ( 40,  50, 160)   # active bleed / erythema
FIBRIN_YELLOW  = ( 80, 190, 220)   # fibrinous slough
SUTURE_COLOR   = ( 20,  20,  20)   # suture thread


# ─────────────────────────────────────────────────────────────────────────────
#  CANVAS  – simulates OR drape + lighting
# ─────────────────────────────────────────────────────────────────────────────
def make_canvas():
    img = np.full((IMG_SIZE, IMG_SIZE, 3), SKIN_BASE, dtype=np.uint8)
    # subtle OR light gradient (brighter center)
    for i in range(IMG_SIZE):
        for j in range(IMG_SIZE):
            dist = np.sqrt((i - IMG_SIZE//2)**2 + (j - IMG_SIZE//2)**2)
            factor = np.clip(1.0 - dist / (IMG_SIZE * 0.9), 0.85, 1.0)
            img[i, j] = np.clip(np.array(img[i, j]) * factor, 0, 255)
    # fine sensor noise
    noise = RNG.integers(-6, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  WOUND MASK GENERATORS
# ─────────────────────────────────────────────────────────────────────────────
def ellipse_mask(rx, ry, angle=0):
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    cv2.ellipse(mask, CENTER, (rx, ry), angle, 0, 360, 255, -1)
    return mask


def irregular_mask(mean_radius, irregularity=0.3, n_pts=24):
    """Polygon blob with controllable irregularity."""
    angles  = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    radii   = mean_radius + RNG.uniform(
        -mean_radius * irregularity,
         mean_radius * irregularity,
        n_pts
    )
    radii   = np.clip(radii, 20, IMG_SIZE // 2 - 20)
    pts_x   = (CENTER[0] + radii * np.cos(angles)).astype(np.int32)
    pts_y   = (CENTER[1] + radii * np.sin(angles)).astype(np.int32)
    pts     = np.stack([pts_x, pts_y], axis=1).reshape(-1, 1, 2)
    mask    = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask, pts


# ─────────────────────────────────────────────────────────────────────────────
#  TEXTURE FILL  – paints tissue texture onto wound region
# ─────────────────────────────────────────────────────────────────────────────
def paint_tissue(img, mask, base_color, noise_range=18):
    """Paint a noisy tissue color into mask region."""
    overlay = img.copy()
    ys, xs  = np.where(mask > 0)
    for y, x in zip(ys, xs):
        n   = RNG.integers(-noise_range, noise_range, 3)
        col = np.clip(np.array(base_color) + n, 0, 255).astype(np.uint8)
        overlay[y, x] = col
    # micro-texture via Gaussian blur
    overlay = cv2.GaussianBlur(overlay, (3, 3), 0.8)
    img[mask > 0] = overlay[mask > 0]
    return img


def add_specular_glint(img, mask):
    """Simulate moist tissue reflectance (OR light glint)."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return img
    n_glints = RNG.integers(3, 8)
    for _ in range(n_glints):
        idx = RNG.integers(0, len(ys))
        cv2.circle(img, (xs[idx], ys[idx]), RNG.integers(2, 6),
                   (230, 240, 255), -1)
    return img


def add_sutures(img, cx, cy, radius, n=6):
    """Draw evenly-spaced suture stitches around wound border."""
    for i in range(n):
        angle  = 2 * np.pi * i / n
        bx     = int(cx + radius * np.cos(angle))
        by     = int(cy + radius * np.sin(angle))
        inner  = (int(cx + (radius - 12) * np.cos(angle)),
                  int(cy + (radius - 12) * np.sin(angle)))
        outer  = (int(cx + (radius + 12) * np.cos(angle)),
                  int(cy + (radius + 12) * np.sin(angle)))
        cv2.line(img, inner, outer, SUTURE_COLOR, 1, cv2.LINE_AA)
        cv2.circle(img, (bx, by), 2, SUTURE_COLOR, -1)
    return img


# ─────────────────────────────────────────────────────────────────────────────
#  GRADE 1 – HEALTHY WOUND
#  Symmetric ellipse | clear border | uniform pink | target diameter | stable
# ─────────────────────────────────────────────────────────────────────────────
def generate_healthy(idx):
    img  = make_canvas()
    rx, ry = 90, 85                       # near-circular, controlled size

    mask = ellipse_mask(rx, ry)
    img  = paint_tissue(img, mask, HEALTHY_PINK, noise_range=14)
    img  = add_specular_glint(img, mask)

    # clean, sharp border
    cv2.ellipse(img, CENTER, (rx, ry), 0, 0, 360, (60, 80, 140), 2, cv2.LINE_AA)

    # suture track
    img  = add_sutures(img, *CENTER, rx + 8, n=8)

    path = os.path.join(OUTPUT_DIR, f"healthy_{idx:02d}.jpg")
    cv2.imwrite(path, img)
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  GRADE 2 – CONCERNING WOUND
#  Mild asymmetry | fuzzy border | pale patches | slightly oversized
# ─────────────────────────────────────────────────────────────────────────────
def generate_concerning(idx):
    img  = make_canvas()

    mask, pts = irregular_mask(mean_radius=100, irregularity=0.22)
    img = paint_tissue(img, mask, HEALTHY_PINK, noise_range=20)

    # pale ischemic patch (1–2 zones)
    for _ in range(RNG.integers(1, 3)):
        px = CENTER[0] + RNG.integers(-50, 50)
        py = CENTER[1] + RNG.integers(-50, 50)
        pm = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        cv2.ellipse(pm, (px, py),
                    (RNG.integers(18, 35), RNG.integers(18, 35)),
                    RNG.integers(0, 180), 0, 360, 255, -1)
        pm = cv2.bitwise_and(pm, mask)
        img = paint_tissue(img, pm, PALE_TISSUE, noise_range=10)

    img = add_specular_glint(img, mask)

    # fuzzy / blurred border
    border = np.zeros_like(mask)
    cv2.polylines(border, [pts], True, 255, 4)
    border_blur = cv2.GaussianBlur(border, (11, 11), 4)
    alpha = border_blur.astype(np.float32) / 255.0
    dark  = np.full_like(img, (50, 65, 110))
    for c in range(3):
        img[:, :, c] = (alpha * dark[:, :, c] + (1 - alpha) * img[:, :, c]).astype(np.uint8)

    img = add_sutures(img, *CENTER, 108, n=8)

    path = os.path.join(OUTPUT_DIR, f"concerning_{idx:02d}.jpg")
    cv2.imwrite(path, img)
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  GRADE 3 – CRITICAL WOUND
#  Severe irregularity | ragged border | necrotic + fibrin | oversized
#  + satellite tissue zones (dehiscence indicators)
# ─────────────────────────────────────────────────────────────────────────────
def generate_critical(idx):
    img  = make_canvas()

    mask, pts = irregular_mask(mean_radius=130, irregularity=0.48)
    img = paint_tissue(img, mask, HEALTHY_PINK, noise_range=16)

    # necrotic dark zone (centre)
    nm = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    cv2.ellipse(nm, CENTER,
                (RNG.integers(35, 55), RNG.integers(35, 55)),
                RNG.integers(0, 180), 0, 360, 255, -1)
    nm = cv2.bitwise_and(nm, mask)
    img = paint_tissue(img, nm, NECROTIC_DARK, noise_range=8)

    # fibrinous slough patches
    for _ in range(RNG.integers(2, 5)):
        fx = CENTER[0] + RNG.integers(-70, 70)
        fy = CENTER[1] + RNG.integers(-70, 70)
        fm = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        cv2.ellipse(fm, (fx, fy),
                    (RNG.integers(12, 28), RNG.integers(12, 28)),
                    RNG.integers(0, 180), 0, 360, 255, -1)
        fm = cv2.bitwise_and(fm, mask)
        img = paint_tissue(img, fm, FIBRIN_YELLOW, noise_range=12)

    # erythema ring (inflammation)
    ring = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    cv2.ellipse(ring, CENTER, (140, 140), 0, 0, 360, 255, -1)
    cv2.ellipse(ring, CENTER, (130, 130), 0, 0, 360,   0, -1)
    ring = cv2.bitwise_and(ring, cv2.bitwise_not(mask))
    img  = paint_tissue(img, ring, BLOOD_RED, noise_range=20)

    # satellite tissue / dehiscence spots
    for _ in range(RNG.integers(3, 6)):
        sx = CENTER[0] + RNG.integers(-160, 160)
        sy = CENTER[1] + RNG.integers(-160, 160)
        sr = RNG.integers(8, 20)
        sm = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        cv2.circle(sm, (sx, sy), sr, 255, -1)
        col = RNG.choice([NECROTIC_DARK, PALE_TISSUE, FIBRIN_YELLOW])
        img = paint_tissue(img, sm, col, noise_range=10)

    img  = add_specular_glint(img, mask)

    # ragged border — no smoothing
    cv2.polylines(img, [pts], True, (15, 20, 40), 3, cv2.LINE_AA)

    path = os.path.join(OUTPUT_DIR, f"critical_{idx:02d}.jpg")
    cv2.imwrite(path, img)
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    healthy    = [generate_healthy(i)    for i in range(1, 6)]
    concerning = [generate_concerning(i) for i in range(1, 6)]
    critical   = [generate_critical(i)   for i in range(1, 6)]

    total = len(healthy) + len(concerning) + len(critical)
    print(f"✅  Generated {total} synthetic wound images → ./{OUTPUT_DIR}/")
    print(f"   Healthy:    {len(healthy)}")
    print(f"   Concerning: {len(concerning)}")
    print(f"   Critical:   {len(critical)}")
