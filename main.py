"""
main.py
──────────────────────────────────────────────────────────────────────────────
Entry point for the Intraoperative Wound Assessment System.

Steps:
  1. Generate synthetic wound images (3 grades × 5 samples = 15 images)
  2. Run ABCDE analysis pipeline on all images
  3. Build a visual dashboard (matplotlib grid) saved as report_dashboard.jpg
──────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")                 # headless – no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# local modules
sys.path.insert(0, os.path.dirname(__file__))
from generate_wounds  import generate_healthy, generate_concerning, generate_critical
from wound_analyzer   import analyze_wound, run_batch

OUTPUT_DIR     = "output"
SAMPLE_DIR     = "sample_images"


# ─── Step 1: generate images ─────────────────────────────────────────────────
def generate_dataset():
    print("\n[1/3]  Generating synthetic surgical wound images …")
    for i in range(1, 6):
        generate_healthy(i)
        generate_concerning(i)
        generate_critical(i)
    print(f"       15 images written → ./{SAMPLE_DIR}/\n")


# ─── Step 2: run ABCDE pipeline ──────────────────────────────────────────────
def run_pipeline():
    print("[2/3]  Running ABCDE wound analysis pipeline …\n")
    results = run_batch(SAMPLE_DIR)
    return results


# ─── Step 3: build dashboard ─────────────────────────────────────────────────
VERDICT_COLOR_MAP = {
    "PASS":    "#00c832",
    "CAUTION": "#ffb400",
    "FAIL":    "#e3001b",
}

def verdict_hex(verdict: str) -> str:
    for k, v in VERDICT_COLOR_MAP.items():
        if k in verdict:
            return v
    return "#888888"


def build_dashboard(results):
    print("[3/3]  Building visual dashboard …")

    # pick one representative image per grade from analyzed outputs
    def pick(keyword):
        matches = [r for r in results if keyword in os.path.basename(r.filename)]
        return matches[0] if matches else None

    samples = [pick("healthy_01"), pick("concerning_01"), pick("critical_01")]
    samples = [s for s in samples if s is not None]

    n_samples = len(samples)

    fig = plt.figure(figsize=(18, 14), facecolor="#0d0d0d")
    fig.suptitle(
        "Intraoperative Wound Assessment System  –  Surgical ABCDE Report",
        fontsize=16, color="white", fontweight="bold", y=0.97
    )

    gs = GridSpec(3, n_samples + 1, figure=fig,
                  hspace=0.45, wspace=0.3,
                  left=0.05, right=0.97, top=0.92, bottom=0.06)

    categories = ["A\nAsymmetry", "B\nBorder", "C\nColor", "D\nDiameter", "E\nEvolution"]
    bar_colors = ["#1e90ff", "#00c8a0", "#ffa500", "#c87aff", "#ff6060"]

    for col, report in enumerate(samples):
        # ── row 0: annotated image ──────────────────────────────────────────
        ax_img = fig.add_subplot(gs[0, col])
        out_name = os.path.splitext(os.path.basename(report.filename))[0] + "_analyzed.jpg"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        if os.path.exists(out_path):
            bgr  = cv2.imread(out_path)
            rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            ax_img.imshow(rgb)
        ax_img.set_title(
            os.path.basename(report.filename),
            fontsize=8, color="#aaaaaa", pad=3
        )
        vc = verdict_hex(report.verdict)
        ax_img.set_xlabel(report.verdict, fontsize=8,
                          color=vc, fontweight="bold", labelpad=4)
        ax_img.axis("off")
        for spine in ax_img.spines.values():
            spine.set_edgecolor(vc)
            spine.set_linewidth(2)

        # ── row 1: ABCDE radar / bar chart ─────────────────────────────────
        ax_bar = fig.add_subplot(gs[1, col])
        scores = [report.A_score, report.B_score, report.C_score,
                  report.D_score, report.E_score]
        bars   = ax_bar.bar(categories, scores, color=bar_colors,
                            edgecolor="white", linewidth=0.5)
        ax_bar.set_ylim(0, 2.5)
        ax_bar.set_yticks([0, 1, 2])
        ax_bar.set_yticklabels(["Normal", "Mild", "Severe"],
                               fontsize=7, color="#cccccc")
        ax_bar.set_title(f"Score: {report.total_score}/10",
                         fontsize=9, color="white", pad=4)
        ax_bar.set_facecolor("#1a1a1a")
        ax_bar.tick_params(axis="x", colors="#bbbbbb", labelsize=7)
        ax_bar.spines[:].set_color("#444444")
        for bar, sc in zip(bars, scores):
            ax_bar.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.05,
                        str(sc), ha="center", va="bottom",
                        fontsize=8, color="white", fontweight="bold")

        # ── row 2: measurement table ────────────────────────────────────────
        ax_tbl = fig.add_subplot(gs[2, col])
        ax_tbl.set_facecolor("#111111")
        ax_tbl.axis("off")

        rows = [
            ["Metric",       "Value",                            "Status"],
            ["Solidity",     f"{report.solidity:.3f}",          "✓" if report.A_score==0 else "⚠"],
            ["Border Shp",   f"{report.border_sharpness:.1f}",  "✓" if report.B_score==0 else "⚠"],
            ["Healthy Ratio",f"{report.color_score:.3f}",       "✓" if report.C_score==0 else "⚠"],
            ["Diameter px",  f"{report.diameter_px:.0f}",       "✓" if report.D_score==0 else "⚠"],
            ["Evolution %",  f"{report.evolution_pct*100:.1f}%","✓" if report.E_score==0 else "⚠"],
            ["Defect Pts",   str(len(report.defect_points)),    "—"],
        ]
        tbl = ax_tbl.table(
            cellText   = rows[1:],
            colLabels  = rows[0],
            cellLoc    = "center",
            loc        = "center",
            bbox       = [0, 0, 1, 1]
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_facecolor("#1e1e1e" if r % 2 == 0 else "#161616")
            cell.set_text_props(color="#dddddd")
            cell.set_edgecolor("#333333")
            if r == 0:
                cell.set_facecolor("#2a2a2a")
                cell.set_text_props(color="white", fontweight="bold")
            if c == 2 and r > 0:
                txt = cell.get_text().get_text()
                cell.set_text_props(color="#00c832" if txt == "✓" else
                                          ("#ffb400" if txt == "⚠" else "#888888"))

    # ── right column: system legend & summary ────────────────────────────────
    ax_leg = fig.add_subplot(gs[:, -1])
    ax_leg.set_facecolor("#111111")
    ax_leg.axis("off")

    legend_text = (
        "SURGICAL ABCDE FRAMEWORK\n"
        "──────────────────────────\n\n"
        "A  ASYMMETRY\n"
        "   Irregular wound shape\n"
        "   Metric: contour solidity\n\n"
        "B  BORDER\n"
        "   Wound edge clarity\n"
        "   Metric: Laplacian variance\n\n"
        "C  COLOR\n"
        "   Tissue health status\n"
        "   Pink=healthy, dark=necrotic\n"
        "   pale=ischemic\n\n"
        "D  DIAMETER\n"
        "   Size vs surgical target\n"
        f"   Target: {180} px  (±30%)\n\n"
        "E  EVOLUTION\n"
        "   Area change between frames\n"
        "   Alert: >25% change\n\n"
        "──────────────────────────\n"
        "SCORING\n"
        "  0/10 – 1/10  →  PASS\n"
        "  2/10 – 4/10  →  CAUTION\n"
        "  5/10 – 10/10 →  CRITICAL\n\n"
        "──────────────────────────\n"
        "PIPELINE  (IPO)\n"
        "  INPUT:    Grayscale + Blur\n"
        "            Adaptive Threshold\n"
        "  PROCESS:  Contours\n"
        "            Convex Hull\n"
        "            Convexity Defects\n"
        "            HSV Color Analysis\n"
        "  OUTPUT:   ABCDE Scores\n"
        "            Verdict + Bbox\n\n"
        "──────────────────────────\n"
        "Powered by DecodeLabs\n"
        "Batch 2026"
    )
    ax_leg.text(
        0.05, 0.97, legend_text,
        transform=ax_leg.transAxes,
        fontsize=7.8, color="#cccccc",
        verticalalignment="top",
        fontfamily="monospace",
        linespacing=1.55
    )

    # verdict patch legend
    patches = [
        mpatches.Patch(color="#00c832", label="PASS  — Healing Normal"),
        mpatches.Patch(color="#ffb400", label="CAUTION — Monitor"),
        mpatches.Patch(color="#e3001b", label="FAIL  — Critical Wound"),
    ]
    ax_leg.legend(handles=patches, loc="lower center",
                  fontsize=7.5, framealpha=0.2,
                  labelcolor="white", facecolor="#1a1a1a",
                  edgecolor="#444444")

    # save
    dash_path = os.path.join(OUTPUT_DIR, "report_dashboard.jpg")
    plt.savefig(dash_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"       Dashboard saved → {dash_path}\n")
    return dash_path


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    generate_dataset()
    results   = run_pipeline()
    dash_path = build_dashboard(results)
    print("✅  All done.")
    print(f"   Dashboard : wound_analyzer/output/report_dashboard.jpg")
    print(f"   JSON log  : wound_analyzer/output/abcde_report.json")
    print(f"   Per-image : wound_analyzer/output/*_analyzed.jpg\n")
