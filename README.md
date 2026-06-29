# Wound Analyzer — Computer Vision
### Intraoperative Wound Assessment System using OpenCV & Python



---

## ![Dashboard](<img width="2581" height="1941" alt="report_dashboard" src="https://github.com/user-attachments/assets/61fae5f0-c19b-4dcf-b668-b8a8c18b6264" />)


---

## Overview

An automated optical inspection system recontextualized from industrial gear defect detection to **intraoperative surgical wound monitoring**. Uses the clinical ABCDE framework adapted for surgical use — processing wound images through a full IPO (Input-Process-Output) computer vision pipeline to generate PASS / CAUTION / FAIL verdicts in real time.

---

## Surgical ABCDE Framework

| Criterion | Surgical Meaning | Metric | Method |
|-----------|-----------------|--------|--------|
| **A** — Asymmetry | Irregular wound shape | Contour solidity | Convex hull ratio |
| **B** — Border | Wound edge clarity | Sharpness score | Laplacian variance |
| **C** — Color | Tissue health status | Healthy pink ratio | HSV pixel classification |
| **D** — Diameter | Wound size vs target | Bounding circle px | vs ±30% tolerance |
| **E** — Evolution | Change between frames | Area delta % | Frame comparison |

---

## Verdict System

| ABCDE Score | Verdict | Meaning |
|-------------|---------|---------|
| 0–1 / 10 | ✅ PASS | Healing normal |
| 2–4 / 10 | ⚠️ CAUTION | Monitor closely |
| 5–10 / 10 | 🛑 FAIL | Critical — intervention required |

---

## Pipeline (IPO Architecture)

**INPUT** — Capture & Clean
- Grayscale conversion (`cv2.cvtColor`)
- Gaussian blur (`cv2.GaussianBlur`)
- Adaptive threshold (`cv2.adaptiveThreshold`)

**PROCESS** — Extract & Measure
- Contour detection (`cv2.findContours`)
- Convex hull (`cv2.convexHull`)
- Convexity defects (`cv2.convexityDefects`)
- HSV color zone analysis (healthy pink vs pale vs necrotic)

**OUTPUT** — Decide & Act
- ABCDE scores (0–10)
- PASS / CAUTION / FAIL verdict
- Annotated bounding box on defect points
- JSON report + visual dashboard

---

## Wound Grades (Synthetic Dataset)

| Grade | Description | Sample Count |
|-------|-------------|--------------|
| Healthy | Symmetric, clear border, pink tissue | 5 |
| Concerning | Mild asymmetry, pale patches, blurred border | 5 |
| Critical | Necrotic tissue, fibrin slough, satellite zones | 5 |

---

## Running

```bash
# Install dependencies
pip3 install opencv-python numpy matplotlib scipy

# Generate synthetic wound images + run full pipeline + build dashboard
python3 main.py
```

Output saved to `output/`:
- `report_dashboard.jpg` — full visual report
- `*_analyzed.jpg` — per-image annotated results
- `abcde_report.json` — machine-readable scores

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Computer Vision | OpenCV 4.x |
| Image Processing | NumPy, SciPy |
| Visualization | Matplotlib |
| Language | Python 3.10+ |

---

## ROS2 Integration

This project is the CV core of a larger ROS2 surgical robotics system:  
👉 [surgical-vision-ros2](https://github.com/Lamia-Islam/surgical-vision-ros2)

---

## Author

**Lamia Islam**  
Pabna University of Science and Technology
