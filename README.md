# Plant-as-ID: Leaf-Vein Biometric Identification for Botanical Traceability

> Major Project, Department of CSE (Artificial Intelligence & Machine Learning)
> Sai Vidya Institute of Technology (SVIT), Rajanukunte, Bengaluru

| | |
|---|---|
| **Project title** | Plant-as-ID: Leaf-Vein Biometrics for a Blockchain-Based Botanical Traceability System |
| **Team** | Team 13 |
| **Team members** | Aditya Krishnan (1VA23CI003), Ajay Kumar (1VA23CI005), Anurag Jangid (1VA23CI012), Sachin Duhan (1VA23CI098) |
| **Project guide** | Prof. Deepika G S |
| **Academic year** | 2026-27 |
| **Institution** | Sai Vidya Institute of Technology, Rajanukunte, Bengaluru 560 119, Karnataka |

---

## 1. About the Institution

**Sai Vidya Institute of Technology (SVIT)** is an engineering college located at Rajanukunte, Bengaluru, Karnataka (PIN 560 119). It is approved by AICTE, New Delhi, and affiliated to Visvesvaraya Technological University (VTU), Belagavi. Its CSE, ECE and ISE programmes are accredited by the NBA, New Delhi, and the institute holds NAAC "A" grade accreditation.

This project was carried out as a **Major Project** under the **Department of Computer Science & Engineering (Artificial Intelligence & Machine Learning)** as part of the final-year B.E. curriculum. Progress was recorded weekly in the department's project logbook and reviewed by the project guide.

---

## 2. Problem Statement

Botanical traceability systems track plants (for example medicinal or protected species) from source to market, and blockchain-based systems make that record tamper-resistant. They still depend on a **physical tag**, such as a QR code or RFID chip, to link a real plant to its digital record. Tags can be removed, swapped or cloned, which opens the door to smuggling and mislabelling.

## 3. Our Idea

**Use the plant itself as the identifier.** A leaf's vein network is a natural pattern, much like a fingerprint. Instead of attaching a tag, we:

1. Photograph a leaf and extract its **vein pattern**.
2. Use that pattern as the **identifier** linking the physical plant to its digital twin.
3. On every later checkpoint, re-photograph the leaf and **verify** it matches the stored identity.
4. Use the same checkpoint photo to detect **severe degradation** (senescence, wilting, rot) so the token can be flagged as expired.

---

## 4. Pipeline Overview

The system is implemented in a Google Colab notebook (`plant_as_id_vein_extraction_stage4c.ipynb`). Most stages are classical computer vision with **no training required**.

| Stage | What it does | Method |
|---|---|---|
| **1. Vein extraction** | Separates leaf from background and extracts a thin vein skeleton from a normal photo | Otsu on saturation for leaf mask, CLAHE, Frangi ridge filter, skeletonization |
| **2. Minutiae extraction** | Finds vein **endpoints** and **branch points** on the skeleton, with spur pruning and leaf-border filtering | Neighbour-count analysis (fingerprint-style minutiae) |
| **3. Minutiae thinning** | Keeps only the strongest, well-separated minutiae, scored by local vein width, with endpoints and branch points thinned separately | Distance transform plus non-max suppression |
| **4. Minutiae matching** | Compares two minutiae sets using rotation- and scale-invariant local descriptors | Ratio test, mutual matching, RANSAC affine fit |
| **5. Vein-region SIFT matching** | Matches SIFT keypoints restricted to a dilated vein region | SIFT, ratio test, RANSAC |
| **6. Learned matching** | Matches pretrained learned features inside the vein region, more tolerant of lighting changes | SuperPoint + LightGlue via ONNX Runtime (no training, no GPU) |
| **7. Degradation check** | Compares a current photo against the baseline to flag token expiry | HSV colour-histogram drift plus leaf-area shrinkage |

---

## 5. Key Findings

- **Scale matters.** The first real comparison failed because the two photos differed by roughly 4.6x in zoom. Normalising descriptors by the square root of leaf area fixed this.
- **Skeleton minutiae alone were too weak.** On a rotated and cropped duplicate photo the skeleton matcher found only 2 of 17 inliers, while plain SIFT found 170 of 180 and recovered the exact 90 degree rotation. Two point types (endpoint and branch) discard most of the repeatable signal.
- **Vein-region SIFT** keeps the vein-based framing and is strongly rotation-invariant.
- **SuperPoint + LightGlue** is robust to illumination and contrast changes and to in-plane rotation up to about 30-40 degrees. It fails at 60-90 degree rotation and under heavy defocus blur. On a simulated recapture (15 degree rotation, slight scale change, light blur, brightness jitter) it reached an inlier fraction of about 0.92, and about 0.00 on a real different-leaf pair.
- **No single matcher wins everywhere.** Learned matching is better on lighting, SIFT is better on rotation.
- **Degradation check** was validated on real checkpoint photos and on a synthetically senesced leaf (hue/saturation/brightness shift plus mask shrinkage).

---

## 6. Setup and Usage

The notebook is designed for **Google Colab**.

```bash
pip install scikit-image opencv-python-headless onnxruntime
```

1. Open the notebook in Colab and run the setup cells.
2. Upload leaf photos (`.jpg` / `.jpeg` / `.png`) on a plain background.
3. Run stage 1 to generate, per photo, the leaf mask, vein mask, skeleton and overlay (saved to `/content/out`).
4. Run stages 2 and 3 to extract and thin the minutiae.
5. Run stage 4, 5 or 6 to compare two photos. Each returns candidates, inliers, an inlier fraction and the estimated transform.
6. For stage 6, run the setup cell once per session to download the pretrained SuperPoint and LightGlue ONNX models.
7. Run stage 7 with a baseline and a current photo to check degradation.

### Tunable parameters

| Parameter | Purpose |
|---|---|
| `prune_len` | Removes tiny fake spur branches from the skeleton |
| `border_margin` | Drops minutiae near the leaf outline |
| `strength_percentile`, `nms_radius` | Control how aggressively minutiae are thinned |

---

## 7. Limitations

- Match and expiry decision thresholds are **placeholders**. They have not been calibrated on a large labelled set of genuine and impostor pairs.
- Learned matching is sensitive to large in-plane rotation and strong defocus. Capture should be roughly overhead against a plain background.
- Testing covered a small number of leaves, not a large dataset.
- Leaves change over time (growth, ageing), so long-term identity stability needs further study.

## 8. Future Work

- Connect the vein identifier and expiry flag to an **actual blockchain** (identifier registration and on-chain token expiry). The current work covers the biometric and degradation side only.
- Build a labelled genuine/impostor dataset and calibrate thresholds properly.
- Add rotation normalisation before learned matching.
- Build a mobile capture workflow with guidance for consistent photos.

---

## 9. Acknowledgements

We thank our project guide, Prof. Deepika G S, the Department of CSE (AI & ML), and Sai Vidya Institute of Technology for their support and guidance. We also acknowledge the authors of SuperPoint, LightGlue and the LightGlue-ONNX project.

## 10. Repository Contents

```
plant_as_id_vein_extraction_stage4c.ipynb   # full pipeline, stages 1-7
README.md                                   # this file
```
