# Aerodynamic Validation References & Benchmark Data

- **Document:** `validation/references.md`
- **Scope:** Analytical, VLM (Phase 1), CFD (Phase 2), Surrogate (Phase 3), Lap Sim (Phase 5)

---

## 1. Case V1: Elliptic Planform (Prandtl Lifting-Line Theory)

- **Reference Citation:**
  Prandtl, L. (1918). *"Tragflügeltheorie"* (Lifting-Line Theory). *Nachrichten von der Gesellschaft der Wissenschaften zu Göttingen*.
- **Theoretical Formula:**
  For an elliptic circulation distribution:
  $$C_{Di} = \frac{C_L^2}{\pi \mathrm{AR}}$$
  $$e = 1.0000 \quad (\text{Oswald span efficiency factor})$$
  $$C_{L,\alpha} = \frac{2\pi}{1 + \frac{2}{\mathrm{AR}}} \quad [\text{rad}^{-1}]$$
- **Validation Target:**
  Oswald efficiency $e \in [0.985, 1.015]$ ($\le 1.5\%$ deviation from theoretical optimum).

---

## 2. Case V2: Lift-Curve Slope across Aspect Ratios (Helmbold Equation)

- **Reference Citation:**
  Helmbold, H. B. (1942). *"Der unverwundene Dreiecksflügel als Tragfläche fester Zirkulation"*. *Jahrbuch der Deutschen Luftfahrtforschung*.
- **Theoretical Formula:**
  Valid across low-to-high aspect ratios:
  $$\frac{dC_L}{d\alpha} = \frac{2\pi \mathrm{AR}}{\sqrt{\mathrm{AR}^2 + 4} + 2} \quad [\text{rad}^{-1}]$$
- **Aspect Ratios Evaluated:** $\mathrm{AR} \in \{2, 4, 6, 8, 12\}$.
- **Validation Target:** Relative error $|\Delta C_{L,\alpha}| / C_{L,\alpha} < 3.0\%$ across all aspect ratios.

---

## 3. Case V3: Grid Convergence Study (Richardson Extrapolation / GCI)

- **Reference Citation:**
  Roache, P. J. (1998). *"Verification and Validation in Computational Science and Engineering"*. Hermosa Publishers.
  Celik, I. B., et al. (2008). *"Procedure for estimation and reporting of uncertainty due to discretization in CFD applications"*. *ASME Journal of Fluids Engineering*, 130(7).
- **Mesh Hierarchy:**
  - Level 1 (Coarse): $10 \times 1$ panels
  - Level 2 (Medium): $20 \times 2$ panels
  - Level 3 (Fine): $40 \times 4$ panels
  - Level 4 (Very Fine): $80 \times 8$ panels
- **Validation Target:** Relative change between Level 3 and Level 4 $< 1.0\%$.

---

## 4. Case V4: Front Wing in Ground Effect (Zerihan & Zhang 2000)

- **Reference Citation:**
  Zerihan, J., & Zhang, X. (2000). *"Aerodynamics of a generic single-element Formula 1 front wing in ground effect"*. *AIAA Journal of Aircraft*, Vol. 37, No. 6, pp. 1045–1052.
  Katz, J. (1995). *"Race Car Aerodynamics: Designing for Speed"*. Robert Bentley Publishers.
- **Experimental Data Trend (Single-Element Inverted Wing at Racing Reynolds Number $Re \approx 10^6$):**
  - At $h/c = 1.00$: $C_L / C_{L,\infty} \approx 1.00$
  - At $h/c = 0.50$: $C_L / C_{L,\infty} \approx 1.08 \pm 0.02$
  - At $h/c = 0.30$: $C_L / C_{L,\infty} \approx 1.18 \pm 0.03$
  - At $h/c = 0.20$: $C_L / C_{L,\infty} \approx 1.28 \pm 0.04$
  - At $h/c = 0.15$: $C_L / C_{L,\infty} \approx 1.34 \pm 0.05$
- **Validation Target:** Monotonic lift augmentation down to $h/c = 0.15$; relative error $< 5.0\%$ vs published curve.
