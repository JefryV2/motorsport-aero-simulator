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
- **Validation Target:** Relative error $|\Delta C_{L,\alpha}| / C_{L,\alpha} < 3.5\%$ for $\mathrm{AR} \ge 4$. Helmbold is not a low-AR reference; at $\mathrm{AR}=2$ the comparison uses a 16% tolerance and the measured error is 15.4%, so it is reported separately rather than counted as a tight pass.

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

## 4. Case V4: Ground-Effect Trend (potential flow) and external reference (pending)

V4 is split into two checks so that a self-consistency test is never reported as
an external validation.

### 4a. Potential-flow image-system check (pass/fail)

- **Physics:** image-system ground effect for an inverted wing (Tuck 1975;
  Wieselsberger). The VLM implements this, so it is the appropriate
  self-consistency target.
- **Theoretical augmentation ratio** $DF(h)/DF(h\to\infty)$:
  - $h/c = 1.00$: 1.120
  - $h/c = 0.50$: 1.280
  - $h/c = 0.30$: 1.560
  - $h/c = 0.20$: 2.000
  - $h/c = 0.15$: 2.550
- **Validation Target:** monotonic augmentation; relative error $< 6.0\%$ against
  the image-system trend.

### 4b. External experimental reference (PENDING)

- **Reference Citation:**
  Zerihan, J., & Zhang, X. (2000). *"Aerodynamics of a generic single-element Formula 1 front wing in ground effect"*. *AIAA Journal of Aircraft*, Vol. 37, No. 6, pp. 1045–1052.
  Katz, J. (1995). *"Race Car Aerodynamics: Designing for Speed"*. Robert Bentley Publishers.
- **Indicative published trend** (not digitized, retained for gap reporting only):
  - $h/c = 1.00$: $C_L / C_{L,\infty} \approx 1.00$
  - $h/c = 0.50$: $C_L / C_{L,\infty} \approx 1.08 \pm 0.02$
  - $h/c = 0.30$: $C_L / C_{L,\infty} \approx 1.18 \pm 0.03$
  - $h/c = 0.20$: $C_L / C_{L,\infty} \approx 1.28 \pm 0.04$
  - $h/c = 0.15$: $C_L / C_{L,\infty} \approx 1.34 \pm 0.05$
- **Status:** PENDING. The modelled section is not the Zerihan & Zhang
  multi-element wing. The measured augmentation is 2.585 at $h/c=0.15$, about
  1.93x the published 1.34. Digitized data and a matching geometry are required
  before any external accuracy claim.

---

## 5. Case V5: CFD Mesh Independence Study (3-Level Richardson & GCI)

- **Reference Citation:**
  - Roache, P. J. (1998). *"Verification and Validation in Computational Science and Engineering"*. Hermosa Publishers.
  - Celik, I. B., et al. (2008). *"Procedure for estimation and reporting of uncertainty due to discretization in CFD applications"*. *ASME Journal of Fluids Engineering*, 130(7), 078001.
- **Methodology:**
  - 3 systematically refined grid levels: Coarse ($h_3$), Medium ($h_2$), Fine ($h_1$) with grid refinement ratio $r = h_{coarse} / h_{fine} \approx 1.5$.
  - Apparent order of convergence:
    $$p = \frac{1}{\ln(r_{21})} |\ln|\epsilon_{32} / \epsilon_{21}| + q(p)|$$
  - Extrapolated asymptotic value:
    $$\phi_{ext}^{21} = \frac{r_{21}^p \phi_1 - \phi_2}{r_{21}^p - 1}$$
  - Fine-grid convergence index ($\mathrm{GCI}_{fine}$ with safety factor $F_s = 1.25$):
    $$\mathrm{GCI}_{fine} = \frac{1.25 |e_{a}^{21}|}{r_{21}^p - 1}$$
- **Validation Target:** $\mathrm{GCI}_{fine} < 3.0\%$ on lift coefficient $C_L$; asymptotic range condition $R_{asymp} = \frac{\mathrm{GCI}_{23}}{r^p \mathrm{GCI}_{12}} \approx 1.0 \pm 0.1$.

---

## 6. Case V6: CFD Aerodynamic Benchmark (NACA 0012 & Inverted Wing)

- **Reference Citations:**
  - McCroskey, W. J. (1987). *"A critical assessment of wind tunnel results for the NACA 0012 airfoil"*. *AGARD-CP-429*.
  - Ladson, C. L. (1988). *"Effects of independent variation of Mach and Reynolds numbers on the low-speed aerodynamic characteristics of the NACA 0012 airfoil section"*. *NASA TM-4074*.
  - Zerihan, J., & Zhang, X. (2000). *"Aerodynamics of a generic single-element Formula 1 front wing in ground effect"*. *AIAA Journal of Aircraft*, 37(6): 1045–1052.
- **Reference Polar (NACA 0012 at $Re = 3.0 \times 10^6$, $\mathrm{Mach} \approx 0.15$):**
  - $\alpha = 0^\circ$: $C_L = 0.000 \pm 0.005$, $C_D = 0.0082 \pm 0.0004$
  - $\alpha = 4^\circ$: $C_L = 0.440 \pm 0.015$, $C_D = 0.0090 \pm 0.0005$
  - $\alpha = 8^\circ$: $C_L = 0.865 \pm 0.025$, $C_D = 0.0118 \pm 0.0008$
  - $\alpha = 10^\circ$: $C_L = 1.060 \pm 0.030$, $C_D = 0.0145 \pm 0.0010$
  - $\alpha = 12^\circ$: $C_L = 1.230 \pm 0.035$, $C_D = 0.0185 \pm 0.0015$
  - Stall angle: $\alpha_{stall} \approx 14.5^\circ - 16.0^\circ$, $C_{L,max} \approx 1.45 - 1.55$
- **Validation Target:**
  - Linear regime lift slope error $< 4.0\%$ vs Ladson/McCroskey.
  - Zero-lift drag $C_{D0}$ within $\pm 0.0015$ of experimental benchmark ($0.0082$).

