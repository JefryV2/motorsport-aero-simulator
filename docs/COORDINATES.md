# Aerodynamic & 3D Coordinate System Conventions

- **Document:** `docs/COORDINATES.md`
- **Phase:** Phase 0 (Task P0.7)
- **Standard:** ISO 8855 Vehicle Dynamics / Three.js WebGL Mapping

---

## 1. Physical Vehicle Coordinate System (ISO 8855)

The physical and computational aerodynamic models (VLM, OpenFOAM CFD, vehicle lap simulation) adhere to the right-handed **ISO 8855** vehicle-fixed coordinate frame origin at the center of the front axle projected onto ground level:

$$\begin{aligned}
+X &\quad\text{Longitudinal forward (direction of travel)} \\
+Y &\quad\text{Lateral left (driver's left)} \\
+Z &\quad\text{Vertical upward (normal to ground plane)}
\end{aligned}$$

### Force & Moment Sign Conventions

$$\begin{aligned}
\text{Downforce } (DF) &= -F_z \quad (\text{positive down onto track}) \\
\text{Lift } (L) &= +F_z = -DF \\
\text{Drag } (D) &= -F_x \quad (\text{positive opposing forward motion}) \\
\text{Sideforce } (S) &= +F_y \\
\text{Pitching moment } (M_y) &= \text{positive nose-up} \\
\text{Center of Pressure } (\text{CoP}_x) &= \text{longitudinal distance rearward from front axle: } x_{\text{CoP}} = \frac{-M_y}{DF}
\end{aligned}$$

---

## 2. Three.js Scene Coordinate System

Three.js uses a right-handed WebGL screen coordinate frame:

$$\begin{aligned}
+X_{\text{3D}} &\quad\text{Lateral right} \\
+Y_{\text{3D}} &\quad\text{Vertical upward (world up)} \\
+Z_{\text{3D}} &\quad\text{Longitudinal rearward (depth towards camera in default view)}
\end{aligned}$$

### Transformation from Vehicle Frame (ISO) to Three.js

$$\begin{pmatrix} X_{\text{3D}} \\ Y_{\text{3D}} \\ Z_{\text{3D}} \end{pmatrix} = \begin{pmatrix} 0 & -1 & 0 \\ 0 & 0 & 1 \\ -1 & 0 & 0 \end{pmatrix} \begin{pmatrix} X_{\text{ISO}} \\ Y_{\text{ISO}} \\ Z_{\text{ISO}} \end{pmatrix} = \begin{pmatrix} -Y_{\text{ISO}} \\ Z_{\text{ISO}} \\ -X_{\text{ISO}} \end{pmatrix}$$

---

## 3. Surface Mapping Frame (`aero_field.py`)

For synthetic surface field evaluation and aerodynamic texture projection on uploaded 3D meshes, vertices are mapped into a dimensionless unit cube $[0, 1]^3$ defined by the mesh axis-aligned bounding box (AABB):

$$\begin{aligned}
u_x &= \frac{x - x_{\min}}{x_{\max} - x_{\min}} \in [0, 1] \quad (\text{front } 0 \to \text{rear } 1) \\
u_y &= \frac{y - y_{\min}}{y_{\max} - y_{\min}} \in [0, 1] \quad (\text{left } 0 \to \text{right } 1) \\
u_z &= \frac{z - z_{\min}}{z_{\max} - z_{\min}} \in [0, 1] \quad (\text{ground } 0 \to \text{top } 1)
\end{aligned}$$

### Regional Component Masking Bounds:
- **Front Wing:** $u_x \le 0.18$, $u_z \le 0.35$
- **Rear Wing:** $u_x \ge 0.78$, $u_z \ge 0.55$
- **Diffuser / Underbody:** $0.62 \le u_x \le 1.0$, $u_z \le 0.22$
- **Main Chassis:** Remainder of the volume envelope
