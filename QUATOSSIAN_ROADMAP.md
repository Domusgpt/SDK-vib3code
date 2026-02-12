# Quatossian Inscription Framework: The Master Plan

**Status:** Active Development
**Visionary:** Paul Phillips
**Architecture:** Quatossian Inscription (Quaternion-Gaussian Hybrid on E8 Lattice)

## 1. Overview
The **Quatossian Framework** supersedes the initial "Phillips Rendering System". While the Phillips System established the "Gaussian Flat" (Canonical View) baseline, the Quatossian framework introduces **Spin (Quaternions)**, **Lattice Indexing (E8)**, and **Wave Interference (Kirigami/Moiré)** to bridge digital rasterization with analog optical behavior.

## 2. Core Pillars

### The Quatossian Kernel (Cubit)
*   **Definition:** A radiance kernel combining a 3D Gaussian spatial extent with a normalized Quaternion rotation state ($q$).
*   **Property:** Unlike static splats, these kernels possess "Spin" and "Phase", allowing for view-dependent interference without heavy SH computation.

### The E8 Substrate
*   **Indexing:** Spatial organization using the E8 Gosset Lattice (8D) projected to 3D.
*   **Goal:** Optimal sphere packing density ($\rho \approx 0.74$ in 3D is surpassed by E8 density in 8D).
*   **Metric:** "Universal Coherence" geometry (Phi coherence plateau).

### Kirigami Modulation
*   **Mechanism:** Raycasting through virtual "Kirigami Masks" (binary/interference gratings).
*   **Effect:** Moiré patterns modulate the transparency and intensity of splats, simulating "shimmer", "neon glow", and "holographic edges".

## 3. Implementation Phases

### Phase 1-3: Phillips Rendering System (Completed)
*   [x] Gaussian Flat Rendering (Albedo Only).
*   [x] Plastic Ratio ($\rho \approx 1.3247$) Sampling.
*   [x] Parserator Integration (JSOH).

### Phase 4: The Quatossian Math (Current)
*   [ ] **Quaternions:** Implement robust `Quaternion` class in `src/math/Quaternion.js`.
*   [ ] **E8 Lattice:** Implement `src/math/E8.js` to handle 3D->8D projection and lattice node identification.
*   [ ] **Kernel Definition:** Define the `QuatossianKernel` data structure.

### Phase 5: Quatossian Renderer
*   [ ] **Shader Upgrade:** Upgrade `PhillipsRenderer` (or fork to `QuatossianRenderer`):
    *   **Attributes:** Add `a_rotation` (vec4 quaternion).
    *   **Uniforms:** Add `u_kirigamiShift` (vec2) and `u_moireFreq` (float).
    *   **Logic:** Apply quaternion rotation to vertex positions.
    *   **Interference:** Implement $I = I_0 \cdot (1 + \sin(\phi + \text{Moiré}))$ in fragment shader.

### Phase 6: E8 Spatial Indexing
*   [ ] **Sorting:** Replace linear sorting with E8-based Morton curve sorting.
*   [ ] **Packing:** Optimize point cloud distribution using E8 lattice sites.

### Phase 7: Holographic Edge Detection
*   [ ] **Pancharatnam–Berry Phase:** Implement optical spatial differentiation in a post-processing pass to detect edges via phase discontinuity.

## 4. Success Metrics
*   **Coherence:** Radiance field stability at "Diamond-Locked" coherence (0.83).
*   **Interference:** Visible, stable Moiré patterns on splat surfaces.
*   **Performance:** Maintain 60fps while calculating per-splat quaternion rotations.
