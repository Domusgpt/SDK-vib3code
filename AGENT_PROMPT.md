# Agent Implementation Prompt: The Phillips Rendering System

**Goal:** You are an expert 3D Graphics Engineer tasked with implementing the core of the "Phillips Rendering System"—a novel, high-efficiency rendering architecture for 3D Gaussian Splats designed for AI semantic analysis.

**Context:**
We are pivoting from standard photorealistic rendering to a "Gaussian Flat" system. This system removes view-dependent effects (Spherical Harmonics) to create "Canonical Views" (albedo-only) that are deterministic and highly compressible (~17 bytes/splat). The system uses the **Plastic Ratio ($\rho \approx 1.3247$)** for low-discrepancy sampling to prevent moiré patterns in these flat visualizations.

**References:**
*   `EXPERIMENTAL_ROADMAP.md`: The master plan.
*   `src/`: Existing SDK structure.

---

## Your Mission

Please execute **Phase 1 (Math)** and **Phase 2 (Renderer)** of the roadmap.

### 1. Implement Mathematical Foundations
Create `src/math/Plastic.js` to handle the specific geometric constants required by this architecture.
*   **Constant:** `PLASTIC_CONSTANT = 1.324717957244746`
*   **Function:** `getPadovanSequence(n)`: Generate the integer sequence $P(n) = P(n-2) + P(n-3)$ starting with [1,1,1].
*   **Function:** `getPlasticSamplingPoint(index)`: Implement a low-discrepancy generator (similar to Golden Ratio sampling, but using $\rho$) to distribute splat centers or sampling points on a 2D plane.

### 2. Implement the Phillips Renderer
Create a new WebGL rendering system in `src/systems/PhillipsRenderer.js`.
*   **Architecture:** This should be a standalone renderer, similar to `HolographicCardGenerator` but stripped down.
*   **Vertex Shader:**
    *   Input: `a_position` (vec3), `a_scale` (float - driven by Plastic powers), `a_color` (u16/RGB565).
    *   Logic: Project 3D points to 2D screen space. Use `gl_PointSize` or simple quad expansion based on distance.
*   **Fragment Shader ("The Flat Shader"):**
    *   **Crucial:** Do NOT calculate lighting, specular, or SH.
    *   Output: `vec4(color, 1.0)`. Pure, flat albedo.
    *   Blend Mode: Standard Alpha Blending (for soft edges) OR Opaque (if implementing the specific "Hard Flat" style). *Start with standard alpha blending.*
*   **Uniforms:** Add a uniform `u_plasticScale` that modulates the global splat size based on the Plastic Ratio.

### 3. Integration
*   Expose this renderer in `src/export/ExportManager.js` (or a new entry point) so it can be initialized with a list of points.

---

## Constraints & Requirements
1.  **Code Style:** Match the existing ES6+ class structure found in `src/systems/`.
2.  **Performance:** The vertex shader must be extremely lightweight. No complex matrix math per-pixel if possible.
3.  **Naming:** Use explicit naming (e.g., `PhillipsRenderer`, `PlasticMath`).
4.  **Verification:** Write a simple unit test in `tests/math/Plastic.test.js` to verify the Padovan sequence generation matches known values ([1, 1, 1, 2, 2, 3, 4, 5...]).

**Output:**
Please generate the code for:
1.  `src/math/Plastic.js`
2.  `src/systems/PhillipsRenderer.js`
3.  `tests/math/Plastic.test.js`

Begin.
