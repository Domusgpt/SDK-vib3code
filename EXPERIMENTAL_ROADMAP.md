# Experimental Roadmap: The Phillips Rendering System

**Status:** Draft
**Owner:** Paul Phillips (Clear Seas Solutions)
**Objective:** Implement the "Gaussian Flat Shader System" to enable high-performance, semantic-ready 3D visualizations for AI agents (Parserator).

## 1. Vision & Core Concepts
The **Phillips Rendering System** is a divergence from standard 3D Gaussian Splatting. Instead of photorealistic, view-dependent rendering (using Spherical Harmonics), it prioritizes:
*   **Data Density:** Reducing 224 bytes/splat → ~17 bytes/splat (92% reduction).
*   **Semantic Clarity:** "Canonical Views" (Albedo-only) that are deterministic and noise-free for AI analysis.
*   **Performance:** 60-120 FPS on mobile via WebXR.

### Key Mathematical Pillars
*   **Plastic Ratio ($\rho \approx 1.3247$):** Used for low-discrepancy sampling and splat distribution to prevent moiré patterns in flat shading.
*   **Padovan Sequence:** Integer approximation series for the Plastic Ratio.

## 2. Implementation Phases

### Phase 1: Mathematical Foundation (Done)
**Goal:** Establish the core constants and sampling logic.
*   [x] Create `src/math/Plastic.js`:
    *   Define `PLASTIC_CONSTANT` ($\rho$).
    *   Implement `getPadovanSequence(n)`.
    *   Implement `getPlasticSamplingPoint(index)` (Low-discrepancy generator).

### Phase 2: The Phillips Renderer (Done)
**Goal:** A standalone WebGL renderer for "Gaussian Flats".
*   [x] Create `src/systems/PhillipsRenderer.js`:
    *   **Vertex Shader:** Project 3D gaussians to 2D screen space (Billboard/Quad).
    *   **Fragment Shader:** "Flat" mode. Output pure Albedo. Ignore lighting/specular.
    *   **Uniforms:** `u_plasticScale` (controlled by $\rho$).

### Phase 3: Parserator Integration (Done)
**Goal:** Export data in a format consumable by `Parserator.com`.
*   [x] Define "JSOH" (JSON Structured Object Hierarchy) schema for 3D scenes (`src/export/JSOH.js`).
*   [x] Update `ExportManager.js`:
    *   Add `exportToParserator()` method.
    *   Generate the "Canonical View" screenshot (PNG) + JSOH metadata.

### Phase 5: Interactive Visualization (In Progress)
**Goal:** Create a live browser demo to verify the system visually.
*   [ ] Create `docs/phillips-demo.html`.
*   [ ] Implement interactive controls for Plastic Scale and Cloud Density.

### Phase 4: Verification
*   [ ] **Unit Tests:** Verify Padovan sequence and Plastic Ratio precision.
*   [ ] **Visual Tests:** Compare "Standard Splat" vs. "Phillips Flat" render.
*   [ ] **Performance:** Benchmark fps on low-end devices.

## 3. Success Metrics
*   Splat data size < 20 bytes/splat.
*   Render loop maintains 60fps with 100k splats on standard mobile hardware.
*   Output images are successfully processed by Parserator (simulated).
