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

### Phase 1: Mathematical Foundation
**Goal:** Establish the core constants and sampling logic.
*   [ ] Create `src/math/Plastic.js`:
    *   Define `PLASTIC_CONSTANT` ($\rho$).
    *   Implement `getPadovanSequence(n)`.
    *   Implement `getPlasticSamplingPoint(index)` (Low-discrepancy generator).
*   [ ] Create `src/math/SplatMath.js`:
    *   Logic to strip Spherical Harmonics (SH) from standard Gaussian data.
    *   Quantization logic (Float32 -> RGB565).

### Phase 2: The Phillips Renderer
**Goal:** A standalone WebGL renderer for "Gaussian Flats".
*   [ ] Create `src/systems/PhillipsRenderer.js`:
    *   **Vertex Shader:** Project 3D gaussians to 2D screen space (Billboard/Quad).
    *   **Fragment Shader:** "Flat" mode. Output pure Albedo. Ignore lighting/specular.
    *   **Uniforms:** `u_plasticScale` (controlled by $\rho$).
*   [ ] **Optimization:** Implement Instanced Rendering using the reduced byte structure (17 bytes/instance).

### Phase 3: Parserator Integration
**Goal:** Export data in a format consumable by `Parserator.com`.
*   [ ] Define "JSOH" (JSON Structured Object Hierarchy) schema for 3D scenes.
    *   Likely a JSON structure defining the "Semantic Massing" of the scene.
*   [ ] Update `ExportManager.js`:
    *   Add `exportToParserator()` method.
    *   Generate the "Canonical View" screenshot (PNG) + JSOH metadata.

### Phase 4: Verification
*   [ ] **Unit Tests:** Verify Padovan sequence and Plastic Ratio precision.
*   [ ] **Visual Tests:** Compare "Standard Splat" vs. "Phillips Flat" render.
*   [ ] **Performance:** Benchmark fps on low-end devices.

## 3. Success Metrics
*   Splat data size < 20 bytes/splat.
*   Render loop maintains 60fps with 100k splats on standard mobile hardware.
*   Output images are successfully processed by Parserator (simulated).
