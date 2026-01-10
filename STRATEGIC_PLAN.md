# VIB3+ Strategic Development Plan

## Executive Summary

VIB3+ is a **4D geometry visualization engine** with unique market positioning. No competitor offers browser-based 4D visualization with trading card export, agentic control, and multi-system architecture.

---

## Part 1: Competitive Landscape

### Direct Competitors
| Product | Focus | Price | Gap VIB3+ Fills |
|---------|-------|-------|-----------------|
| **Shadertoy** | Shader coding | Free | No 4D abstraction, requires GLSL |
| **cables.gl** | Visual programming | Free | No 4D, complex node-based |
| **TouchDesigner** | Pro VJ/installations | $400+ | No 4D, desktop-only |
| **Art Blocks** | NFT generative art | Variable | No 4D, no immersive viewer |
| **fxhash** | NFT generative art | Variable | No 4D, no trading card focus |

### VIB3+ Unique Value Proposition
1. **Only browser-based 4D visualization engine** with full 6D rotation
2. **Trading card export** with immersive viewer (device tilt, card bending)
3. **Agentic control** via telemetry + MCP server
4. **Three visualization paradigms** (Quantum, Holographic, Faceted)
5. **24-geometry encoding** (8 base x 3 core types)

---

## Part 2: Current Architecture Issues

### Critical Bugs (IMMEDIATE)
1. **Faceted not rendering** - Canvas ID mismatch + initialization timing
2. **Geometry changes partial** - Missing `setVariant()` in some systems
3. **Rotation sliders inconsistent** - `updateParameters()` vs `updateParameter()`

### Architecture Debt
1. **3 competing engine patterns** - VIB3Engine, UnifiedEngine, VIB34DIntegratedEngine
2. **3 Polychora implementations** - None fully integrated
3. **Global state pollution** - `window.currentSystem`, `window.updateParameter`
4. **Geometry abstraction disconnected** - Generators exist but not used

---

## Part 3: Feature Gaps

### Missing Core Features
| Feature | Status | Priority |
|---------|--------|----------|
| Polytope archetype selector (1/3 core types) | Missing | HIGH |
| Randomize All button | Missing | HIGH |
| Randomize Left Menu only | Missing | MEDIUM |
| Randomize including system | Missing | MEDIUM |
| Collapsible mobile menus | Missing | HIGH |
| User reactivity telemetry | Partial | MEDIUM |
| Preset save/load UI | Backend exists, no UI | HIGH |
| Export modal with options | Missing | MEDIUM |

### Missing UX
- No feedback when parameters change
- No animation when switching systems
- No onboarding/tutorial
- No keyboard shortcuts

---

## Part 4: Implementation Plan

### Phase 1: Critical Bug Fixes (Week 1)

#### 1.1 Fix Faceted Rendering
```javascript
// Problem: FacetedSystem looks for 'content-canvas' but CanvasManager creates generic IDs
// Solution: Update FacetedSystem to use CanvasManager's ID pattern
```

#### 1.2 Add Polytope Core Type Selector
```javascript
// UI: 3 buttons above geometry grid
// [Base] [Hypersphere Core] [Hypertetrahedron Core]
// Formula: geometry = coreIndex * 8 + baseIndex
```

#### 1.3 Add Randomize Buttons
```javascript
// Randomize All: Both panels + system
// Randomize Params: Left panel only (rotation, visualization)
// Randomize Variant: Right panel only (geometry, color)
```

### Phase 2: Mobile-First UI (Week 2)

#### 2.1 Collapsible Panels
```css
/* Mobile portrait: Stack panels vertically, collapse by default */
@media (max-width: 768px) {
  .control-panel {
    position: fixed;
    bottom: 0;
    height: 40vh;
    transform: translateY(calc(100% - 48px));
  }
  .control-panel.expanded {
    transform: translateY(0);
  }
}
```

#### 2.2 Swipe Gestures
- Swipe up: Expand current panel
- Swipe left/right: Switch systems
- Pinch: Zoom/scale visualization

### Phase 3: Preset System (Week 3)

#### 3.1 Preset Data Structure
```javascript
{
  id: "uuid",
  name: "Cosmic Torus",
  system: "quantum",
  version: "1.0",
  created: "ISO-8601",
  parameters: { /* all params */ },
  metadata: {
    author: "username",
    tags: ["cosmic", "torus", "audio-reactive"],
    thumbnail: "base64-png"
  }
}
```

#### 3.2 Preset UI
- Save button with name prompt
- Load modal with grid view
- Cloud sync (optional Firebase)
- Share via URL parameters

### Phase 4: Agentic Integration (Week 4)

#### 4.1 Parameter Telemetry
```javascript
// Every parameter change emits event
eventEmitter.emit('parameter:change', {
  param: 'hue',
  value: 280,
  source: 'user' | 'agent' | 'audio' | 'random',
  timestamp: Date.now()
});
```

#### 4.2 Agent Commands
```javascript
// MCP tools for Claude/LLM control
{
  "set_parameter": { param: string, value: number },
  "randomize": { scope: "all" | "left" | "right" | "system" },
  "switch_system": { system: "quantum" | "faceted" | "holographic" },
  "export": { format: "png" | "json" | "html" },
  "save_preset": { name: string }
}
```

### Phase 5: Export & Sharing (Week 5)

#### 5.1 Export Modal
```
[Export Visualization]
  ├── [PNG] High-res image
  ├── [GIF] Animated loop (5 seconds)
  ├── [Trading Card] With overlay template
  ├── [JSON] Parameters only
  ├── [HTML] Standalone viewer
  └── [NFT] Mint to blockchain (future)
```

#### 5.2 Share URLs
```
https://vib3.app/?s=quantum&v=15&h=280&r=1.5,0.3,0,0,0,0
// s=system, v=variant, h=hue, r=rotations (comma-separated)
```

### Phase 6: Architecture Consolidation (Ongoing)

#### 6.1 Unified Engine Interface
```javascript
interface VisualizationSystem {
  initialize(): Promise<void>;
  setActive(active: boolean): void;
  setVariant(variant: number): void;
  updateParameters(params: object): void;
  getParameters(): object;
  destroy(): void;
}
```

#### 6.2 Remove Global State
- Pass context explicitly
- Use EventEmitter for cross-component communication
- Store state in single source (Redux-like pattern)

---

## Part 5: Monetization Strategy

### Tier 1: Free (Open Source Core)
- Full visualization engine
- Basic export (PNG, JSON)
- Local presets
- Community gallery

### Tier 2: Pro ($9.99/month)
- Cloud preset sync
- HD/4K export
- GIF/video export
- Custom watermark removal
- Priority support

### Tier 3: Enterprise ($49.99/month)
- API access for agentic integration
- White-label embedding
- Custom branding
- SSO authentication
- SLA support

### Tier 4: NFT Marketplace (Transaction Fee)
- 2.5% on NFT mints
- 1% on secondary sales
- Featured artist program
- Curated collections

---

## Part 6: Target Users

### Primary: Creative Coders
- **Need**: Visual experimentation without GLSL expertise
- **Value**: 4D abstraction, instant feedback, export to portfolio

### Secondary: Generative Artists
- **Need**: Unique visual style for NFT/art market
- **Value**: 24-geometry system, trading card format, blockchain integration

### Tertiary: Developers/Agents
- **Need**: Programmable visualization for dashboards, AI art
- **Value**: MCP integration, telemetry API, embeddable viewer

### Quaternary: Educators
- **Need**: Demonstrate 4D geometry concepts
- **Value**: Interactive sliders, multiple projection types, export for presentations

---

## Part 7: Success Metrics

### Technical KPIs
- Page load < 2 seconds
- 60 FPS on mid-range devices
- WASM core used for math (not JS fallback)
- All 3 systems render correctly

### User KPIs
- 5-minute session average
- 20% export rate (users who export something)
- 10% preset save rate
- 5% share rate

### Business KPIs
- 1000 MAU (Month 3)
- 5% Pro conversion (Month 6)
- First NFT mint (Month 9)
- Break-even on hosting (Month 12)

---

## Part 8: Immediate Action Items

### Today
1. [ ] Fix Faceted rendering (canvas ID issue)
2. [ ] Add Core Type selector (3 buttons)
3. [ ] Add Randomize buttons (3 variants)

### This Week
4. [ ] Collapsible mobile panels
5. [ ] Preset save/load UI
6. [ ] Parameter change telemetry

### Next Week
7. [ ] Export modal with options
8. [ ] Share URL generation
9. [ ] Agent command set

### This Month
10. [ ] Architecture consolidation
11. [ ] Performance optimization
12. [ ] Documentation site

---

## Appendix: File Structure Recommendation

```
vib3-plus-engine/
├── src/
│   ├── core/
│   │   ├── Engine.js          # Single unified engine
│   │   ├── Parameters.js      # Parameter management
│   │   └── EventBus.js        # Cross-component events
│   ├── systems/
│   │   ├── BaseSystem.js      # Interface all systems implement
│   │   ├── QuantumSystem.js
│   │   ├── FacetedSystem.js
│   │   └── HolographicSystem.js
│   ├── geometry/
│   │   ├── Polytope.js        # Base class
│   │   ├── CoreTypes.js       # Hypersphere, Hypertetrahedron wrappers
│   │   └── generators/        # 8 base geometries
│   ├── ui/
│   │   ├── ControlPanel.js    # Mobile-responsive panels
│   │   ├── PresetModal.js
│   │   └── ExportModal.js
│   ├── export/
│   │   ├── ExportService.js   # Unified export
│   │   └── TradingCard.js
│   ├── agent/
│   │   ├── Telemetry.js       # Event tracking
│   │   └── MCPServer.js       # Agent integration
│   └── wasm/
│       └── Vib3Core.js        # C++ math bindings
├── docs/                       # GitHub Pages deployment
├── tests/                      # Vitest test suite
└── STRATEGIC_PLAN.md          # This document
```

---

*Generated: 2026-01-10*
*Author: Claude (with human direction)*
*Version: 1.0*
