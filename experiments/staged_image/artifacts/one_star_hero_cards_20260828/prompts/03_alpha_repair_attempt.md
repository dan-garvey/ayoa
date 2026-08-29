# Built-in image edit prompt: rejected alpha repair

The most recent generated Obsidian Orrery image was the edit target.

```text
Use case: background-extraction
Asset type: reusable transparent game UI frame overlay
Primary request: Remove only the gray-and-white checkerboard background from the most recent generated ornate frame and replace it with genuine alpha transparency.
Input images: Image 1 is the edit target, the Obsidian Orrery frame candidate.
Constraints: preserve the frame's metalwork, geometry, exact composition, empty circular top bezel, and empty bottom plaque unchanged; keep the complete outer frame visible; make every checkerboard pixel outside the frame and within the portrait aperture, top bezel center, and bottom plaque center genuinely transparent; output RGBA PNG; no new shadow, color, fill, symbol, text, portrait, person, face, star, logo, watermark, or ornament.
Avoid: faux checkerboard; white/gray backing; eroding dark metal edges; filling any aperture; restyling.
```

The tool returned another RGB PNG with baked checkerboard pixels, so the edit
was rejected. The review overlays instead use frozen BiRefNet masks and
deterministic RGBA construction.
