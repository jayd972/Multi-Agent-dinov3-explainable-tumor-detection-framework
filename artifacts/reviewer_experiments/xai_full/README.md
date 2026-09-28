# XAI full output

Smoke run (one official-test image per BRISC class, deterministic first filename):

- `smoke/` maps, figures (PNG+PDF, 300 DPI), sanity, reports
- `smoke_per_image.csv` / `smoke_overall.csv` / `smoke_overall.tex`
- `threshold_selection.json` — default 0.5; **test masks were not used**
- `REVIEWER_EVIDENCE_CHECKLIST.md`

Official BRISC segmentation masks are **not present**. Localization Dice/IoU are `MASK_UNAVAILABLE` and were not invented.

Full official-test evaluation: `python -m xai evaluate-test --primary-only`
