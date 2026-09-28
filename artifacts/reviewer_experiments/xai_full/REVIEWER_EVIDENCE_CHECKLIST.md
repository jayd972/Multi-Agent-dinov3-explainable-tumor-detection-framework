# Reviewer evidence checklist (XAI, Lucent/Deep Dream, LLM reports)

Status values: `IMPLEMENTED` (code + tests), `EXECUTED` (this run wrote the result file), `BLOCKED` (missing data, not invented), `REMAINING` (needs a longer official-test or real-API run).

| Reviewer concern | Implementation | Result / figure | Status | Remaining limitation |
|------------------|----------------|-----------------|--------|----------------------|
| XAI only on selected images | `xai/evaluate.py`, `python -m xai evaluate-test` | `artifacts/reviewer_experiments/xai_full/*_per_image.csv` | IMPLEMENTED | Full official-test metrics require the evaluate-test command. Smoke uses one deterministic image per class. |
| Missing Grad-CAM | `xai/methods/gradcam.py` | `maps/*/gradcam/` | IMPLEMENTED | Uses a configurable transformer block (default last). |
| Missing attention rollout | `xai/methods/attention.py` | `maps/*/attention_rollout/`, `figures/attention.pdf` | IMPLEMENTED | Rollout is not class-specific and is not causal proof. |
| Unclear Lucent layer/neuron | `xai/lucent_dream.py` | `feature_viz/lucent/*/*.json` | IMPLEMENTED | Hidden-neuron images are labeled as hidden, not class images. |
| Blurred Lucent / Deep Dream | metadata `limitation` fields | same JSON + PNG | IMPLEMENTED | Blur is documented as a method limitation, not a medical feature. |
| No localization measurements | `xai/metrics.py` Dice/IoU/pointing/mass/insertion/deletion | `*_per_image.csv`, `*_per_class.csv` | IMPLEMENTED / BLOCKED if masks absent | Official BRISC masks were not on disk at implementation time. Metrics are `MASK_UNAVAILABLE` until `BRISC_MASK_ROOT` is set. Values are not invented. |
| Unclear LLM input modality | `reports/generator.py` request manifest | `reports_full/**/request_manifest.json` | IMPLEMENTED | Manifest records whether MRI / XAI files were in the actual payload. |
| Missing complete LLM prompts | `reports/prompts.py` | `system_prompt.txt`, `user_prompt.txt` | IMPLEMENTED | Appendix-ready full text. |
| Missing generated report examples | one case per BRISC class in smoke | `xai_full/smoke/reports/` | IMPLEMENTED | Smoke examples are `offline_template_not_llm` unless `--real` is used. |
| Unsupported anatomical statements | `reports/factual.py` + system prompt | `factual_consistency.json` | IMPLEMENTED | Programmatic checks; clinician review is not available (`reports/clinician_form.md`). |
| No report factual consistency eval | `reports/factual.py`, `python -m xai evaluate-reports` | `report_evaluation.csv` | IMPLEMENTED | BLEU/ROUGE are not used. The generating model is not the judge. |
| Small unreadable figures | `xai/figures.py` 300 DPI + PDF, colorbars, labels | `figures/*.png`, `*.pdf` | IMPLEMENTED | Crowded panels are split (main vs attention vs galleries). |
| Missing reproducibility details | checkpoint SHA256, configs, seeds, manifests | `config.json`, `request_manifest.json`, `smoke_summary.json` | IMPLEMENTED | Credentials are never saved. |
| AttnLRP requested | Fallback documented in `xai/methods/chefer.py` | `ATTLRP_FALLBACK_REASON` | IMPLEMENTED | Named `chefer_attribution` everywhere. Not renamed attention rollout. |
| Salience vs saliency | Consolidated to `vanilla_gradient` | method config `not_a_second_major_method` | IMPLEMENTED | Supplementary baseline only. |
| Threshold on test | `select_threshold_on_validation` | `threshold_selection.json` | IMPLEMENTED | Default 0.5 if no val masks. Test masks are not used for selection. |
| Sanity tests | `xai/sanity.py`, `run_sanity_suite` | `sanity_tests.json` | IMPLEMENTED | Cross-seed map stability is `INSUFFICIENT EVIDENCE` (one checkpoint). |

## Output tree

```text
artifacts/reviewer_experiments/xai_full/
  maps/<image_id>/<method>/{attribution.npy,normalized.npy,overlay.png,heatmap.png,config.json}
  {split}_per_image.csv
  {split}_per_class.csv
  {split}_overall.csv
  {split}_overall.tex
  threshold_selection.json
  occlusion_settings.json
  sanity_tests.json
  figures/{main_xai,attention,*.png,*.pdf}
artifacts/reviewer_experiments/feature_viz/{lucent,deepdream}/
artifacts/reviewer_experiments/reports_full/{structured,multimodal}/<case>/
```
