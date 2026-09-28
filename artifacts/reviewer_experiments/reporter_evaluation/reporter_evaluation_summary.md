# LLM Reporter Evaluation Summary

**Date:** 2026-08-27T03:15:21.060222

## Key Finding: Reporter Produces Unsupported Clinical Claims

The existing LLM-generated report contains fabricated anatomical and clinical
information. The LLM does NOT receive the raw MRI image, yet makes specific
claims about tumor location, tissue characteristics, and morphology.

### Evidence from Existing Report

- **Unsupported claims detected:** 4
- **Anatomical references without anatomical input:** 2

### Specific Unsupported Claims in image(1)_report.txt

- `edema`
- `infiltrat`
- `parietal lobe`
- `mm`

### What the LLM Actually Receives

1. Predicted class name and confidence probability
2. Top-k class probabilities
3. DeepDream visualization (NOT raw MRI - this is a feature visualization)
4. Occlusion sensitivity map (shows model sensitivity, NOT tumor boundaries)
5. Probability distribution chart

### What the LLM Does NOT Receive

- The raw MRI image
- Any validated anatomical segmentation
- Any tissue characterization data
- Any measurements
- Any clinical context

### Implications for the Paper

1. The reporter CANNOT make anatomically specific claims
2. Any such claims in the generated reports are hallucinations
3. The prompt actively ENCOURAGES hallucination by asking for anatomical specificity
4. This must be acknowledged as a limitation
5. The reporter's role should be reframed as 'structured summary of model predictions'
   rather than 'radiological interpretation'

### Reporter Configuration Conflict

| Parameter | run_pipeline.py | reporter_server.py |
|-----------|----------------|-------------------|
| Temperature | 0.3 | 0.2 |
| System prompt | Detailed radiology persona | Brief accuracy instruction |
| Input images | Dream + Probs + Occlusion | Attention + Gradient + Dream + Probs + Lucent |

### Evaluation Status

| Evaluation | Status |
|-----------|--------|
| Existing report analysis | COMPLETED |
| Factual consistency check | COMPLETED (rule-based) |
| Unsupported claim detection | COMPLETED (keyword-based) |
| Multi-generation repeatability | REQUIRES API KEY |
| Ablation (input modalities) | REQUIRES API KEY |
| LLM-as-judge (secondary) | REQUIRES API KEY |

### Hallucination Rate (Existing Evidence)

Based on the single available report:

- Unsupported clinical claims: 4
- Unsupported anatomical references: 2

**LIMITATION:** This is based on a single report (n=1). A proper evaluation
requires generating reports for a balanced sample across all classes with
multiple seeds. This requires API access and budget approval.