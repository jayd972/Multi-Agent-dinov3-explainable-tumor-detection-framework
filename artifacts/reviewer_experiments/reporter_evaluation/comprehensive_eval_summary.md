# DINOv3 Comprehensive Report Evaluation Summary

## Performance Metrics

| Metric | Value | 95% Confidence Interval |
| :--- | :---: | :---: |
| **Accuracy** | 0.9900 | [0.9700, 1.0000] |
| **Balanced Accuracy** | 0.9900 | [0.9643, 1.0000] |
| **Macro Precision** | 0.9904 | - |
| **Macro Recall** | 0.9900 | - |
| **Macro F1** | 0.9900 | [0.9644, 1.0000] |

## Per-Class Performance

| Class | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| glioma | 1.0000 | 0.9600 | 0.9796 |
| meningioma | 0.9615 | 1.0000 | 0.9804 |
| no | 1.0000 | 1.0000 | 1.0000 |
| pituitary | 1.0000 | 1.0000 | 1.0000 |

## LLM-as-a-Judge Quality Metrics

- **Default Judge Model:** `gpt-4o`
- **Average Unsupported Claims per Report:** `0.44`
- **Average Hallucination Severity (1-5):** `1.34/5`
- **Average Report Clarity Score (1-5):** `4.52/5`
- **Average Report Structure Score (1-5):** `4.53/5`
- **Total OpenAI Token Cost:** `$0.3417`
