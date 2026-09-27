"""Unit tests for the comprehensive evaluation pipeline, mocking all API calls."""
from __future__ import annotations

import json
import os
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

from experiments.run_comprehensive_evaluation import (
    LLMJudge,
    bootstrap_ci,
    run_evaluation,
)


@pytest.fixture
def mock_manifest_df():
    data = []
    classes = ["glioma", "meningioma", "no_tumor", "pituitary"]
    for i, cls in enumerate(classes):
        for rep in range(3):  # 3 of each to make it quick
            data.append({
                "image_id": f"case_{cls}_{rep}",
                "true_label": cls,
                "predicted_class": f"{cls}_tumor" if cls != "no_tumor" else "no_tumor",
                "confidence": 0.95,
                "path": f"dummy_path/Testing/{cls}/case_{cls}_{rep}.jpg",
                "attn_rollout": f"dummy_path/case_{cls}_{rep}/attn_rollout.png",
                "grad_saliency": f"dummy_path/case_{cls}_{rep}/grad_saliency.png",
            })
    return pd.DataFrame(data)


@pytest.fixture
def mock_openai_client():
    client = MagicMock()
    mock_choice = MagicMock()
    # Mock LLM Judge response JSON
    mock_choice.message.content = json.dumps({
        "extracted_diagnosis": "glioma_tumor",
        "diagnosis_correctness": True,
        "prediction_consistency": True,
        "unsupported_claims": ["edema", "size"],
        "contradictions": [],
        "hallucination_severity": 2,
        "clarity_score": 4,
        "structure_score": 4,
        "reasoning": "Consistent with glioma pred."
    })
    
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 100
    mock_usage.completion_tokens = 50
    
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage = mock_usage
    
    client.chat.completions.create.return_value = mock_resp
    return client


def test_bootstrap_ci():
    y_true = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"] * 10
    y_pred = ["glioma_tumor", "meningioma_tumor", "no_tumor", "no_tumor"] * 10  # some mismatches
    
    from sklearn.metrics import accuracy_score
    low, high = bootstrap_ci(y_true, y_pred, accuracy_score, num_bootstraps=100, seed=42)
    assert 0.0 <= low <= high <= 1.0


def test_llm_judge_caching_and_cost(mock_openai_client, tmp_path, monkeypatch):
    cache_path = tmp_path / "judge_cache.json"
    monkeypatch.setattr("experiments.run_comprehensive_evaluation.JUDGE_CACHE_PATH", cache_path)
    
    judge = LLMJudge(mock_openai_client, "gpt-5.6-luna")
    assert judge.model == "gpt-5.6-luna"
    
    # Query judge
    res = judge.query_judge("case1", "report text", "glioma_tumor", "glioma_tumor", 0.95, True)
    assert res["extracted_diagnosis"] == "glioma_tumor"
    assert judge.total_prompt_tokens == 100
    assert judge.total_completion_tokens == 50
    assert judge.get_total_cost() > 0.0
    
    # Second query should hit cache (so tokens won't increase)
    res2 = judge.query_judge("case1", "report text", "glioma_tumor", "glioma_tumor", 0.95, True)
    assert res2["extracted_diagnosis"] == "glioma_tumor"
    assert judge.total_prompt_tokens == 100  # no change
    
    assert cache_path.exists()


@patch("experiments.run_comprehensive_evaluation.pd.read_csv")
@patch("experiments.run_comprehensive_evaluation.find_mask_for_image")
@patch("experiments.run_comprehensive_evaluation.generate_report")
@patch("experiments.run_comprehensive_evaluation.load_openai_client")
@patch("experiments.run_comprehensive_evaluation.LLMJudge")
@patch("experiments.run_comprehensive_evaluation.plt.savefig")
def test_run_evaluation_pipeline(
    mock_savefig,
    mock_judge_cls,
    mock_load_client,
    mock_gen_report,
    mock_find_mask,
    mock_read_csv,
    mock_manifest_df,
    tmp_path,
    monkeypatch
):
    # Setup mocks
    mock_read_csv.return_value = mock_manifest_df
    mock_find_mask.return_value = None
    
    mock_gen_report.return_value = {
        "report": {"model_prediction": "glioma_tumor", "explanation_summary": "No edema."}
    }
    
    client = MagicMock()
    mock_load_client.return_value = (client, "gpt-5.6-luna")
    
    # Mock LLM Judge instance
    judge_inst = MagicMock()
    judge_inst.model = "gpt-5.6-luna"
    judge_inst.get_total_cost.return_value = 0.05
    judge_inst.query_judge.return_value = {
        "extracted_diagnosis": "glioma_tumor",
        "diagnosis_correctness": True,
        "prediction_consistency": True,
        "unsupported_claims": [],
        "contradictions": [],
        "hallucination_severity": 1,
        "clarity_score": 5,
        "structure_score": 5,
        "reasoning": "Consistent"
    }
    mock_judge_cls.return_value = judge_inst
    
    # Re-route outputs directory to tmp_path
    monkeypatch.setattr("experiments.run_comprehensive_evaluation.OUT_DIR", tmp_path)
    monkeypatch.setattr("experiments.run_comprehensive_evaluation.JUDGE_CACHE_PATH", tmp_path / "cache.json")
    
    # Run evaluation (sampling will sample 25 but since we only have 12 mock images, it samples all 12)
    run_evaluation()
    
    # Verify outputs saved
    assert (tmp_path / "comprehensive_eval_cases.csv").exists()
    assert (tmp_path / "comprehensive_eval_summary.md").exists()
