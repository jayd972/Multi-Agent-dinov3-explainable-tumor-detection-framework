from utils.io import read_json
from utils.paths import FEATS, MODELS, METRICS, EXPLAIN, O1, DREAM, REPORTS
from agents.modeler_client import extract_features, train_probe, eval_probe, finetune_model
from agents.explainer_client import lucent_prototypes, o2_grid, deepdream, explain_image
from agents.reporter_client import explain_results

from pathlib import Path
import os
from utils.secrets import load_secrets
load_secrets()

# Load API and server config
api_cfg = read_json("config/api_config.json")

MODELER = os.getenv("MODELER_URL", api_cfg["servers"]["modeler"]["base_url"])
EXPLAINER = os.getenv("EXPLAINER_URL", api_cfg["servers"]["explainer"]["base_url"])
REPORTER = os.getenv("REPORTER_URL", api_cfg["servers"]["reporter"]["base_url"])

# Set tokens globally
os.environ["OPENAI_API_KEY"] = api_cfg["api_keys"]["openai"]["key"]
os.environ["HUGGINGFACE_HUB_TOKEN"] = api_cfg["api_keys"]["huggingface"]["token"]

MODELER = "http://127.0.0.1:8001"
EXPLAINER = "http://127.0.0.1:8002"
REPORTER = "http://127.0.0.1:8003"

def run_pipeline(cfg: dict):
    # 0) Optional fine-tuning step
    finetune_out = None
    if cfg.get("do_finetune", False):
        print("Starting fine-tuning on training images...")
        finetune_out = finetune_model(
            MODELER,
            data_root=cfg["data_root"],
            model_id=cfg["model_id"],
            size=cfg.get("img_size", 224),
            epochs=cfg.get("epochs", 3),
            lr=cfg.get("lr", 1e-5),
            out_dir=str(MODELS / "finetuned")
        )
        print("finetune:", finetune_out)
    else:
        print("Skipping fine-tuning (do_finetune=false)")

    # 1) Feature extraction (skip if features exist and model_id matches)
    train_feats = FEATS / "train.npz"
    test_feats = FEATS / "test.npz"
    
    if train_feats.exists() and test_feats.exists():
        # Check if model_id matches
        try:
            import numpy as np
            train_data = np.load(train_feats, allow_pickle=True)
            if "model_id" in train_data:
                feat_model_id = train_data["model_id"].tolist()[0]
                if feat_model_id == cfg["model_id"]:
                    print(f"✅ Using existing features (model_id: {feat_model_id})")
                    f_out = {
                        "ok": True,
                        "train": str(train_feats),
                        "test": str(test_feats),
                        "classes": train_data["class_names"].tolist() if "class_names" in train_data else []
                    }
                    print("Using existing features:", f_out)
                else:
                    print(f"⚠️  Model ID mismatch - re-extracting features")
                    f_out = extract_features(
                        MODELER,
                        data_root=cfg["data_root"],
                        model_id=cfg["model_id"],
                        size=cfg["img_size"],
                        out_train_npz=str(train_feats),
                        out_test_npz=str(test_feats),
                    )
                    print("extract:", f_out)
            else:
                print("⚠️  Features missing model_id - re-extracting")
                f_out = extract_features(
                    MODELER,
                    data_root=cfg["data_root"],
                    model_id=cfg["model_id"],
                    size=cfg["img_size"],
                    out_train_npz=str(train_feats),
                    out_test_npz=str(test_feats),
                )
                print("extract:", f_out)
        except Exception as e:
            print(f"⚠️  Error checking features: {e} - re-extracting")
            f_out = extract_features(
                MODELER,
                data_root=cfg["data_root"],
                model_id=cfg["model_id"],
                size=cfg["img_size"],
                out_train_npz=str(train_feats),
                out_test_npz=str(test_feats),
            )
            print("extract:", f_out)
    else:
        f_out = extract_features(
            MODELER,
            data_root=cfg["data_root"],
            model_id=cfg["model_id"],
            size=cfg["img_size"],
            out_train_npz=str(train_feats),
            out_test_npz=str(test_feats),
        )
        print("extract:", f_out)

    # 2) Check for existing model (clf.pth or clf.pkl)
    clf_pth = MODELS / "clf.pth"
    clf_pkl = MODELS / "clf.pkl"
    
    if clf_pth.exists():
        print(f"✅ Using existing PyTorch model: {clf_pth}")
        # Default class names for brain tumor classification
        class_names = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
        p_out = {
            "ok": True,
            "pkl": str(clf_pth),  # Use .pth path
            "classes": class_names
        }
        print("Using existing PyTorch model:", p_out)
    elif clf_pkl.exists():
        print(f"✅ Using existing sklearn classifier: {clf_pkl}")
        import joblib
        meta = joblib.load(clf_pkl)
        p_out = {
            "ok": True,
            "pkl": str(clf_pkl),
            "classes": meta.get("class_names", [])
        }
        print("Using existing sklearn model:", p_out)
    else:
        p_out = train_probe(MODELER, train_npz=f_out["train"], out_pkl=str(clf_pkl))
        print("train:", p_out)

    # 3) Probe evaluation
    m_out = eval_probe(MODELER, test_npz=f_out["test"], pkl=p_out["pkl"], out_json=str(METRICS / "metrics.json"))
    print("eval:", m_out)

    # 4) Optional Lucent O1 and O2 visualizations
    if cfg.get("do_o1_o2", True):
        lp_out = lucent_prototypes(
            EXPLAINER,
            pkl=p_out["pkl"],
            layer=cfg["layer"],
            n_prototypes=cfg["n_prototypes"],
            iters=cfg["iters"],
            out_dir=str(O1),
        )
        print("lucent_prototypes:", lp_out)

        grid_out = o2_grid(
            EXPLAINER,
            pkl=p_out["pkl"],
            layer=cfg["layer"],
            classes=p_out["classes"],
            n_prototypes=cfg["n_prototypes"],
            iters=cfg["iters"],
            out_png=str(EXPLAIN / "o2_grid.png"),
        )
        print("o2_grid:", grid_out)
    else:
        lp_out, grid_out = None, None

    # 5) Single image explanations
    single_cfg = cfg.get("single_image")
    explain_out = None
    report_out = None
    if single_cfg and single_cfg.get("image_path"):
        img_path = single_cfg["image_path"]
        out_dir = EXPLAIN / "by_image" / Path(img_path).stem
        out = explain_image(
            EXPLAINER,
            pkl=p_out["pkl"],
            image_path=img_path,
            out_dir=str(out_dir),
            layer_suffix=single_cfg.get("layer_suffix", cfg["deepdream"]["layer_suffix"]),
            steps=single_cfg.get("steps", cfg["deepdream"][cfg["deepdream"]["preset"]]["steps"]),
            lr=single_cfg.get("lr", cfg["deepdream"][cfg["deepdream"]["preset"]]["lr"]),
            tv_weight=single_cfg.get("tv_weight", cfg["deepdream"][cfg["deepdream"]["preset"]]["tv_weight"]),
            octaves=single_cfg.get("octaves", cfg["deepdream"][cfg["deepdream"]["preset"]]["octaves"]),
            num_octaves=single_cfg.get("num_octaves", cfg["deepdream"][cfg["deepdream"]["preset"]]["num_octaves"]),
            octave_scale=single_cfg.get("octave_scale", cfg["deepdream"][cfg["deepdream"]["preset"]]["octave_scale"]),
            jitter=single_cfg.get("jitter", cfg["deepdream"][cfg["deepdream"]["preset"]]["jitter"]),
            do_attention=single_cfg.get("do_attention", True),
            do_gradients=single_cfg.get("do_gradients", True),
            do_deepdream=single_cfg.get("do_deepdream", True),
            do_probs=single_cfg.get("do_probs", True),
            do_lucent=single_cfg.get("do_lucent", True),
            do_occlusion=single_cfg.get("do_occlusion", True),
            lucent_iters=single_cfg.get("lucent_iters", 256),
            occlusion_patch_size=single_cfg.get("occlusion_patch_size", 16),
            occlusion_stride=single_cfg.get("occlusion_stride", 8)
        )
        explain_out = out
        print("explain_image:", out)

        # 6) Optional LLM-based report
        if cfg.get("generate_report", True):
            rep_dir = REPORTS
            rep_dir.mkdir(parents=True, exist_ok=True)
            rep_out_path = rep_dir / f"{Path(img_path).stem}_report.txt"
            try:
                report_out = explain_results(
                    REPORTER,
                    images=[out.get("attn_rollout_png"), out.get("grad_saliency_png"),
                            out.get("dream_png"), out.get("probs_png"), out.get("lucent_png")],
                    prediction=out.get("prediction"),
                    topk=out.get("topk"),
                    out_txt=str(rep_out_path),
                )
                print("llm_report:", report_out)
            except Exception as e:
                print("llm_report failed:", e)
        else:
            print("Skipping ChatGPT explainer (generate_report=false)")
            report_out = None

    return {
        "ok": True,
        "finetune": finetune_out,
        "feats": f_out,
        "probe": p_out,
        "metrics": m_out,
        "o1": lp_out,
        "o2": grid_out,
        "single_image": explain_out,
        "report": report_out
    }

if __name__ == "__main__":
    cfg = read_json("config/task_card.json")
    out = run_pipeline(cfg)
    print(out)
