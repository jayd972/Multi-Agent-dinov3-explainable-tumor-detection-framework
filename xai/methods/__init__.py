from xai.methods.attention import last_layer_attention, run_attention_rollout
from xai.methods.chefer import run_chefer_attribution
from xai.methods.gradcam import run_gradcam
from xai.methods.integrated_gradients import run_integrated_gradients
from xai.methods.occlusion import run_occlusion
from xai.methods.vanilla_grad import run_vanilla_gradient

METHOD_REGISTRY = {
    "gradcam": run_gradcam,
    "chefer_attribution": run_chefer_attribution,
    "integrated_gradients": run_integrated_gradients,
    "occlusion": run_occlusion,
    "attention_rollout": run_attention_rollout,
    "last_layer_attention": last_layer_attention,
    "vanilla_gradient": run_vanilla_gradient,
}

PRIMARY_METHODS = [
    "gradcam",
    "chefer_attribution",
    "integrated_gradients",
    "occlusion",
]
SUPPLEMENTARY_METHODS = [
    "attention_rollout",
    "last_layer_attention",
    "vanilla_gradient",
]
