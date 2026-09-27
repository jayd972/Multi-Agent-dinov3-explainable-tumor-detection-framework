"""
Helper module to load PyTorch models (clf.pth) for the pipeline.
"""
import torch
import torch.nn as nn
from transformers import AutoModel
from pathlib import Path
import os
from huggingface_hub import HfFolder


class DINOv3Classifier(nn.Module):
    """DINOv3 Classifier with configurable unfreezing and dropout."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int,
        num_classes: int,
        unfreeze_last_n: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.backbone = backbone

        # Freeze all backbone parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last n layers
        if unfreeze_last_n > 0:
            # Try to find the encoder layers
            if hasattr(self.backbone, 'encoder') and hasattr(self.backbone.encoder, 'layer'):
                layers = self.backbone.encoder.layer
            elif hasattr(self.backbone, 'layer'):
                layers = self.backbone.layer
            elif hasattr(self.backbone, 'blocks'):
                layers = self.backbone.blocks
            else:
                layers = None
                print("⚠️ Could not find transformer layers, unfreezing entire backbone")
                for param in self.backbone.parameters():
                    param.requires_grad = True

            if layers is not None:
                num_layers = len(layers)
                start_layer = max(0, num_layers - unfreeze_last_n)
                for i in range(start_layer, num_layers):
                    for param in layers[i].parameters():
                        param.requires_grad = True
                print(f"✅ Unfroze layers {start_layer} to {num_layers-1} ({unfreeze_last_n} layers)")

        # Matches the head trained in dinov3_augmented_finetune_experiment.ipynb:
        # Linear(hidden_dim -> 256) -> GELU -> Dropout -> Linear(256 -> num_classes).
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x=None, **kwargs):
        # Handle both dict input and keyword arguments (from AutoImageProcessor)
        if x is not None:
            if isinstance(x, dict):
                outputs = self.backbone(**x)
            else:
                outputs = self.backbone(x)
        else:
            # Use kwargs directly
            outputs = self.backbone(**kwargs)

        # Get CLS token or pooled output
        if hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state[:, 0]  # CLS token
        elif hasattr(outputs, 'pooler_output'):
            features = outputs.pooler_output
        else:
            features = outputs[0][:, 0]

        return self.classifier(features)


def load_pytorch_model(model_path: str, model_id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m", num_classes: int = 4):
    """
    Load a PyTorch model from clf.pth file.
    
    Args:
        model_path: Path to clf.pth file
        model_id: HuggingFace model ID for the backbone
        num_classes: Number of classes
        
    Returns:
        model: Loaded PyTorch model
        class_names: List of class names (if available)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load backbone
    hf_token = HfFolder.get_token() or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    backbone = AutoModel.from_pretrained(model_id, token=hf_token)
    
    # Get hidden dimension
    if hasattr(backbone.config, 'hidden_size'):
        hidden_dim = backbone.config.hidden_size
    else:
        hidden_dim = 768  # Default for ViT-Base
    
    # Create model architecture
    model = DINOv3Classifier(
        backbone=backbone,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        unfreeze_last_n=0,  # Not needed for inference
        dropout=0.0
    )
    
    # Load weights. Two known key-naming mismatches are remapped here:
    # 1) Checkpoints produced by dinov3_augmented_finetune_experiment.ipynb store the
    #    head under "head.net.*" (its DINOv3Classifier.head is an MLPHead with a "net"
    #    submodule); remap those keys onto this module's "classifier.*" attribute.
    # 2) transformers-version drift: some transformers versions nest DINOv3's
    #    transformer blocks under "backbone.model.layer.*", others expose them
    #    directly as "backbone.layer.*" (embeddings.* and norm.* are unaffected).
    #    Detect which naming the *current* backbone actually uses and remap the
    #    checkpoint to match, so this keeps working across transformers versions.
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    current_backbone_keys = set(model.backbone.state_dict().keys())
    uses_nested_model_layer = any(k.startswith("layer.") for k in current_backbone_keys) is False and \
        any(k.startswith("model.layer.") for k in current_backbone_keys)

    remapped_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("head.net."):
            key = "classifier." + key[len("head.net."):]
        elif key.startswith("backbone.model.layer.") and not uses_nested_model_layer:
            key = "backbone.layer." + key[len("backbone.model.layer."):]
        elif key.startswith("backbone.layer.") and uses_nested_model_layer:
            key = "backbone.model.layer." + key[len("backbone.layer."):]
        remapped_state_dict[key] = value
    model.load_state_dict(remapped_state_dict)
    model.eval()
    model.to(device)
    
    # Default class names (should match your dataset)
    class_names = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
    
    return model, class_names, model_id

