from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

CLASS_ORDER = ["glioma", "meningioma", "pituitary", "no_tumor"]


def _as_array(img) -> np.ndarray | None:
    if img is None:
        return None
    if isinstance(img, (str, Path)):
        p = Path(img)
        if not p.exists():
            return None
        return np.asarray(Image.open(p).convert("RGB"))
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("RGB"))
    return np.asarray(img)


def _panel(ax, img, title=None, cmap=None, vmin=0, vmax=1, overlay_base=None, colorbar=False, fig=None):
    ax.set_xticks([])
    ax.set_yticks([])
    if img is None:
        ax.text(0.5, 0.5, "unavailable", ha="center", va="center", fontsize=9)
        ax.axis("off")
        if title:
            ax.set_title(title, fontsize=10)
        return
    arr = np.asarray(img)
    if overlay_base is not None:
        ax.imshow(np.asarray(overlay_base))
        im = ax.imshow(arr, cmap=cmap or "jet", alpha=0.45, vmin=vmin, vmax=vmax)
    elif arr.ndim == 2:
        im = ax.imshow(arr, cmap=cmap or "gray", vmin=vmin if cmap == "jet" else None, vmax=vmax if cmap == "jet" else None)
    else:
        im = ax.imshow(arr)
        ax.imshow  # keep MRI as-is
    if title:
        ax.set_title(title, fontsize=10)
    if colorbar and fig is not None and arr.ndim == 2:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def make_main_figure(examples: dict, out_png: Path, out_pdf: Path | None = None):
    """examples[class] = {mri, mask, gradcam, chefer, ig, occlusion, true, pred, prob}"""
    import matplotlib.pyplot as plt

    cols = ["MRI", "GT mask", "Grad-CAM", "Chefer attrib.", "Int. Gradients", "Occlusion"]
    keys = ["mri", "mask", "gradcam", "chefer", "ig", "occlusion"]
    n = max(1, len(examples))
    fig, axes = plt.subplots(n, 6, figsize=(20, 3.4 * n), dpi=300)
    if n == 1:
        axes = np.array([axes])
    for r, cls in enumerate(examples):
        ex = examples[cls]
        for c, (title, key) in enumerate(zip(cols, keys)):
            ax = axes[r, c]
            img = ex.get(key)
            if key == "mask":
                _panel(ax, img, title if r == 0 else None, cmap="gray")
            elif key == "mri":
                _panel(ax, img, title if r == 0 else None)
            else:
                _panel(ax, img, title if r == 0 else None, cmap="jet", overlay_base=ex.get("mri"), colorbar=(r == 0), fig=fig)
            if c == 0:
                ax.set_ylabel(
                    f"{cls}\ntrue={ex.get('true')}\npred={ex.get('pred')}\nP={ex.get('prob', 0):.2f}",
                    fontsize=9,
                )
    fig.suptitle(
        "Main XAI figure. Jet overlay: yellow/red = higher attribution, blue = lower. "
        "Maps are not tumor segmentations. Mask column is official BRISC GT or 'unavailable'.",
        fontsize=11,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def make_attention_figure(examples: dict, out_png: Path, out_pdf: Path | None = None):
    import matplotlib.pyplot as plt

    cols = ["MRI", "Rollout", "Last-layer mean", "Head 0", "Head 1", "Head 2"]
    n = max(1, len(examples))
    fig, axes = plt.subplots(n, 6, figsize=(20, 3.4 * n), dpi=300)
    if n == 1:
        axes = np.array([axes])
    for r, cls in enumerate(examples):
        ex = examples[cls]
        heads = ex.get("heads") or []
        maps = [ex.get("mri"), ex.get("rollout"), ex.get("last_mean")]
        maps += [heads[i] if i < len(heads) else None for i in range(3)]
        for c, (title, img) in enumerate(zip(cols, maps)):
            ax = axes[r, c]
            if c == 0:
                _panel(ax, img, title if r == 0 else None)
                ax.set_ylabel(
                    f"{cls}\ntrue={ex.get('true')}\npred={ex.get('pred')}\nP={ex.get('prob', 0):.2f}",
                    fontsize=9,
                )
            else:
                _panel(ax, img, title if r == 0 else None, cmap="jet", overlay_base=ex.get("mri"), colorbar=(r == 0), fig=fig)
    fig.suptitle(
        "Attention figure. Rollout and last-layer attention are NOT class-specific "
        "and are not proof of prediction causality. Jet: higher attention weight.",
        fontsize=11,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def make_gallery(title: str, items: list[dict], out_png: Path, out_pdf: Path | None = None, ncols: int = 4):
    import matplotlib.pyplot as plt

    n = max(1, len(items))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows), dpi=300)
    axes = np.atleast_2d(axes)
    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        if i >= n:
            ax.axis("off")
            continue
        it = items[i]
        img = it.get("image")
        base = it.get("mri")
        if img is not None and it.get("overlay"):
            _panel(ax, img, it.get("caption"), cmap="jet", overlay_base=base, colorbar=True, fig=fig)
        else:
            _panel(ax, img if img is not None else base, it.get("caption"))
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def write_latex_table(df, path: Path, caption: str, label: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        tex = df.to_latex(index=False, float_format="%.3f", caption=caption, label=label)
    except Exception:
        tex = df.to_csv(index=False)
    path.write_text(tex, encoding="utf-8")
