from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
logger = logging.getLogger("neuralforge.utils.viz")

def plot_training_history(history, save_path=None, title="Training History"):
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5)); fig.suptitle(title, fontsize=14, fontweight="bold")
        ax = axes[0]
        if "train_loss" in history: ax.plot(history["train_loss"], label="Train Loss", linewidth=2)
        if "val_loss" in history: ax.plot(history["val_loss"], label="Val Loss", linewidth=2)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss Curves"); ax.legend(); ax.grid(True, alpha=0.3)
        ax = axes[1]
        if "lr" in history: ax.plot(history["lr"], linewidth=2, color="green"); ax.set_xlabel("Step"); ax.set_ylabel("Learning Rate"); ax.set_title("LR Schedule"); ax.set_yscale("log"); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
    except ImportError: logger.warning("matplotlib not installed"); return None

def plot_confusion_matrix(confusion_matrix, class_names=None, save_path=None, title="Confusion Matrix"):
    try:
        import matplotlib.pyplot as plt
        cm = np.array(confusion_matrix); nc = cm.shape[0]
        if class_names is None: class_names = [str(i) for i in range(nc)]
        fig, ax = plt.subplots(figsize=(max(6, nc), max(5, nc*0.8)))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues"); ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(nc), yticks=np.arange(nc), xticklabels=class_names, yticklabels=class_names, title=title, ylabel="True", xlabel="Predicted")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        thresh = cm.max() / 2.0
        for i in range(nc):
            for j in range(nc):
                ax.text(j, i, format(cm[i,j],"d"), ha="center", va="center", color="white" if cm[i,j] > thresh else "black")
        plt.tight_layout()
        if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
    except ImportError: logger.warning("matplotlib not installed"); return None
