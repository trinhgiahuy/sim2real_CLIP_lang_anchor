import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

def plot_advanced_confusion_matrices(y_true, y_pred, class_names, out_path, title_prefix="Fine-tuned CLIP Evaluation"):
    """
    Simple, clean seaborn-based confusion matrix:
      - Uses Blues colormap
      - No duplicate imshow/cbar
      - Single set of annotations (with contrast-aware colors)
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    acc = accuracy_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    # use seaborn's Blues colormap (no imshow)
    hm = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=sns.color_palette("Blues", as_cmap=True),
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="white",
        ax=ax
    )

    # set annotation colors for contrast without adding a 2nd text layer
    # (Seaborn stores the annotation Text objects in ax.collections[0].axes.texts)
    thresh = cm.max() * 0.55 if cm.max() > 0 else 0
    for txt, val in zip(ax.texts, cm.flatten()):
        txt.set_color("white" if val > thresh else "black")
        txt.set_fontsize(11)

    ax.set_title(f"{title_prefix} (acc={acc:.3f})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import pandas as pd

def calculate_metrics(true_labels, pred_labels):
    """
    Calculates a dictionary of classification metrics.
    This is the missing function required by the prompt evaluation script.
    """
    metrics = {
        'accuracy': accuracy_score(true_labels, pred_labels),
        'f1_macro': f1_score(true_labels, pred_labels, average='macro', zero_division=0),
        'precision_macro': precision_score(true_labels, pred_labels, average='macro', zero_division=0),
        'recall_macro': recall_score(true_labels, pred_labels, average='macro', zero_division=0)
    }
    return metrics

# def plot_advanced_confusion_matrices(true_labels, pred_labels, class_names, output_path, title_prefix=""):
    """
    Calculates and plots an advanced confusion matrix with both counts and percentages.
    This function is used by your main evaluation script.
    """
    # Calculate main metrics for the title
    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
    
    title = f"{title_prefix} (acc={acc:.3f}, F1-macro={f1:.3f})"
    
    # Calculate confusion matrix for counts
    cm_counts = confusion_matrix(true_labels, pred_labels)
    
    # Calculate confusion matrix for percentages
    cm_percent = cm_counts.astype('float') / cm_counts.sum(axis=1)[:, np.newaxis]

    # Create annotations that combine counts and percentages
    annot = np.array([
        [f"{count}\n({pct:.1%})" for count, pct in zip(row_counts, row_pcts)]
        for row_counts, row_pcts in zip(cm_counts, cm_percent)
    ])

    # Plotting
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_counts, annot=annot, fmt='', cmap='viridis', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"\n--- {title_prefix} Results ---")
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Macro F1-Score: {f1:.4f}")
    print(f"✅ Confusion matrix plot saved to {output_path}")