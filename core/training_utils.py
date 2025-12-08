# core/training_utils.py
"""
Training utility functions for feature pruning, importance computation, and diagnostics.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def prune_features_by_importance(
    feature_importances: np.ndarray,
    feature_names: List[str],
    bottom_percentile: float = 20.0
) -> List[str]:
    """
    Identify features to prune based on importance (bottom percentile).
    
    Args:
        feature_importances: Array of feature importances
        feature_names: List of feature names
        bottom_percentile: Percentile threshold (e.g., 20.0 = bottom 20%)
        
    Returns:
        List of feature names to keep (pruned list)
    """
    if len(feature_importances) == 0 or len(feature_names) == 0:
        return feature_names
    
    # Create DataFrame for easier manipulation
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    })
    
    # Calculate threshold
    threshold = np.percentile(feature_importances, bottom_percentile)
    
    # Keep features above threshold
    keep_mask = importance_df['importance'] > threshold
    kept_features = importance_df[keep_mask]['feature'].tolist()
    
    pruned_count = len(feature_names) - len(kept_features)
    print(f"  Pruning {pruned_count} features (bottom {bottom_percentile}%): {len(feature_names)} -> {len(kept_features)}")
    
    return kept_features


def save_feature_importances(
    feature_names: List[str],
    feature_importances: np.ndarray,
    output_path: Path
) -> None:
    """
    Save feature importances to CSV.
    
    Args:
        feature_names: List of feature names
        feature_importances: Array of feature importances
        output_path: Path to save CSV
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(output_path, index=False)
    print(f"  Feature importances saved to {output_path}")


def save_selected_features(
    feature_names: List[str],
    output_path: Path
) -> None:
    """
    Save list of selected features to text file.
    
    Args:
        feature_names: List of feature names
        output_path: Path to save text file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for feat in feature_names:
            f.write(f"{feat}\n")
    print(f"  Selected features saved to {output_path}")


def plot_probability_histograms(
    proba_df: pd.DataFrame,
    class_names: List[str],
    output_path: Path,
    title: str = "Probability Distribution"
) -> None:
    """
    Plot probability histograms for each class.
    
    Args:
        proba_df: DataFrame with probability columns
        class_names: List of class names (column names in proba_df)
        output_path: Path to save plot
        title: Plot title
    """
    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 4))
    if n_classes == 1:
        axes = [axes]
    
    for idx, class_name in enumerate(class_names):
        if class_name not in proba_df.columns:
            continue
        
        probs = proba_df[class_name].values
        probs = probs[~np.isnan(probs)]
        
        axes[idx].hist(probs, bins=50, alpha=0.7, edgecolor='black')
        axes[idx].set_title(f"{class_name}\nMean: {probs.mean():.3f}, Std: {probs.std():.3f}")
        axes[idx].set_xlabel("Probability")
        axes[idx].set_ylabel("Count")
        axes[idx].grid(True, alpha=0.3)
        
        # Add percentile lines
        p5 = np.percentile(probs, 5)
        p95 = np.percentile(probs, 95)
        axes[idx].axvline(p5, color='red', linestyle='--', alpha=0.5, label=f'5th: {p5:.3f}')
        axes[idx].axvline(p95, color='red', linestyle='--', alpha=0.5, label=f'95th: {p95:.3f}')
        axes[idx].legend()
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Probability histogram saved to {output_path}")


def create_training_diagnostics_report(
    model_name: str,
    n_features_before: int,
    n_features_after: int,
    feature_importance_stats: dict,
    calibration_stats: dict,
    histogram_stats: dict,
    output_path: Path
) -> None:
    """
    Create a training diagnostics report.
    
    Args:
        model_name: Name of the model (e.g., 'SignalBlender', 'DirectionBlender')
        n_features_before: Number of features before pruning
        n_features_after: Number of features after pruning
        feature_importance_stats: Dict with importance statistics
        calibration_stats: Dict with calibration statistics
        histogram_stats: Dict with histogram statistics
        output_path: Path to save report
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"TRAINING DIAGNOSTICS: {model_name}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("FEATURE PRUNING:\n")
        f.write(f"  Features before: {n_features_before}\n")
        f.write(f"  Features after: {n_features_after}\n")
        f.write(f"  Pruned: {n_features_before - n_features_after}\n\n")
        
        f.write("FEATURE IMPORTANCE:\n")
        for key, value in feature_importance_stats.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        f.write("CALIBRATION:\n")
        for key, value in calibration_stats.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        f.write("PROBABILITY DISTRIBUTION:\n")
        for key, value in histogram_stats.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
    
    print(f"  Training diagnostics saved to {output_path}")

