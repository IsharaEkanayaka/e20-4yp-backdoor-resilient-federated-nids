"""
UMAP Embedding Utilities for Model Analysis and Visualization

This module provides functions to:
1. Extract penultimate-layer embeddings from trained models
2. Run UMAP for dimensionality reduction
3. Visualize embeddings with class labels
4. Compare model embeddings with raw features
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import umap.umap_ as umap
from torch.utils.data import DataLoader


def get_embeddings(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    layer_name: str = 'fc3',
    return_logits: bool = False
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Extract embeddings from a specified layer of the model.
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader with (X, y) batches
        device: torch.device (cuda or cpu)
        layer_name: Name of the layer to extract embeddings from (default: 'fc3')
        return_logits: If True, also return model logits/predictions
    
    Returns:
        embeddings: numpy array of shape (n_samples, embedding_dim)
        labels: numpy array of shape (n_samples,)
        logits: (optional) numpy array of shape (n_samples, num_classes)
    """
    model.eval()
    embeddings = []
    labels = []
    logits = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            
            # Forward pass and extract embeddings based on layer
            if layer_name == 'fc3':
                # For Net and WiderNet architecture
                x = torch.relu(model.fc1(X_batch))
                x = model.dropout(x)
                x = torch.relu(model.fc2(x))
                x = model.dropout(x)
                x = torch.relu(model.fc3(x))  # 32-dim embedding
                embedding = x
                
                # Get full output if requested
                if return_logits:
                    logit = model.fc4(x)
                    logits.append(logit.cpu())
            
            elif layer_name == 'fc2':
                # Get 64-dim embedding from fc2
                x = torch.relu(model.fc1(X_batch))
                x = model.dropout(x)
                x = torch.relu(model.fc2(x))
                embedding = x
                
                if return_logits:
                    x = model.dropout(x)
                    x = torch.relu(model.fc3(x))
                    logit = model.fc4(x)
                    logits.append(logit.cpu())
            
            else:
                raise ValueError(f"Unsupported layer: {layer_name}")
            
            embeddings.append(embedding.cpu())
            labels.append(y_batch)
    
    embeddings = torch.cat(embeddings, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    
    result = (embeddings, labels)
    if return_logits:
        logits = torch.cat(logits, dim=0).numpy()
        result = (embeddings, labels, logits)
    
    return result


def run_umap(
    data: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    max_points: Optional[int] = None,
    seed: int = 42
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Run UMAP dimensionality reduction.
    
    Args:
        data: Input data of shape (n_samples, n_features)
        n_neighbors: Number of neighbors for UMAP (default: 15)
        min_dist: Minimum distance for UMAP (default: 0.1)
        random_state: Random seed for reproducibility
        max_points: If set, randomly subsample to this many points
        seed: Seed for random subsampling
    
    Returns:
        umap_result: UMAP 2D projections of shape (n_samples, 2)
        subsample_indices: Indices of subsampled points (if max_points is used)
    """
    subsample_indices = None
    
    if max_points is not None and data.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        subsample_indices = rng.choice(data.shape[0], size=max_points, replace=False)
        data = data[subsample_indices]
    
    print(f"Running UMAP on {data.shape[0]:,} samples with {data.shape[1]} features...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state
    )
    umap_result = reducer.fit_transform(data)
    print(f"✅ UMAP complete! Shape: {umap_result.shape}")
    
    return umap_result, subsample_indices


def visualize_umap(
    umap_data: np.ndarray,
    labels: np.ndarray,
    title: str = "UMAP Visualization",
    label_map: Optional[Dict[int, str]] = None,
    figsize: Tuple[int, int] = (12, 9),
    s: float = 2,
    cmap: str = 'tab10'
) -> plt.Figure:
    """
    Visualize UMAP projections with class labels.
    
    Args:
        umap_data: UMAP 2D projections of shape (n_samples, 2)
        labels: Class labels of shape (n_samples,)
        title: Title for the plot
        label_map: Dictionary mapping class indices to class names
        figsize: Figure size as (width, height)
        s: Marker size
        cmap: Colormap name
    
    Returns:
        fig: Matplotlib figure object
    """
    fig = plt.figure(figsize=figsize)
    
    plt.scatter(
        umap_data[:, 0],
        umap_data[:, 1],
        c=labels,
        cmap=cmap,
        s=s,
        alpha=0.6,
        linewidths=0,
    )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    
    # Build legend
    handles = []
    for cls in sorted(np.unique(labels)):
        cls_int = int(cls)
        if label_map:
            label_name = label_map.get(cls_int, str(cls_int))
        else:
            label_name = str(cls_int)
        
        color = plt.cm.get_cmap(cmap)(cls_int % 10)
        handles.append(
            mlines.Line2D(
                [], [],
                color=color,
                marker="o",
                linestyle="None",
                markersize=8,
                label=label_name
            )
        )
    
    plt.legend(
        handles=handles,
        title="Class",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )
    plt.tight_layout()
    
    return fig


def compare_embeddings_vs_raw(
    model: torch.nn.Module,
    data_loader: DataLoader,
    raw_data: np.ndarray,
    raw_labels: np.ndarray,
    device: torch.device,
    label_map: Optional[Dict[int, str]] = None,
    max_points: int = 50_000,
    figsize: Tuple[int, int] = (20, 8)
) -> plt.Figure:
    """
    Create side-by-side comparison of UMAP on model embeddings vs raw features.
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader for embedding extraction
        raw_data: Raw feature matrix of shape (n_samples, n_features)
        raw_labels: Labels of shape (n_samples,)
        device: torch.device
        label_map: Dictionary mapping class indices to class names
        max_points: Maximum number of points to visualize
        figsize: Figure size
    
    Returns:
        fig: Matplotlib figure object with side-by-side comparison
    """
    # Extract model embeddings
    print("📊 Extracting model embeddings...")
    embeddings, _ = get_embeddings(model, data_loader, device)
    
    # Subsample if needed
    rng = np.random.default_rng(42)
    if embeddings.shape[0] > max_points:
        idx = rng.choice(embeddings.shape[0], size=max_points, replace=False)
        emb_vis = embeddings[idx]
        raw_vis = raw_data[idx]
        labels_vis = raw_labels[idx]
    else:
        emb_vis = embeddings
        raw_vis = raw_data
        labels_vis = raw_labels
    
    # Run UMAP on both
    print("\n🗺️ Running UMAP on model embeddings...")
    umap_emb, _ = run_umap(emb_vis, max_points=None)
    
    print("🗺️ Running UMAP on raw features...")
    umap_raw, _ = run_umap(raw_vis, max_points=None)
    
    # Create side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left: Raw features
    axes[0].scatter(
        umap_raw[:, 0],
        umap_raw[:, 1],
        c=labels_vis,
        cmap='tab10',
        s=3,
        alpha=0.6,
        linewidths=0,
    )
    axes[0].set_title(f"Raw Features ({raw_vis.shape[1]}-dim)", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("UMAP-1")
    axes[0].set_ylabel("UMAP-2")
    
    # Right: Model embeddings
    axes[1].scatter(
        umap_emb[:, 0],
        umap_emb[:, 1],
        c=labels_vis,
        cmap='tab10',
        s=3,
        alpha=0.6,
        linewidths=0,
    )
    axes[1].set_title(f"Model Embeddings ({emb_vis.shape[1]}-dim from fc3)", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("UMAP-1")
    axes[1].set_ylabel("UMAP-2")
    
    # Shared legend
    handles = []
    for cls in sorted(np.unique(labels_vis)):
        cls_int = int(cls)
        if label_map:
            label_name = label_map.get(cls_int, str(cls_int))
        else:
            label_name = str(cls_int)
        
        color = plt.cm.tab10(cls_int % 10)
        handles.append(
            mlines.Line2D(
                [], [],
                color=color,
                marker="o",
                linestyle="None",
                markersize=8,
                label=label_name
            )
        )
    fig.legend(handles=handles, title="Class", bbox_to_anchor=(1.02, 0.9), loc="upper left")
    
    plt.tight_layout()
    
    return fig


def save_umap_results(
    output_dir: Path,
    umap_data: np.ndarray,
    labels: np.ndarray,
    embeddings: np.ndarray,
    metadata: Optional[Dict] = None
):
    """
    Save UMAP results and embeddings to disk.
    
    Args:
        output_dir: Directory to save results
        umap_data: UMAP 2D projections
        labels: Class labels
        embeddings: Original embedding vectors
        metadata: Additional metadata to save (model config, etc.)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / "umap_projections.npy", umap_data)
    np.save(output_dir / "labels.npy", labels)
    np.save(output_dir / "embeddings.npy", embeddings)
    
    if metadata:
        import json
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"✅ Results saved to {output_dir}")
