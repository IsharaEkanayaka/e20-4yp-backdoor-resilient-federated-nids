"""
Example: Integrating UMAP into the Main Model Training Process

This script demonstrates how to:
1. Train your federated model
2. Extract embeddings after training
3. Run UMAP analysis
4. Save visualization results

Usage:
    python scripts/analyze_with_umap.py
"""

import argparse
from pathlib import Path
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Import your project modules
from src.client.model import Net
from src.data.loader import load_dataset, get_data_loaders
from src.utils.umap_embeddings import (
    get_embeddings,
    run_umap,
    visualize_umap,
    compare_embeddings_vs_raw,
    save_umap_results
)


def analyze_trained_model(
    model_path: str,
    config_path: str,
    data_path: str = "data/unsw-nb15/processed/global_test.pt",
    output_dir: str = "outputs/umap_analysis",
    classification_mode: str = "binary",
    device: str = "cuda",
    max_points: int = 50_000
):
    """
    Load a trained model and analyze its embeddings with UMAP.
    
    Args:
        model_path: Path to saved model state dict
        config_path: Path to config file used in training
        data_path: Path to preprocessed test data
        output_dir: Directory to save UMAP visualizations
        classification_mode: 'binary' or 'multiclass'
        device: 'cuda' or 'cpu'
        max_points: Max points for visualization (for memory efficiency)
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📍 Device: {device}")
    print(f"📁 Output directory: {output_dir}")
    
    # ============================================================================
    # 1. Load Configuration
    # ============================================================================
    print("\n📋 Loading configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    input_dim = config['input_dim']
    num_classes = config['num_classes']
    label_map = config.get('label_map', {})
    
    print(f"   Input dim: {input_dim}")
    print(f"   Num classes: {num_classes}")
    
    # ============================================================================
    # 2. Load Model
    # ============================================================================
    print("\n🏗️ Loading trained model...")
    model = Net(input_dim=input_dim, num_classes=num_classes).to(device)
    
    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device)
        # Handle both direct state dict and checkpoint dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✅ Model loaded from {model_path}")
    else:
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # ============================================================================
    # 3. Load Data
    # ============================================================================
    print("\n📂 Loading data...")
    
    # Load preprocessed test data
    test_data = torch.load(data_path, weights_only=False)
    if isinstance(test_data, dict):
        X_test, y_test = test_data['X'], test_data['y']
    else:
        X_test, y_test = test_data.tensors[0], test_data.tensors[1]
    
    print(f"✅ Test set: {X_test.shape[0]:,} samples, {X_test.shape[1]} features")
    
    # Create data loader
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # ============================================================================
    # 4. Extract Embeddings
    # ============================================================================
    print("\n🧠 Extracting model embeddings...")
    embeddings, labels, logits = get_embeddings(
        model,
        test_loader,
        device,
        layer_name='fc3',
        return_logits=True
    )
    print(f"✅ Embeddings shape: {embeddings.shape}")
    print(f"   Embedding dimension: {embeddings.shape[1]}")
    
    # ============================================================================
    # 5. Run UMAP on Model Embeddings
    # ============================================================================
    print("\n🗺️ Running UMAP on model embeddings...")
    umap_emb, emb_indices = run_umap(
        embeddings,
        max_points=max_points,
        random_state=42
    )
    
    # Get corresponding labels for visualization
    if emb_indices is not None:
        vis_labels = labels[emb_indices]
    else:
        vis_labels = labels
    
    # ============================================================================
    # 6. Visualize UMAP (Embeddings)
    # ============================================================================
    print("\n🎨 Creating visualizations...")
    fig_emb = visualize_umap(
        umap_emb,
        vis_labels,
        title="UMAP of Model Embeddings (penultimate layer, 32-dim)",
        label_map=label_map,
        figsize=(12, 9)
    )
    fig_emb.savefig(output_dir / "umap_embeddings.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'umap_embeddings.png'}")
    plt.close(fig_emb)
    
    # ============================================================================
    # 7. Compare with Raw Features
    # ============================================================================
    print("\n📊 Comparing embeddings vs raw features...")
    X_test_np = X_test.numpy()
    
    fig_comp = compare_embeddings_vs_raw(
        model,
        test_loader,
        X_test_np,
        y_test.numpy(),
        device,
        label_map=label_map,
        max_points=max_points,
        figsize=(20, 8)
    )
    fig_comp.savefig(output_dir / "comparison_embeddings_vs_raw.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'comparison_embeddings_vs_raw.png'}")
    plt.close(fig_comp)
    
    # ============================================================================
    # 8. Save Results
    # ============================================================================
    print("\n💾 Saving analysis results...")
    metadata = {
        'input_dim': input_dim,
        'embedding_dim': embeddings.shape[1],
        'num_samples': len(labels),
        'num_classes': num_classes,
        'model_path': str(model_path),
        'config_path': str(config_path),
        'max_points_visualized': len(vis_labels)
    }
    
    save_umap_results(
        output_dir,
        umap_emb,
        vis_labels,
        embeddings if emb_indices is None else embeddings[emb_indices],
        metadata=metadata
    )
    
    print(f"\n✨ UMAP analysis complete!")
    print(f"   - UMAP projections: {output_dir / 'umap_projections.npy'}")
    print(f"   - Embeddings: {output_dir / 'embeddings.npy'}")
    print(f"   - Labels: {output_dir / 'labels.npy'}")
    print(f"   - Metadata: {output_dir / 'metadata.json'}")
    print(f"   - Visualization: {output_dir / 'umap_embeddings.png'}")
    print(f"   - Comparison: {output_dir / 'comparison_embeddings_vs_raw.png'}")


def main():
    parser = argparse.ArgumentParser(description="Analyze trained model with UMAP")
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/trained_model.pth",
        help="Path to trained model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/central/m1_baseline.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/unsw-nb15/processed/global_test.pt",
        help="Path to test data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/umap_analysis",
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=50_000,
        help="Max points for visualization"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    
    args = parser.parse_args()
    
    analyze_trained_model(
        model_path=args.model,
        config_path=args.config,
        data_path=args.data,
        output_dir=args.output,
        max_points=args.max_points,
        device=args.device
    )


if __name__ == "__main__":
    main()
