import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import wandb
from pathlib import Path
import matplotlib.pyplot as plt

# Import our custom modules
from src.data.loader import load_dataset, get_data_loaders
from src.data.partition import partition_data
from src.client.client import Client
from src.client.model import Net
from src.server.server import Server
from src.utils.logger import Logger
from src.utils.umap_embeddings import (
    get_embeddings,
    run_umap,
    visualize_umap,
    compare_embeddings_vs_raw,
    save_umap_results
)

# Ensure your Hydra config path is correct relative to where you run this!
@hydra.main(config_path="configs/federated", config_name="baseline", version_base=None)
def main(cfg: DictConfig):
    print(f"🚀 Starting Experiment: {cfg.simulation.partition_method} Partition")

    # Get classification mode early for W&B grouping
    classification_mode = cfg.data.get('classification_mode', 'binary')
    print(f"🎯 Classification Mode: {classification_mode}")
    
    # Check if a group is defined (e.g. from command line +group=exp1)
    base_group = cfg.get("group", "default")
    # Automatically append classification mode to group name
    wandb_group = f"{base_group}_{classification_mode}"
    print(f"📊 W&B Group: {wandb_group}")
    
    # Prepare tags - automatically generate from config
    wandb_tags = cfg.get("tags", [])
    if isinstance(wandb_tags, str):
        wandb_tags = [wandb_tags]
    elif wandb_tags is None:
        wandb_tags = []
    
    # Auto-generate tags from config settings
    wandb_tags.append(classification_mode)  # binary or multiclass
    wandb_tags.append(cfg.simulation.partition_method)  # iid or non-iid
    wandb_tags.append(cfg.server.defense)  # avg, krum, median, etc.
    wandb_tags.append(cfg.attack.type)  # clean, backdoor, label_flip
    
    # Add dataset tag from path
    if "unsw" in cfg.data.path.lower():
        wandb_tags.append("unsw-nb15")
    elif "cic" in cfg.data.path.lower():
        wandb_tags.append("cic-ids2017")
    
    # Add attack intensity tag if backdoor/label_flip
    if cfg.attack.type != "clean":
        wandb_tags.append(f"malicious-{cfg.attack.num_malicious_clients}")
        if cfg.attack.get("aggressive", False):
            wandb_tags.append("model-replacement")
    
    print(OmegaConf.to_yaml(cfg))

    # 🛡️ 0. INITIALIZE LOGGER
    logger = Logger(
        cfg, 
        project_name="e20-4yp-backdoor-resilient-federated-nids",
        group_name=wandb_group,
        tags=wandb_tags
    )

    if wandb.run:
        # If wandb has overridden params, update our Hydra config 'cfg'
        if wandb.config.get('client.lr'):
            cfg.client.lr = wandb.config['client.lr']
        if wandb.config.get('client.batch_size'):
            cfg.client.batch_size = wandb.config['client.batch_size']
        if wandb.config.get('client.epochs'):
            cfg.client.epochs = wandb.config['client.epochs']

    # 1. SETUP DATA
    # classification_mode already extracted above for W&B grouping
    
    train_pool, input_dim, num_classes = load_dataset(
        cfg.data.path, 
        classification_mode=classification_mode
    )
    
    # Load the specific Global Test Set (for the Server)
    _, test_loader, _, _ = get_data_loaders(
        path=cfg.data.path,         
        batch_size=cfg.client.batch_size,
        classification_mode=classification_mode
    )
    
    # Partition the data
    client_indices = partition_data(
        train_pool, 
        n_clients=cfg.simulation.n_clients, 
        method=cfg.simulation.partition_method, 
        alpha=cfg.simulation.alpha
    )

    # 😈 RED TEAM LOGIC START 😈
    attack_type = cfg.get("attack", {}).get("type", "clean")
    malicious_ids = []
    num_malicious = 0
    
    if attack_type != "clean":
        # Get count from config, default to 1 if not set
        num_malicious = cfg.attack.get("num_malicious_clients", 1)
        
        # Pick random clients to be malicious
        malicious_ids = np.random.choice(
            range(cfg.simulation.n_clients), 
            num_malicious, 
            replace=False
        ).tolist()
        
        print(f"⚠️ ATTACK ACTIVE: {attack_type}")
        print(f"⚠️ {len(malicious_ids)} Malicious Clients: {malicious_ids}")
    # 😈 RED TEAM LOGIC END 😈

    # 2. SETUP AGENTS
    # Define the Global Model (The "Brain")
    global_model = Net(input_dim=input_dim, num_classes=num_classes)
    
    # Initialize Server
    # PASS 'num_malicious' HERE so Krum knows how many to reject
    server = Server(
        cfg,
        global_model, 
        test_loader, 
        device=cfg.client.device,
        defense=cfg.server.defense,
        expected_malicious=num_malicious,
        num_classes=num_classes
    )
    
    # Initialize Clients
    clients = []
    print("👥 Initializing Clients...")
    
    for cid in range(cfg.simulation.n_clients):
        # Determine if this specific client is malicious
        is_malicious = (cid in malicious_ids)
        
        client = Client(
            client_id=cid,
            dataset=train_pool,
            indices=client_indices[cid],
            model=global_model,
            config=cfg,  # <--- PASS THE FULL CONFIG HERE
            lr=cfg.client.lr,
            device=cfg.client.device,
            is_malicious=is_malicious # <--- PASS THE FLAG
        )
        clients.append(client)

    best_acc = 0.0

    # 3. FEDERATED LEARNING LOOP
    print("\n🔄 Starting FL Loop...")
    for round_id in range(cfg.simulation.rounds):
        print(f"\n--- Round {round_id + 1}/{cfg.simulation.rounds} ---")
        
        # A. Client Selection
        n_participants = int(cfg.simulation.n_clients * cfg.simulation.fraction)
        n_participants = max(1, n_participants)
        
        active_clients_indices = np.random.choice(
            range(cfg.simulation.n_clients), n_participants, replace=False
        )
        
        # B. Training Phase
        client_updates = []
        
        for cid in active_clients_indices:
            client = clients[cid]
            
            # Train and get updates
            w_local, n_samples, loss = client.train(
                global_weights=server.global_model.state_dict(),
                epochs=cfg.client.epochs,
                batch_size=cfg.client.batch_size
            )
            
            client_updates.append((w_local, n_samples, loss))

        # C. Aggregation Phase (Server)
        server.aggregate(client_updates)
        
        # D. Evaluation Phase
        acc, f1_score = server.evaluate()
        asr = server.test_attack_efficacy(cfg.attack)
        
        print(f"📊 Round {round_id+1} | Accuracy: {acc:.2f}% | F1-score: {f1_score:.2f} | 😈 Backdoor ASR: {asr:.2f}%")
        print(f"📊 Global Accuracy: {acc:.2f}%")

        # E. LOGGING
        logger.log_metrics(
            metrics={
                "Accuracy": acc,
                "f1-score": f1_score,
                "ASR": asr
            },
            step=round_id + 1
        )
    
    # 🧠 OPTIONAL: UMAP ANALYSIS
    if cfg.get("umap", {}).get("enabled", False):
        print("\n" + "="*60)
        print("🧠 Running UMAP Embedding Analysis...")
        print("="*60)
        
        try:
            run_umap_analysis(
                model=server.global_model,
                test_loader=test_loader,
                test_data_raw=None,  # Will be loaded inside function
                config=cfg,
                num_classes=num_classes,
                input_dim=input_dim
            )
        except Exception as e:
            print(f"⚠️ UMAP analysis failed: {str(e)}")
            print("   Continuing without UMAP analysis...")
        
    print("\n✅ Experiment Complete!")


    # 💾 SAVE THE TRAINED MODEL MANUALLY
    # This saves the Server's final global model, which is what you want.
    save_path = "final_model.pt"
    torch.save(server.global_model.state_dict(), save_path)
    print(f"💾 Global Model successfully saved to: {save_path}")

    
    logger.finish()


def run_umap_analysis(
    model: torch.nn.Module,
    test_loader,
    test_data_raw,
    config: DictConfig,
    num_classes: int,
    input_dim: int
):
    """
    Run UMAP embedding analysis after training completes.
    
    Args:
        model: Trained global model
        test_loader: DataLoader for test set
        test_data_raw: Raw test features (if available)
        config: Hydra config with UMAP settings
        num_classes: Number of classes
        input_dim: Input feature dimension
    """
    device = torch.device(config.client.device if torch.cuda.is_available() else 'cpu')
    umap_cfg = config.get("umap", {})
    
    # Get UMAP settings with defaults
    max_points = umap_cfg.get("max_points", 50_000)
    enable_comparison = umap_cfg.get("compare_with_raw_features", True)
    output_dir = Path(umap_cfg.get("output_dir", "outputs/umap_analysis"))
    
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Dataset size for UMAP: {max_points:,} points")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract embeddings
    print("\n📊 Extracting model embeddings from test set...")
    embeddings, labels = get_embeddings(
        model,
        test_loader,
        device,
        layer_name='fc3',
        return_logits=False
    )
    
    print(f"✅ Extracted {embeddings.shape[0]:,} embeddings of dimension {embeddings.shape[1]}")
    
    # Run UMAP
    print(f"\n🗺️ Running UMAP (using {max_points:,} points)...")
    umap_proj, subsample_idx = run_umap(
        embeddings,
        n_neighbors=umap_cfg.get("n_neighbors", 15),
        min_dist=umap_cfg.get("min_dist", 0.1),
        max_points=max_points,
        random_state=umap_cfg.get("random_state", 42)
    )
    
    # Get corresponding labels for visualization
    if subsample_idx is not None:
        vis_labels = labels[subsample_idx]
        vis_embeddings = embeddings[subsample_idx]
    else:
        vis_labels = labels
        vis_embeddings = embeddings
    
    # Visualize embeddings
    print(f"\n🎨 Creating UMAP visualization...")
    fig_emb = visualize_umap(
        umap_proj,
        vis_labels,
        title=f"UMAP: Model Embeddings (fc3 layer, {embeddings.shape[1]}-dim)\n"
              f"Classification Mode: {config.data.get('classification_mode', 'binary')} | "
              f"Defense: {config.server.get('defense', 'avg')} | "
              f"Attack: {config.attack.get('type', 'clean')}",
        figsize=(14, 10)
    )
    
    # Save main visualization
    emb_path = output_dir / "umap_embeddings.png"
    fig_emb.savefig(emb_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {emb_path}")
    plt.close(fig_emb)
    
    # Optional: Compare with raw features
    if enable_comparison:
        print(f"\n📊 Comparing model embeddings with raw features...")
        try:
            from torch.utils.data import TensorDataset
            
            # Load raw test data
            test_data_path = Path(config.data.path)
            if "train_pool" in str(test_data_path):
                test_data_path = test_data_path.parent / "global_test.pt"
            
            if test_data_path.exists():
                test_data = torch.load(test_data_path, weights_only=False)
                if isinstance(test_data, dict):
                    X_test = test_data['X'].numpy()
                else:
                    X_test = test_data.tensors[0].numpy()
                
                # Run UMAP on raw features with same subsample
                if subsample_idx is not None:
                    X_test_vis = X_test[subsample_idx]
                else:
                    if X_test.shape[0] > max_points:
                        rng = np.random.default_rng(42)
                        idx = rng.choice(X_test.shape[0], size=max_points, replace=False)
                        X_test_vis = X_test[idx]
                    else:
                        X_test_vis = X_test
                
                print(f"🗺️ Running UMAP on raw features ({X_test_vis.shape[1]}-dim)...")
                umap_raw, _ = run_umap(
                    X_test_vis,
                    n_neighbors=umap_cfg.get("n_neighbors", 15),
                    min_dist=umap_cfg.get("min_dist", 0.1),
                    max_points=None,
                    random_state=umap_cfg.get("random_state", 42)
                )
                
                # Create side-by-side comparison
                print(f"\n🎨 Creating comparison visualization...")
                fig, axes = plt.subplots(1, 2, figsize=(20, 9))
                
                # Raw features
                axes[0].scatter(
                    umap_raw[:, 0], umap_raw[:, 1],
                    c=vis_labels, cmap='tab10', s=3, alpha=0.6, linewidths=0
                )
                axes[0].set_title(f"Raw Features ({X_test_vis.shape[1]}-dim)", 
                                  fontsize=14, fontweight='bold')
                axes[0].set_xlabel("UMAP-1")
                axes[0].set_ylabel("UMAP-2")
                axes[0].grid(alpha=0.3)
                
                # Model embeddings
                axes[1].scatter(
                    umap_proj[:, 0], umap_proj[:, 1],
                    c=vis_labels, cmap='tab10', s=3, alpha=0.6, linewidths=0
                )
                axes[1].set_title(f"Model Embeddings ({embeddings.shape[1]}-dim from fc3)", 
                                  fontsize=14, fontweight='bold')
                axes[1].set_xlabel("UMAP-1")
                axes[1].set_ylabel("UMAP-2")
                axes[1].grid(alpha=0.3)
                
                plt.suptitle("Comparison: Raw Features vs Learned Embeddings", 
                             fontsize=16, fontweight='bold', y=1.00)
                plt.tight_layout()
                
                comp_path = output_dir / "umap_comparison.png"
                fig.savefig(comp_path, dpi=150, bbox_inches='tight')
                print(f"✅ Saved: {comp_path}")
                plt.close(fig)
            
        except Exception as e:
            print(f"⚠️ Comparison visualization failed: {str(e)}")
    
    # Save numerical results
    print(f"\n💾 Saving UMAP results...")
    metadata = {
        'embedding_dim': int(embeddings.shape[1]),
        'total_samples': int(embeddings.shape[0]),
        'visualized_samples': int(len(vis_labels)),
        'num_classes': int(num_classes),
        'input_dim': int(input_dim),
        'classification_mode': str(config.data.get('classification_mode', 'binary')),
        'defense_method': str(config.server.get('defense', 'avg')),
        'attack_type': str(config.attack.get('type', 'clean')),
        'federated_rounds': int(config.simulation.rounds),
        'num_clients': int(config.simulation.n_clients),
        'umap_n_neighbors': int(umap_cfg.get("n_neighbors", 15)),
        'umap_min_dist': float(umap_cfg.get("min_dist", 0.1)),
    }
    
    save_umap_results(
        output_dir,
        umap_proj,
        vis_labels,
        vis_embeddings,
        metadata=metadata
    )
    
    print(f"\n✨ UMAP Analysis Complete!")
    print(f"   📊 Results saved to: {output_dir}")
    print(f"   📈 Visualizations:")
    print(f"      - {output_dir / 'umap_embeddings.png'}")
    if enable_comparison:
        print(f"      - {output_dir / 'umap_comparison.png'}")
    print(f"   💾 Data files:")
    print(f"      - {output_dir / 'umap_projections.npy'}")
    print(f"      - {output_dir / 'embeddings.npy'}")
    print(f"      - {output_dir / 'labels.npy'}")
    print(f"      - {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()