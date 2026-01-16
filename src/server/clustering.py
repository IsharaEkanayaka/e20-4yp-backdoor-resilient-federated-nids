#FLAME's defences's clustering method (need for "flame" defence)

import torch
import numpy as np
import hdbscan
from sklearn.metrics.pairwise import cosine_distances

def fl_trust_clustering(weights_list):
    n_clients = len(weights_list)
    
    # 1. Flatten updates into vectors
    flat_updates = []
    for w in weights_list:
        concat_list = []
        for key in sorted(w.keys()):
            concat_list.append(w[key].view(-1).float())
        flat_updates.append(torch.cat(concat_list).cpu().numpy())
    
    flat_updates = np.array(flat_updates)

    # 2. Calculate Pairwise Cosine Distances
    # 🔧 FIX: HDBSCAN requires float64 (double), but PyTorch gave float32.
    distances = cosine_distances(flat_updates).astype(np.float64)

    # 3. Apply HDBSCAN
    # FLAME uses min_cluster_size > n/2 to find the "Majority" (Honest) group[cite: 274].
    min_cluster_size = int(n_clients / 2) + 1
    
    clusterer = hdbscan.HDBSCAN(
        metric='precomputed', 
        min_cluster_size=min_cluster_size, 
        min_samples=1,
        allow_single_cluster=True
    )
    
    labels = clusterer.fit_predict(distances)
    
    # 4. Select the "Benign" Cluster
    # Labels: -1 is noise (malicious), 0+ are clusters.
    
    # Check if we found ANY cluster (if everything is noise, fallback)
    if np.max(labels) < 0:
        print("⚠️ FLAME Clustering: No majority group found! Accepting all.")
        return weights_list

    # Find the cluster with the most clients (The Majority)
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Filter out the noise label (-1) from consideration
    valid_mask = unique_labels != -1
    unique_labels = unique_labels[valid_mask]
    counts = counts[valid_mask]
    
    if len(counts) == 0:
         print("⚠️ FLAME Clustering: Only noise found. Accepting all.")
         return weights_list
         
    benign_cluster_id = unique_labels[np.argmax(counts)]
    
    # 5. Filter the weights
    selected_indices = np.where(labels == benign_cluster_id)[0]
    
    print(f"🔥 FLAME Clustering: Selected {len(selected_indices)}/{n_clients} clients (Rejected {n_clients - len(selected_indices)})")
    
    accepted_weights = [weights_list[i] for i in selected_indices]
    
    return accepted_weights