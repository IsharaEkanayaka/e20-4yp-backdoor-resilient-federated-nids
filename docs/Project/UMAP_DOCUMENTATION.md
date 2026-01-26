# 📊 UMAP Embedding Analysis - Complete Documentation

**Table of Contents**
- [Quick Start](#-quick-start)
- [What is UMAP](#what-is-umap)
- [Configuration](#-configuration)
- [How to Run](#-how-to-run)
- [Understanding Results](#-understanding-results)
- [Examples](#-examples)
- [Troubleshooting](#-troubleshooting)
- [Advanced Usage](#-advanced-usage)

---

## 🚀 Quick Start

### The Simplest Way (2 minutes)

```bash
# 1. Activate your conda environment
conda activate fl-nids

# 2. Enable UMAP in config
nano configs/federated/baseline.yaml
# Change: umap.enabled: true

# set the below code (in the baseline.yaml) to true to start the UMAP visualization. If it has enabled, the UMAP visualization process will execute automatically when you run the main.py.
umap:
  enabled: false

# 3. Run training
python main.py

# UMAP will automatically run after training completes!
```

### View Results
```bash
code outputs/umap_analysis/umap_embeddings.png
```

**That's it!** ✨

---

## What is UMAP?

**UMAP** (Uniform Manifold Approximation and Projection) is a dimensionality reduction technique that:
- Takes high-dimensional data (your 32-dim embeddings from fc3 layer)
- Projects it into 2D for visualization
- Preserves both local and global structure
- Shows how well your model separates different attack classes

**In your case:**
- Input: 32-dimensional learned embeddings from the model's fc3 layer
- Output: 2D coordinates for visualization
- Result: Beautiful plots showing class separation

---

## ⚙️ Configuration

### Location
`configs/federated/baseline.yaml` - UMAP section

### All Options

```yaml
umap:
  # MAIN CONTROL
  enabled: true              # Enable/disable UMAP (true or false)
  
  # PERFORMANCE
  max_points: 5000           # How many points to visualize
                             # 5000  = Fast (15 sec)
                             # 10000 = Standard (20 sec)
                             # 50000 = Detailed (2 min)
                             # 100000 = Full data (3+ min)
  
  # VISUALIZATIONS
  compare_with_raw_features: false    # Compare with raw features (adds 30 sec)
  
  # UMAP PARAMETERS
  n_neighbors: 15            # 5 = local focus, 15 = balanced, 30 = global
  min_dist: 0.1             # 0.05 = tight clusters, 0.1 = medium, 0.5 = spread
  random_state: null         # null = parallel (faster), 42 = reproducible (slower)
  
  # OUTPUT
  output_dir: "outputs/umap_analysis"  # Where to save files
```

### Quick Presets

**Ultra Fast (10 seconds):**
```yaml
umap:
  enabled: true
  max_points: 2000
  compare_with_raw_features: false
  random_state: null
```

**Fast (15-20 seconds) ⭐ RECOMMENDED:**
```yaml
umap:
  enabled: true
  max_points: 5000
  compare_with_raw_features: false
  random_state: null
```

**Standard (30 seconds):**
```yaml
umap:
  enabled: true
  max_points: 10000
  compare_with_raw_features: false
  random_state: null
```

**High Quality (2 minutes):**
```yaml
umap:
  enabled: true
  max_points: 50000
  compare_with_raw_features: true
  random_state: 42
```

**Full Analysis (3+ minutes):**
```yaml
umap:
  enabled: true
  max_points: 100000
  compare_with_raw_features: true
  random_state: 42
```

**Disabled (Training Only):**
```yaml
umap:
  enabled: false
```

---

## 🏃 How to Run

### Method 1: Automatic (Recommended)

UMAP runs automatically after training completes:

```bash
conda activate fl-nids
python main.py
```

**What happens:**
1. ✅ Federated training runs normally
2. ✅ After training finishes, UMAP starts automatically
3. ✅ Extracts 32-dim embeddings from fc3 layer
4. ✅ Runs UMAP dimensionality reduction
5. ✅ Creates visualizations
6. ✅ Saves results to `outputs/umap_analysis/`

### Method 2: Standalone (For Existing Models)

Analyze a model you already trained:

```bash
python scripts/analyze_with_umap.py \
    --model outputs/trained_model.pth \
    --config configs/federated/baseline.yaml \
    --output outputs/my_umap_results \
    --max-points 10000
```

### Method 3: Programmatic (For Advanced Users)

```python
from src.utils.umap_embeddings import (
    get_embeddings,
    run_umap,
    visualize_umap
)

# Extract embeddings
embeddings, labels = get_embeddings(
    model, test_loader, device, layer_name='fc3'
)

# Run UMAP
umap_proj, _ = run_umap(
    embeddings,
    max_points=10000,
    n_neighbors=15,
    min_dist=0.1
)

# Visualize
fig = visualize_umap(
    umap_proj, labels,
    title="My UMAP Analysis"
)
plt.show()
```

---

## 📊 Understanding Results

### Output Files

After UMAP completes, you get:

```
outputs/umap_analysis/
├── umap_embeddings.png          ⭐ Main visualization
├── embeddings.npy               💾 Raw 32-dim embeddings
├── umap_projections.npy         💾 2D UMAP coordinates
├── labels.npy                   💾 Class labels
└── metadata.json                📋 Analysis metadata
```

### The Visualizations

#### UMAP Embeddings Plot

Shows 2D projection of your model's learned embeddings:

```
   UMAP-2 ↑
          │     🔵🔵🔵  (Normal traffic)
          │   🔵     🔵
          │
          │  🟢🟢         🟡🟡  (DoS attacks)
          │    🟢       🟡
          │
          │  🟠🟠        🔴  (Backdoor)
          │   🟠     🔴🔴
          │
          └────────────────────→ UMAP-1
```

**What to look for:**
- ✅ **Good:** Each color (attack type) is clustered together
- ❌ **Bad:** Colors are mixed and scattered randomly
- 🟢 **Excellent:** Clear separation between all classes

#### Comparison Plot (if enabled)

Side-by-side comparison:
- **Left:** Raw 71-dimensional features
- **Right:** Learned 32-dimensional embeddings

**Interpretation:**
- If RIGHT is more separated than LEFT → Model learned well ✅
- If they're similar → Model doesn't add value ⚠️
- If RIGHT is worse → Model may be overfitting ❌

### Metadata

`metadata.json` contains:
- Model dimensions
- Training parameters
- Number of samples analyzed
- UMAP settings used
- Timestamp

Load and check:
```python
import json
with open('outputs/umap_analysis/metadata.json') as f:
    data = json.load(f)
    print(data)
```

---

## 💡 Examples

### Example 1: Quick Test Run

```bash
# Edit config for fast run
nano configs/federated/baseline.yaml
# Set: max_points: 2000
# Set: compare_with_raw_features: false

# Run
conda activate fl-nids
python main.py

# View (should complete in ~10 seconds)
code outputs/umap_analysis/umap_embeddings.png
```

### Example 2: Detailed Analysis

```bash
# Edit config for detailed analysis
nano configs/federated/baseline.yaml
# Set: max_points: 50000
# Set: compare_with_raw_features: true
# Set: random_state: 42

# Run
conda activate fl-nids
python main.py

# Wait for completion (~2 minutes)
# View both plots
code outputs/umap_analysis/umap_embeddings.png
code outputs/umap_analysis/umap_comparison.png
```

### Example 3: Analyze Different Models

```bash
# Analyze model without attack
python main.py --attack.type clean

# Save results
cp -r outputs/umap_analysis outputs/umap_clean

# Analyze model with backdoor attack
python main.py --attack.type backdoor

# Compare the two
code outputs/umap_clean/umap_embeddings.png
code outputs/umap_analysis/umap_embeddings.png
```

### Example 4: Load and Analyze Embeddings

```python
import numpy as np
from sklearn.metrics import silhouette_score

# Load results
embeddings = np.load('outputs/umap_analysis/embeddings.npy')
labels = np.load('outputs/umap_analysis/labels.npy')

# Compute clustering quality
silhouette = silhouette_score(embeddings, labels)
print(f"Silhouette Score: {silhouette:.3f}")

# Check class distribution
unique, counts = np.unique(labels, return_counts=True)
for cls, count in zip(unique, counts):
    print(f"Class {cls}: {count:,} samples")

# Statistics
print(f"Embedding mean: {embeddings.mean():.3f}")
print(f"Embedding std: {embeddings.std():.3f}")
```

---

## 🆘 Troubleshooting

### Problem: UMAP Installation Missing

**Error:** `ModuleNotFoundError: No module named 'umap'`

**Solution:**
```bash
conda activate fl-nids
pip install umap-learn
```

### Problem: Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
```yaml
umap:
  max_points: 2000  # Reduce from 10000
```

Or use CPU instead:
```bash
# Edit main.py, change device to cpu
```

### Problem: UMAP Takes Too Long

**Solution:** Reduce points and disable comparison:
```yaml
umap:
  enabled: true
  max_points: 5000
  compare_with_raw_features: false
  random_state: null
```

### Problem: No UMAP Output Generated

**Check:** Is UMAP actually enabled?
```bash
grep "enabled:" configs/federated/baseline.yaml
# Should show: enabled: true
```

### Problem: Results Not Showing

**Check:** Did training complete successfully?
```bash
# Check for errors in console output
# Look for: "✨ Training Complete!"
```

**If training was interrupted:**
- Exit code 130 = You pressed Ctrl+C
- Just run again: `python main.py`

### Problem: Weird UMAP Visualization

**Possible causes:**
- Model didn't learn properly (try different hyperparameters)
- Not enough training data
- Imbalanced classes

**Try:**
- Increase training rounds
- Use class weights
- Check training accuracy first

---

## 🔧 Advanced Usage

### Use Different Model Layers

By default, embeddings are extracted from `fc3` (32-dim). To use `fc2` (64-dim):

Edit `src/utils/umap_embeddings.py`:
```python
# In get_embeddings function, change from:
embeddings, labels = get_embeddings(model, loader, device, layer_name='fc3')

# To:
embeddings, labels = get_embeddings(model, loader, device, layer_name='fc2')
```

### Adjust UMAP Parameters

For different visualizations:

```yaml
# More local structure (tight clusters)
umap:
  n_neighbors: 5
  min_dist: 0.05

# More global structure (spread out)
umap:
  n_neighbors: 30
  min_dist: 0.5

# Balanced (default)
umap:
  n_neighbors: 15
  min_dist: 0.1
```

### Custom Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

# Load data
umap_proj = np.load('outputs/umap_analysis/umap_projections.npy')
labels = np.load('outputs/umap_analysis/labels.npy')

# Create custom plot
fig, ax = plt.subplots(figsize=(16, 12))

scatter = ax.scatter(
    umap_proj[:, 0],
    umap_proj[:, 1],
    c=labels,
    cmap='viridis',
    s=5,
    alpha=0.7,
    edgecolors='none'
)

ax.set_title("Custom UMAP Visualization", fontsize=16, fontweight='bold')
ax.set_xlabel("UMAP-1", fontsize=14)
ax.set_ylabel("UMAP-2", fontsize=14)

plt.colorbar(scatter, label="Attack Class")
plt.tight_layout()
plt.savefig('custom_umap.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Batch Analysis

```bash
# Run multiple analyses with different settings
for max_points in 5000 10000 50000; do
    echo "Running with max_points=$max_points"
    
    # Update config
    sed -i "s/max_points: .*/max_points: $max_points/" configs/federated/baseline.yaml
    
    # Run
    python main.py
    
    # Save results
    mkdir -p "outputs/umap_$max_points"
    cp -r outputs/umap_analysis/* "outputs/umap_$max_points/"
done
```

### Track Embeddings Over Training Rounds

Modify `main.py` to save UMAP at different rounds:

```python
# In the training loop, after each round:
if (round_id + 1) % 5 == 0:  # Every 5 rounds
    print(f"Saving UMAP for round {round_id + 1}...")
    run_umap_analysis(
        model=server.global_model,
        test_loader=test_loader,
        config=cfg,
        output_dir=f"outputs/umap_round_{round_id + 1}"
    )
```

---

## 📈 Performance Tips

| Config | Time | Use Case |
|--------|------|----------|
| max_points: 2000 | ~10s | Quick tests |
| max_points: 5000 | ~15s | Default ⭐ |
| max_points: 10000 | ~20s | Standard |
| max_points: 50000 | ~2min | Detailed |
| max_points: 100000 | ~3min | Full data |

**Recommendations:**
- Use **5000-10000** for development
- Use **50000+** for final analysis
- Disable `compare_with_raw_features` if not needed (saves 30s)
- Use `random_state: null` for speed (parallel processing)
- Use `random_state: 42` only if reproducibility is critical

---

## 📞 Quick Reference

```bash
# Check UMAP installed
python -c "import umap; print('✅')"

# View current config
grep -A 10 "umap:" configs/federated/baseline.yaml

# Enable UMAP
nano configs/federated/baseline.yaml  # Change enabled: true

# Run training with UMAP
conda activate fl-nids
python main.py

# View results
code outputs/umap_analysis/umap_embeddings.png

# List all generated files
ls -lh outputs/umap_analysis/

# Load embeddings
python -c "import numpy as np; e=np.load('outputs/umap_analysis/embeddings.npy'); print(e.shape)"

# Delete old results
rm -rf outputs/umap_analysis/
```

---

## 🎓 Learning More

**About UMAP:**
- Official docs: https://umap-learn.readthedocs.io/
- Paper: "UMAP: Uniform Manifold Approximation and Projection"

**About your embeddings:**
- fc3 layer = 32-dimensional learned representation
- Better clustering = model learned better features
- Compare with raw features to see improvement

---

## ✨ Summary

1. **Enable UMAP:** Set `umap.enabled: true` in config
2. **Run training:** `python main.py`
3. **View results:** Check `outputs/umap_analysis/`
4. **Analyze:** Load embeddings and compute metrics

**That's it!** UMAP provides beautiful visualizations of your model's learned embeddings. 🚀

---

**Need help?** Check the Troubleshooting section above, or refer to:
- Config documentation for all options
- Examples section for common use cases
- Advanced Usage for customization
