# Validation Guide for Sparse Convolutional Autoencoder

## Overview

This guide explains how to validate your trained sparse convolutional autoencoder model using real geological test data.

## What is Validation?

**Validation** tests whether your model (trained on synthetic data) can accurately reconstruct real subsurface geology from sparse measurements.

### The Big Picture

```
┌─────────────────────────────────────────────────────────────────────┐
│  TRAINING (What You've Done)                                        │
│  ────────────────────────────────────────────────────────────────   │
│  Synthetic 2-layer models → Your Model → Learned reconstruction     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  VALIDATION (What We're Doing Now)                                  │
│  ────────────────────────────────────────────────────────────────   │
│  Real geology data → Your Model → Does it still work?               │
└─────────────────────────────────────────────────────────────────────┘
```

## Files You've Been Given

### TEST_DATA/geology_bundle/ Directory

| File | What It Contains | How It's Used |
|------|------------------|---------------|
| **geology_bundle.gpkg** | Surface geological map (formations, faults) | Source of geological shapes |
| **geology.csv** | Formation properties (lithology, age, confidence) | Attributes for each formation |
| **faults.csv** | Fault properties (length, confidence) | Fault network data |
| **petrophysical_priors_filled.csv** | Lithology → Resistivity mapping | Converts rocks to resistivity values |
| **final_report.json** | QA/QC metrics | Data quality assurance |

### 3D Geological Test.kml

This is a **boundary file** showing the study area (Camp Stanley, Texas - USGS site). It defines the geographic extent but doesn't contain 3D layers itself. The actual 3D geological model data would come from the USGS report referenced in the messages.

## How Validation Works

### Step-by-Step Process

#### 1. **Load Geological Map**
```
geology_bundle.gpkg → Load formations and their boundaries
```

#### 2. **Convert to Resistivity Grid**
```
Formation "Shale" → Look up in petrophysical_priors.csv → Resistivity = 1-50 Ω·m
Formation "Sandstone" → Look up → Resistivity = 50-500 Ω·m
...
→ Create 2D/3D resistivity grid
```

#### 3. **Create Sparse Sampling**
```
Full resistivity grid → Sample only 10-20 points (simulate drilling)
→ Sparse measurements (like VES data)
```

#### 4. **Run Your Model**
```
Sparse measurements → Your Trained Autoencoder → Reconstructed resistivity grid
```

#### 5. **Compare & Calculate Metrics**
```
Reconstructed grid vs. Original grid → Calculate RMSE, MAE, R², etc.
```

## Using the Validation Scripts

### Option 1: Quick Demo (No Additional Setup)

Test your model with **synthetic geology** that mimics real structures:

```bash
cd /workspace/drake_project
python demo_validation_simple.py --checkpoint models/checkpoints/autoencoder_*.pt
```

**Output:**
- Visualization showing ground truth, sparse input, prediction, and errors
- Metrics: RMSE, MAE, R², Relative Error
- Saved to `validation_results/demo_validation_YYYYMMDD_HHMMSS.png`

**When to use:** Quick sanity check to see if the model works

---

### Option 2: Full Validation on Real Geology

Test your model with the **actual geological test data** provided:

#### Step 1: Install Dependencies
```bash
pip install geopandas pandas fiona
```

#### Step 2: Run Validation
```bash
python validate_real_data.py \
    --checkpoint models/checkpoints/autoencoder_*.pt \
    --geology-gpkg TEST_DATA/geology_bundle/geology_bundle.gpkg \
    --geology-csv TEST_DATA/geology_bundle/geology.csv \
    --priors TEST_DATA/geology_bundle/petrophysical_priors_filled.csv \
    --grid-size 64 128 \
    --sampling-strategy drilling \
    --num-points 100 \
    --output-dir validation_results
```

**Parameters Explained:**
- `--checkpoint`: Your trained model file
- `--geology-gpkg`: GeoPackage with geological polygons
- `--priors`: Lithology → resistivity mapping (already filled for you)
- `--grid-size`: Must be divisible by 32 (e.g., 64×128, 32×64, 128×256)
- `--sampling-strategy`: How to sample sparse points
  - `drilling`: Vertical drill holes (most realistic)
  - `random`: Random points
  - `grid`: Regular grid pattern
  - `clustered`: Clustered sampling
- `--num-points`: Number of sparse measurements

**When to use:** Final validation for publication, reports, or assessing real-world performance

---

## Understanding the Output

### Visualization

The validation scripts create a 6-panel figure:

```
┌─────────────────┬─────────────────┬─────────────────┐
│ Ground Truth    │ Sparse Input    │ Model Prediction│
│ (Full geology)  │ (10-20 points)  │ (Reconstructed) │
├─────────────────┼─────────────────┼─────────────────┤
│ Absolute Error  │ Relative Error  │ Metrics Summary │
│ (|GT - Pred|)   │ (% difference)  │ RMSE, MAE, R²   │
└─────────────────┴─────────────────┴─────────────────┘
```

### Metrics Interpretation

| Metric | What It Means | Good Value | Bad Value |
|--------|---------------|------------|-----------|
| **RMSE** | Average error magnitude | < 50 Ω·m | > 200 Ω·m |
| **MAE** | Typical error | < 30 Ω·m | > 150 Ω·m |
| **R²** | Correlation quality | > 0.7 | < 0.3 |
| **Relative Error** | % error | < 20% | > 50% |

**Note:** "Good" values depend on your application. Geological models often have 20-30% uncertainty naturally.

---

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'geopandas'"
**Solution:**
```bash
pip install geopandas pandas fiona
```

### Problem: "Grid size must be divisible by 32"
**Solution:** Use grid sizes like:
- 32×64
- 64×128
- 128×256

The model downsamples by 32×, so dimensions must be multiples of 32.

### Problem: "Model produces all zeros"
**Solution:** Check that `sparse_nn._cur_active` is being set. This is handled automatically in the validation scripts.

### Problem: Package installation taking too long
**Solution:** The first installation can take 5-10 minutes. Check status:
```bash
pip list | grep geopandas
```

---

## Advanced Usage

### Custom Sampling Patterns

Create your own sampling mask:

```python
import numpy as np

# Create custom mask (1 = sampled, 0 = not sampled)
height, width = 64, 128
mask = np.zeros((1, height, width))

# Example: Sample only the top half
mask[0, :32, :] = 1

# Save and use in validation
np.save('custom_mask.npy', mask)
```

### Multiple Sampling Strategies

Test different sampling densities:

```bash
for num_points in 50 100 200 500; do
    python validate_real_data.py \
        --checkpoint models/checkpoints/autoencoder_*.pt \
        --num-points $num_points \
        --output-dir validation_results/points_${num_points}
done
```

### Batch Validation

Run validation on multiple checkpoints:

```bash
for checkpoint in models/checkpoints/*.pt; do
    echo "Validating: $checkpoint"
    python demo_validation_simple.py --checkpoint $checkpoint
done
```

---

## Next Steps

1. **Run the quick demo** to verify everything works:
   ```bash
   python demo_validation_simple.py --checkpoint models/checkpoints/autoencoder_*.pt
   ```

2. **Install geospatial packages** for full validation:
   ```bash
   pip install geopandas pandas fiona
   ```

3. **Run full validation** on the test data:
   ```bash
   python validate_real_data.py --checkpoint models/checkpoints/autoencoder_*.pt
   ```

4. **Analyze results** in `validation_results/` directory

5. **Iterate:** If performance is poor, consider:
   - Training longer (more epochs)
   - Training on more diverse synthetic data
   - Adjusting model architecture
   - Using different sampling patterns

---

## Questions?

- Check `CLAUDE.md` for general project documentation
- See `validate_real_data.py --help` for all options
- Review the validation script source code for implementation details

---

## Summary

**What:** Validate your sparse autoencoder on real geological data

**Why:** Assess whether your model generalizes beyond synthetic training data

**How:** Two scripts provided:
- `demo_validation_simple.py` - Quick test (no dependencies)
- `validate_real_data.py` - Full validation on real geology (requires geopandas)

**Expected Outcome:** Visualization showing how well your model reconstructs sparse measurements, with quantitative metrics (RMSE, MAE, R²)
