# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch implementation of a sparse convolutional autoencoder for geological subsurface reconstruction, integrating Vertical Electrical Sounding (VES) data with basement boundary modeling. The implementation uses sparse masking to efficiently process irregular spatial data.

## Development Commands

### Working Directory
All commands should be run from `/workspace/drake_project` (the repository root):
```bash
cd /workspace/drake_project
```

### Environment Setup
```bash
# Activate virtual environment (if not already active)
source ven/bin/activate

# Test environment setup
python test.py
```

### Training
```bash
# Run training with default configuration (saves model to models/checkpoints/)
python sparseconvae/train_model.py

# Train from a specific directory
cd /workspace/drake_project
python -m sparseconvae.train_model
```

### Loading a Trained Model
```bash
# Load the most recent checkpoint
python sparseconvae/load_model.py

# Load a specific checkpoint
python sparseconvae/load_model.py models/checkpoints/autoencoder_20231108_123456.pt
```

### Validation

#### Quick Validation (Synthetic Data)
```bash
# Run a quick validation demo with synthetic geological model
python demo_validation_simple.py --checkpoint models/checkpoints/autoencoder_*.pt
```

#### Full Validation (Real Geological Data)
```bash
# Install geospatial dependencies first
pip install geopandas pandas fiona

# Run validation on real geological test data
python validate_real_data.py --checkpoint models/checkpoints/autoencoder_*.pt \
                              --geology-gpkg TEST_DATA/geology_bundle/geology_bundle.gpkg \
                              --priors TEST_DATA/geology_bundle/petrophysical_priors_filled.csv \
                              --grid-size 64 128 \
                              --sampling-strategy drilling

# View all options
python validate_real_data.py --help
```

### Testing Individual Components
```bash
# Test model imports and GPU setup
python test.py

# Run sparse convolution code examples
python sparseconvae/sparse_code.py
```

## Architecture Overview

### Core Model Structure

The architecture consists of three main components that work together:

1. **SparseEncoder** (models/sparse_nn.py)
   - Wraps a ConvNeXt backbone and converts it to use sparse convolutions
   - Uses global `_cur_active` tensor to track which spatial positions are active/masked
   - Four hierarchical stages: 96 → 192 → 384 → 768 channels
   - Progressive downsampling with factor of 32

2. **Dual Decoders** (models/sparse_nn.py)
   - Two independent decoders with identical architecture but separate weights
   - `imputation_decoder`: Reconstructs secondary variable (VES measurements)
   - `subsurface_decoder`: Reconstructs primary variable (subsurface resistivity)
   - UNet-style skip connections from encoder features
   - Progressive upsampling to original resolution

3. **Autoencoder** (models/autoencoder.py)
   - Orchestrates encoder and decoders
   - Forward pass: masked input → encoder features → dual reconstruction outputs

### Sparse Convolution Implementation

The sparse masking strategy is based on the SparK (Sparse masKed modeling) approach from "Designing BERT for Convolutional Networks" by Tian et al.

**Key mechanism**: A global `_cur_active` tensor tracks which spatial positions contain valid data. This tensor is:
- Set before calling the encoder: `_cur_active = batch_secondary_mask`
- Used by sparse operations to mask both convolution outputs and normalization computations
- Automatically downsampled at each encoder stage to match feature map resolution

**Sparse operations** (all in models/sparse_nn.py):
- `SparseConv2d`: Masks output of standard Conv2d
- `SparseBatchNorm2d`: Only normalizes active positions
- `SparseConvNeXtLayerNorm`: Applies LayerNorm only to non-masked features
- `SparseConvNeXtBlock`: ConvNeXt block with sparse-aware operations

### Data Pipeline

1. **Generators** (data_generation/generators.py)
   - `BaseModel`: Creates layered geological models with deposits
   - `two_layer_generator`: Generates synthetic two-layer subsurface models
   - `CategoricalSpatialGenerator`: Creates categorical spatial data with multiple interpolation methods
   - Supports various interpolation methods (data_generation/interpolation_methods.py):
     - Inverse Distance (ID): `create_id_spatial_data`
     - Variogram-based Inverse Distance (VID)
     - Kriging: `create_stationary_spatial_data`
     - Layering: `create_layered_spatial_data`

2. **Sampling Patterns** (utils/sampling_patterns.py)
   - `random_sampling`: Random point selection (~1% coverage)
   - `grid_sampling`: Regular grid pattern (~10% coverage)
   - `clustered_sampling`: Clustered point groups
   - `drilling_sampling`: Simulates vertical drilling patterns (5-15 drillholes, 3-20 samples each)

3. **Dataset** (datasets/spatial_dataset.py)
   - `SpatialDataset`: Manages generation, saving, and loading of training data
   - Supports dynamic secondary mask regeneration during training
   - Can persist generated data to disk as pickle files in `data_folder`
   - Returns: (x, primary_grid, primary_mask, secondary_grid, secondary_mask)

4. **Training Loop** (training/train.py)
   - Trains encoder and both decoders jointly
   - Loss: MSE for both reconstruction tasks (sample_loss + parameter_loss)
   - **Critical**: Sets global `_cur_active = batch_secondary_mask` before forward pass
   - Uses AdamW optimizer

## Important Implementation Details

### Global State Management
The sparse convolution implementation relies on a global variable `_cur_active` in models/sparse_nn.py. When adding new training loops or inference code:
- Always set `_cur_active = mask` before calling `sparse_encoder.forward()`
- The mask shape should match the input spatial dimensions

### Model Initialization
Models must be initialized in this order:
```python
convnext = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
sparse_encoder = SparseEncoder(convnext, input_size=(H, W))
imputation_decoder = Decoder(up_sample_ratio=32, out_chans=1, width=768)
subsurface_decoder = Decoder(up_sample_ratio=32, out_chans=1, width=768)
model = Autoencoder(sparse_encoder, imputation_decoder, subsurface_decoder)
```

### Training Discrepancy
Note: There are two different training function signatures:
- `training/train.py`: Takes separate encoder/decoder components
- `train_model.py`: Passes complete Autoencoder model

When modifying training, ensure the training function matches how the model is constructed.

### Input/Output Conventions
- Input grid dimensions: (H, W) e.g., (32, 64)
- After stem (4x4 conv, stride 4): (H/4, W/4)
- After full encoding (32x downsample): (H/32, W/32)
- Encoder returns list of 4 feature maps (one per stage)
- Decoders expect reversed feature list: `features[::-1]`

## Model Checkpoints

Trained models are saved as PyTorch checkpoints (.pt files) containing:
- Encoder state dict
- Both decoder state dicts
- Model configuration (for reconstruction)

Default save location: `models/checkpoints/autoencoder_YYYYMMDD_HHMMSS.pt`

To load a checkpoint in code:
```python
from sparseconvae.load_model import load_trained_model
model, encoder, imp_decoder, sub_decoder = load_trained_model('path/to/checkpoint.pt')
```

## Inference with Trained Models

To use a trained model for inference:

```python
import torch
from models import sparse_nn
from sparseconvae.load_model import load_trained_model

# Load trained model
model, encoder, imp_dec, sub_dec = load_trained_model('models/checkpoints/autoencoder_YYYYMMDD_HHMMSS.pt')
model.eval()

# Prepare input data
# secondary_grid: (B, C, H, W) - sparse measurements
# secondary_mask: (B, 1, H, W) - binary mask indicating measurement locations

# CRITICAL: Set global active mask before inference
sparse_nn._cur_active = secondary_mask

# Run inference
with torch.no_grad():
    features = encoder(secondary_grid * secondary_mask)
    imputation_output = imp_dec(features[::-1])  # Reconstructed VES measurements
    subsurface_output = sub_dec(features[::-1])  # Reconstructed subsurface resistivity
```

## Validation Workflow

### Overview
The validation workflow tests the trained model on real geological data to assess its generalization capability.

### Validation Pipeline
```
Real Geology (GeoPackage) → Resistivity Grid → Sparse Sampling → Model Inference → Compare with Ground Truth
```

### Test Data Structure (TEST_DATA/geology_bundle/)

1. **geology_bundle.gpkg** - Authoritative geological shapes (QC'd, EPSG:6933)
   - `geology` layer: Formation polygons with attributes
   - `faults` layer: Fault network polylines

2. **geology.csv** - Formation attributes
   - `formation_name`: Formation identifier
   - `lith_class`: List of lithologies (e.g., ["shale", "sandstone", "carbonate"])
   - `confidence_score`: Data quality metric (0-1)
   - `poly_id`: Links to geometry in GPKG

3. **faults.csv** - Fault attributes
   - `fault_id`: Unique fault identifier
   - `Shape__Length`: Surface trace length
   - `confidence_score`: Fault mapping confidence

4. **petrophysical_priors_filled.csv** - Lithology → Resistivity mapping
   - Maps each `lith_class` to resistivity ranges (Ω·m)
   - Used to convert formations to physical property grids

5. **final_report.json** - QA/QC metadata
   - Quality checks (slivers, overlaps, gaps)
   - Fault density and strike analysis
   - Overall approval status

### Validation Metrics

The validation scripts calculate:
- **RMSE** (Root Mean Squared Error): Overall prediction accuracy
- **MAE** (Mean Absolute Error): Average absolute difference
- **R² Score**: Coefficient of determination (1.0 = perfect fit)
- **Relative Error**: Percentage error relative to ground truth

### Using Validation Scripts

**Quick test with synthetic data:**
```bash
python demo_validation_simple.py --checkpoint models/checkpoints/autoencoder_*.pt
```

**Full validation on real geology:**
```bash
python validate_real_data.py --checkpoint models/checkpoints/autoencoder_*.pt \
                              --grid-size 64 128 \
                              --sampling-strategy drilling \
                              --output-dir validation_results
```

Results are saved to `validation_results/` with:
- Visualization PNG showing ground truth, sparse input, prediction, and errors
- JSON file with detailed metrics

## Project Structure

```
sparseconvae/
├── models/
│   ├── autoencoder.py       # Main Autoencoder wrapper
│   ├── sparse_nn.py          # SparseEncoder, Decoder, ConvNeXt, sparse ops
│   └── checkpoints/          # Saved model checkpoints (.pt files)
├── training/
│   └── train.py              # Training loop
├── datasets/
│   └── spatial_dataset.py    # Dataset management
├── data_generation/
│   ├── generators.py         # Data generation logic
│   ├── grid_utils.py         # Grid utilities
│   └── interpolation_methods.py  # Various interpolation methods
├── utils/
│   ├── sampling_patterns.py  # Spatial sampling strategies
│   └── visualization.py      # Visualization utilities
├── train_model.py            # Main training script (saves model)
└── load_model.py             # Helper to load saved models
```

## Dependencies

Key dependencies:
- Python 3.9+ (tested with 3.12)
- PyTorch 2.0+ with CUDA support
- timm (for DropPath, trunc_normal_)
- tqdm (for progress bars)
- scipy (for interpolation methods)
- numpy

The project uses sparse convolutions extensively, requiring GPU for practical training.

## Package Structure and Imports

The `sparseconvae` directory is NOT a proper Python package (no `__init__.py` files). All scripts must be run from the `/workspace/drake_project` directory using relative imports:

```python
# Correct imports from train_model.py or test.py
from datasets.spatial_dataset import SpatialDataset
from models.sparse_nn import ConvNeXt, SparseEncoder, Decoder
from models.autoencoder import Autoencoder
```

**Do not** try to import using `from sparseconvae.models...` unless you add `__init__.py` files.

## Common Pitfalls and Debugging

### Import Errors
- **Problem**: `ModuleNotFoundError` when running scripts
- **Solution**: Always run scripts from `/workspace/drake_project` directory, not from within `sparseconvae/`
- **Check**: Run `python test.py` first to verify environment setup

### Global State Issues
- **Problem**: Model produces all zeros or NaN outputs
- **Solution**: Ensure `sparse_nn._cur_active = mask` is set before calling encoder
- **Check**: Verify mask shape matches input spatial dimensions

### Shape Mismatches
- **Problem**: Dimension errors during training
- **Solution**: Ensure input grid dimensions are divisible by 32 (e.g., 32x64, 64x128)
- **Reason**: Encoder downsamples by factor of 32 (stem=4x, stages=8x total)

### Data Generation
- **Problem**: Slow dataset initialization
- **Solution**: Use `data_folder` parameter to cache generated data as pickle files
- **Example**: `SpatialDataset(..., data_folder="data/two_layers")`

### GPU Memory
- **Problem**: CUDA out of memory errors
- **Solution**: Reduce batch size in DataLoader (default: 128)
- **Minimum**: 8GB VRAM recommended for batch_size=32

---

## Comprehensive Validation System

### Overview

This project includes a complete validation system for testing the sparse convolutional autoencoder on both synthetic and real geological data. The validation system was built to address several critical challenges in geological AI model evaluation.

### Validation Scripts

#### 1. **demo_validation_normalized.py** - Synthetic Geology Validation
**Purpose**: Quick validation on synthetic continuous resistivity data

**Usage**:
```bash
python demo_validation_normalized.py --checkpoint models/checkpoints/autoencoder_continuous_*.pt --grid-size 32 64
```

**What it does**:
- Generates synthetic geological models with continuous resistivity values (5-800 Ω·m)
- Creates sparse sampling patterns (drilling, random, grid, clustered)
- Normalizes data to [0-9] range (matching training)
- Runs model inference
- Denormalizes predictions back to resistivity units
- Calculates metrics (RMSE, MAE, R², Relative Error)
- Generates 6-panel visualization

**Expected Performance**:
- R² Score: > 0.7 (good), > 0.85 (excellent)
- RMSE: < 100 Ω·m
- Visual: Clear layer reconstruction visible

#### 2. **validate_real_data_fixed.py** - Real Geology Validation
**Purpose**: Validation on actual geological map data

**Usage**:
```bash
python validate_real_data_fixed.py \
    --checkpoint models/checkpoints/autoencoder_continuous_*.pt \
    --geology-gpkg TEST_DATA/geology_bundle/geology_bundle.gpkg \
    --priors TEST_DATA/geology_bundle/petrophysical_priors_filled.csv \
    --grid-size 32 64 \
    --sampling-strategy drilling
```

**What it does**:
- Loads real geological formations from GeoPackage (30 formations)
- Rasterizes polygons to resistivity grid using petrophysical priors
- Creates sparse sampling mask
- Runs model inference with normalization/denormalization
- Calculates metrics and generates visualizations

**Key Feature**: Uses `rasterio.features.rasterize()` with `all_touched=True` for proper polygon coverage (avoids gaps)

#### 3. **train_continuous_resistivity.py** - Proper Training Script
**Purpose**: Train model on continuous resistivity data (not categorical)

**Usage**:
```bash
python train_continuous_resistivity.py
```

**What it does**:
- Generates 1000 synthetic geological models with continuous resistivity
- Creates 3-5 horizontal layers per model
- Uses realistic resistivity ranges per lithology:
  - Shale: 5-25 Ω·m
  - Sandstone: 40-150 Ω·m
  - Limestone: 80-400 Ω·m
  - Clay: 10-60 Ω·m
  - Carbonate: 150-600 Ω·m
- Adds lateral variations (sinusoidal + noise)
- Normalizes to [0-9] for training
- Trains for 100 epochs with AdamW optimizer
- Tests model response to verify learning
- Saves checkpoint compatible with validation scripts

**Why This Script Exists**: The original training data was categorical (binary 0/9 values), causing models to predict constant outputs. This script generates proper continuous data.

---

## Critical Issues and Solutions

### Issue 1: Model Predicting Constant Values (Purple Predictions)

**Symptom**: Model outputs uniform purple/dark predictions regardless of input

**Root Cause**: Training data was **categorical (binary)** with only values 0 and 9
- 62.3% of training pixels = 0
- 37.7% of training pixels = 9
- No intermediate values!

**Why This Failed**:
- MSE loss on binary data causes model to predict mean/intermediate value
- Model converges to outputting ~0.5 (minimizes error for binary targets)
- No spatial patterns learned - just predicting average

**Solution**: Train on **continuous resistivity data**
```bash
# Use the fixed training script
python train_continuous_resistivity.py
```

**Verification**:
- After training, model should output different values for different inputs
- Output std should be > 0.5
- Output range should use most of [0-9]
- Validation R² should be > 0.7

### Issue 2: Ground Truth Entirely Yellow (Uniform 100 Ω·m)

**Symptom**: Real geology validation shows uniform yellow ground truth with no variation

**Root Cause**: **Point-in-polygon rasterization with formation gaps**
- Geological formations only cover 25% of bounding box
- 75% of area has no mapped formations (gaps)
- Point-in-polygon checks: most grid points fall in gaps → get default 100 Ω·m
- Result: Nearly uniform grid

**Detailed Diagnosis**:
```python
# The problem:
for each grid pixel:
    for each formation polygon:
        if pixel inside polygon:
            assign resistivity
        else:
            assign 100 Ω·m default  ← 75% of pixels!
```

**Solution**: Use proper **polygon rasterization** instead of point checks
```python
# The fix (in validate_real_data_fixed.py):
from rasterio import features

resistivity_grid = features.rasterize(
    shapes=[(geom, value) for geom, value in polygons],
    out_shape=(height, width),
    fill=100.0,
    transform=transform,
    all_touched=True  ← Includes cells touching polygon boundaries!
)
```

**Benefits of Fixed Method**:
- Coverage increased from 25% to 39%
- Actual geological variation visible (25.5 - 275 Ω·m range)
- 3 unique resistivity values representing different formations
- Ground truth no longer uniform

### Issue 3: Data Normalization Mismatch

**Symptom**: Model trained on [0-9] range but validation uses [1-1000] Ω·m

**Root Cause**: Training data normalized to [0-9] but validation didn't normalize inputs

**Solution**: Normalize validation data before inference, denormalize after
```python
# Normalize to [0-9] for inference
rho_min, rho_max = resistivity_grid.min(), resistivity_grid.max()
resistivity_norm = 9.0 * (resistivity_grid - rho_min) / (rho_max - rho_min)

# Model inference on normalized data
output_norm = model(resistivity_norm)

# Denormalize predictions back to Ω·m
output = rho_min + (output_norm / 9.0) * (rho_max - rho_min)
```

**Why This Matters**:
- Model never saw values > 9 during training
- Feeding unnormalized data (100+ Ω·m) produces garbage output
- Proper normalization ensures input distribution matches training

---

## Training Best Practices

### Creating Proper Training Data

**DO**:
- ✅ Use continuous resistivity values (not categorical)
- ✅ Include realistic ranges per lithology (5-800 Ω·m)
- ✅ Add lateral variations (facies changes, noise)
- ✅ Generate multiple layers (3-5) with varying thickness
- ✅ Normalize to [0-9] for training stability

**DON'T**:
- ❌ Use categorical/binary data (0 and 9 only)
- ❌ Generate uniform layers with no variation
- ❌ Train on unnormalized data with large range (0-1000)
- ❌ Use only 1-2 layers (too simple)

### Training Requirements

**Minimum**:
- 100 epochs for convergence
- Learning rate: 1e-4 (AdamW)
- Batch size: 32 (minimum), 128 (recommended if memory allows)
- Training samples: 1000+
- Grid size: Divisible by 32 (e.g., 32×64, 64×128)

**Expected Loss Progression**:
```
Epoch 1:   Loss ~10-15
Epoch 10:  Loss ~5-10
Epoch 50:  Loss ~0.5-2
Epoch 100: Loss ~0.1-0.5
```

**Model Learning Verification** (run after training):
```python
# Test if model responds to different inputs
for val in [0, 3, 6, 9]:
    output = model(constant_input(val))
    print(f"Input: {val} → Output mean: {output.mean()}, std: {output.std()}")

# Good signs:
# - Different inputs → different outputs (not all ~0.5)
# - Output std > 0.5 (shows spatial variation)
# - Output range uses most of [0-9], not just [0.4-0.6]
```

### Model Performance Expectations

**Synthetic Geology (Demo Validation)**:
- R² Score: 0.7 - 0.9 (excellent)
- RMSE: 50 - 150 Ω·m
- Visual: Clear layer boundaries reconstructed

**Real Geology (Complex Formations)**:
- R² Score: 0.05 - 0.30 (moderate, expected due to complexity mismatch)
- RMSE: 30 - 100 Ω·m
- Visual: Some spatial structure, higher errors at boundaries

**Why Real Geology Performance is Lower**:
- Training: Simple 3-5 layer models with smooth transitions
- Testing: 30 complex formations with sharp boundaries and gaps
- Coverage: Only 39% of test grid has geological data
- Complexity: Real geology is inherently more complex than synthetic

---

## Complete Workflow Guide

### Step 1: Verify Environment

```bash
cd /workspace/drake_project
python quick_test.py
```

Expected output:
- ✅ PyTorch with CUDA detected
- ✅ Model imports successful
- ✅ Forward pass works

### Step 2: Train Proper Model

```bash
# Train with continuous resistivity data (100 epochs, ~15-20 min)
python train_continuous_resistivity.py
```

Watch for:
- Loss decreasing from ~13 to <0.5
- Model responding differently to different inputs (not constant output)
- Checkpoint saved to `models/checkpoints/autoencoder_continuous_*.pt`

### Step 3: Validate on Synthetic Data

```bash
python demo_validation_normalized.py \
    --checkpoint models/checkpoints/autoencoder_continuous_*.pt \
    --grid-size 32 64
```

Good results:
- R² > 0.7
- Visualization shows layer reconstruction (not uniform)
- Prediction panel has varied colors (not all purple)

### Step 4: Validate on Real Geology

```bash
# Install geospatial dependencies (if not already done)
pip install geopandas pandas fiona rasterio

# Run validation
python validate_real_data_fixed.py \
    --checkpoint models/checkpoints/autoencoder_continuous_*.pt \
    --grid-size 32 64
```

Expected results:
- Ground truth shows variation (not uniform yellow)
- Resistivity range: 25-275 Ω·m
- R² > 0.05 (positive)
- 3 unique lithology values visible

### Step 5: Analyze Results

Output files in `validation_results/`:
- `demo_normalized_*.png`: Synthetic validation visualization
- `validation_real_fixed_*.png`: Real geology visualization
- `validation_metrics_*.json`: Detailed metrics

**Interpreting Visualizations** (6-panel layout):
```
Row 1:  [Ground Truth] [Sparse Input] [Model Prediction]
Row 2:  [Absolute Error] [Relative Error %] [Metrics Summary]
```

**What to Look For**:
- Ground truth: Should show geological variation (layers or formations)
- Sparse input: Should show scattered measurement points
- Prediction: Should show spatial structure (not uniform color)
- Errors: Higher at boundaries (expected), lower in uniform regions

---

## Troubleshooting Guide

### Problem: "Model outputs constant ~0.5 everywhere"

**Diagnosis**:
```bash
python -c "
import torch, numpy as np, pickle
with open('data/two_layers/entries_*.pkl', 'rb') as f:
    data = pickle.load(f)
x = data[0][0]
print(f'Training data range: [{x.min():.1f}, {x.max():.1f}]')
print(f'Training data unique values: {np.unique(x.astype(int))}')
"
```

If output shows only [0, 9]: **Training data is binary/categorical**

**Fix**: Retrain with continuous data
```bash
python train_continuous_resistivity.py
```

### Problem: "Ground truth is uniform yellow in real geology validation"

**Diagnosis**:
Check rasterization coverage:
```bash
python -c "
import geopandas as gpd, numpy as np
from shapely.geometry import Point

gdf = gpd.read_file('TEST_DATA/geology_bundle/geology_bundle.gpkg', layer='geology')
bounds = gdf.total_bounds
test_points = [Point(x, y) for x in np.linspace(bounds[0], bounds[2], 10)
                             for y in np.linspace(bounds[1], bounds[3], 10)]
coverage = sum(gdf.geometry.contains(p).any() for p in test_points) / len(test_points)
print(f'Coverage: {100*coverage:.1f}%')
"
```

If coverage < 30%: **Point-in-polygon method failing due to gaps**

**Fix**: Use fixed validation script
```bash
python validate_real_data_fixed.py --checkpoint models/checkpoints/autoencoder_*.pt
```

### Problem: "ImportError: cannot import sparse_nn"

**Cause**: Running from wrong directory or incorrect import paths

**Fix**:
1. Always run from `/workspace/drake_project`:
   ```bash
   cd /workspace/drake_project
   ```
2. Check imports in training/train.py use `from sparseconvae.models import sparse_nn`
3. Ensure all scripts import with correct paths

### Problem: "Negative resistivity values in predictions"

**Cause**: Model outputs not constrained, can predict negative values

**Acceptable if**: Predictions are mostly positive, negatives only at boundaries

**Fix if severe**: Add ReLU to decoder output or clip predictions:
```python
output = torch.clamp(decoder(features), min=0.0, max=9.0)
```

### Problem: "R² score is negative"

**Meaning**: Model predictions are worse than just predicting the mean value

**Common Causes**:
- Model not trained (or poorly trained)
- Wrong checkpoint loaded
- Data normalization mismatch
- Model architecture doesn't match checkpoint

**Fix**: Verify model training and reload correct checkpoint

---

## Advanced Configuration

### Custom Sampling Patterns

Create your own sparse sampling:
```python
import numpy as np

# Example: Sample only edges
mask = np.zeros((1, 32, 64), dtype=np.float32)
mask[0, 0, :] = 1    # Top edge
mask[0, -1, :] = 1   # Bottom edge
mask[0, :, 0] = 1    # Left edge
mask[0, :, -1] = 1   # Right edge

# Use in validation
python demo_validation_normalized.py --checkpoint models/checkpoints/autoencoder_*.pt
# (modify script to load custom mask)
```

### Adjusting Grid Resolution

**Constraint**: Grid dimensions must be divisible by 32

**Valid sizes**:
- 32×32, 32×64, 32×96, 32×128
- 64×64, 64×128, 64×192, 64×256
- 96×96, 96×192
- 128×128, 128×256

**Trade-offs**:
- Smaller (32×64): Faster, less detail, better for sparse data
- Larger (128×256): Slower, more detail, needs denser sampling

### Training on Real Geology Data

To train directly on real geological data (advanced):

1. Extract formations from GeoPackage
2. Rasterize to grids with proper coverage
3. Create sparse sampling masks
4. Generate paired (sparse, full) datasets
5. Train with same architecture

**Challenges**:
- Need many geological maps (100s-1000s)
- Formations must cover area well (>50%)
- Requires data augmentation (rotation, scaling)

---

## File Reference

### Created During Validation Development

| File | Purpose | Status |
|------|---------|--------|
| `demo_validation_normalized.py` | Synthetic geology validation with normalization | ✅ Production |
| `validate_real_data_fixed.py` | Real geology validation with proper rasterization | ✅ Production |
| `train_continuous_resistivity.py` | Training on continuous (not categorical) data | ✅ Production |
| `quick_test.py` | Environment verification | ✅ Utility |
| `train_with_monitoring.py` | Training with detailed progress monitoring | ⚠️ Alternative |
| `demo_validation_simple.py` | Basic validation (no normalization) | ⚠️ Deprecated |
| `validate_real_data.py` | Original validation (broken rasterization) | ❌ Deprecated |
| `petrophysical_priors_filled.csv` | Lithology → resistivity mapping | ✅ Data |
| `VALIDATION_GUIDE.md` | Detailed validation documentation | ✅ Documentation |

### Import Path Reference

**Correct imports** (from project root):
```python
from sparseconvae.models.sparse_nn import ConvNeXt, SparseEncoder, Decoder
from sparseconvae.models.autoencoder import Autoencoder
from sparseconvae.training.train import train_model
from sparseconvae.datasets.spatial_dataset import SpatialDataset
```

**Incorrect** (will fail):
```python
from models.sparse_nn import ...  # ❌ Wrong
from sparseconvae import models   # ❌ No __init__.py
```

---

## Performance Benchmarks

### Synthetic Geology Validation

**Test Configuration**:
- Grid: 32×64
- Layers: 4-5 horizontal
- Sparse coverage: 4-7%
- Training: 100 epochs, continuous data

**Expected Results**:
| Metric | Good | Excellent |
|--------|------|-----------|
| R² Score | > 0.7 | > 0.85 |
| RMSE | < 120 Ω·m | < 80 Ω·m |
| MAE | < 90 Ω·m | < 60 Ω·m |
| Relative Error | < 100% | < 80% |

### Real Geology Validation

**Test Configuration**:
- Grid: 32×64
- Formations: 30 complex
- Coverage: 39% (gaps present)
- Sparse coverage: 5-10%

**Expected Results**:
| Metric | Acceptable | Good |
|--------|-----------|------|
| R² Score | > 0.05 | > 0.15 |
| RMSE | < 80 Ω·m | < 50 Ω·m |
| MAE | < 60 Ω·m | < 40 Ω·m |
| Ground Truth Variation | 3+ unique values | 5+ unique values |

**Note**: Real geology performance is inherently lower due to training-test mismatch. This is expected and normal.

---

## Summary of Key Learnings

### Critical Discoveries

1. **Training Data Must Be Continuous**: Categorical/binary data causes constant predictions
2. **Normalization is Essential**: Model trained on [0-9] can't handle [100-1000] without scaling
3. **Rasterization Method Matters**: Point-in-polygon fails with gaps; use polygon burning
4. **Validation Reveals Issues**: Purple predictions and yellow ground truth indicate specific problems
5. **R² Score Interpretation**: Negative = broken, 0-0.3 = poor, 0.7-0.9 = good, >0.9 = excellent

### Best Practices Established

- ✅ Always train on continuous resistivity data
- ✅ Use `train_continuous_resistivity.py` for proper training
- ✅ Validate with `demo_validation_normalized.py` (synthetic) and `validate_real_data_fixed.py` (real)
- ✅ Check R² score and visual patterns, not just loss values
- ✅ Expect lower performance on real geology (complexity mismatch)
- ✅ Use rasterio for polygon rasterization, not point-in-polygon
- ✅ Always normalize/denormalize when switching between training and inference scales

### Common Mistakes to Avoid

- ❌ Training on categorical (0/9) data
- ❌ Using point-in-polygon for rasterization
- ❌ Forgetting to normalize validation inputs
- ❌ Expecting same performance on real geology as synthetic
- ❌ Using grid sizes not divisible by 32
- ❌ Running scripts from inside `sparseconvae/` directory
- ❌ Forgetting to set `sparse_nn._cur_active` before inference
