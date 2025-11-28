"""
Fixed validation script with proper geology rasterization using burning method.
"""

import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

import geopandas as gpd
import pandas as pd
from rasterio import features
from rasterio.transform import from_bounds

from sparseconvae.load_model import load_trained_model
from sparseconvae.models import sparse_nn


def load_petrophysical_priors(priors_file):
    """Load lithology to resistivity mapping."""
    df = pd.read_csv(priors_file)
    priors = {}
    for _, row in df.iterrows():
        lith = row['lith_class']
        priors[lith] = {
            'resistivity_min': row['resistivity_min_ohm_m'],
            'resistivity_max': row['resistivity_max_ohm_m'],
            'resistivity_mean': (row['resistivity_min_ohm_m'] + row['resistivity_max_ohm_m']) / 2
        }
    return priors


def parse_lithology_list(lith_str):
    """Parse lithology string."""
    import ast
    try:
        return ast.literal_eval(lith_str)
    except:
        return [lith_str] if isinstance(lith_str, str) else lith_str


def get_formation_resistivity(lith_class_str, priors):
    """Get resistivity for a formation."""
    lith_list = parse_lithology_list(lith_class_str)

    # Get dominant (first) lithology
    if isinstance(lith_list, list) and len(lith_list) > 0:
        dominant_lith = lith_list[0]
    elif isinstance(lith_class_str, str):
        dominant_lith = lith_class_str
    else:
        dominant_lith = str(lith_class_str)

    # Ensure it's a string for dict lookup
    dominant_lith = str(dominant_lith).strip()

    if dominant_lith in priors:
        return priors[dominant_lith]['resistivity_mean']
    else:
        print(f"Warning: Lithology '{dominant_lith}' not in priors, using default")
        return 100.0


def geology_to_grid_fixed(geology_gdf, priors, grid_size=(64, 128)):
    """
    Convert geological polygons to resistivity grid using proper rasterization.

    This version uses rasterio.features.rasterize instead of point-in-polygon,
    which correctly handles polygon coverage without gaps.
    """
    height, width = grid_size

    # Get bounds from actual polygon extent (not bounding box)
    bounds = geology_gdf.total_bounds
    minx, miny, maxx, maxy = bounds

    print(f"Geology bounds: ({minx:.0f}, {miny:.0f}) to ({maxx:.0f}, {maxy:.0f})")

    # Create affine transform for the grid
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    # Assign resistivity value to each formation
    geology_gdf = geology_gdf.copy()
    geology_gdf['resistivity'] = geology_gdf['lith_class'].apply(
        lambda lith: get_formation_resistivity(lith, priors)
    )

    print(f"\nResistivity values assigned:")
    print(f"  Min: {geology_gdf['resistivity'].min():.1f} Ω·m")
    print(f"  Max: {geology_gdf['resistivity'].max():.1f} Ω·m")
    print(f"  Unique values: {geology_gdf['resistivity'].nunique()}")

    # Create list of (geometry, value) pairs for rasterization
    shapes = [(geom, value) for geom, value in zip(geology_gdf.geometry, geology_gdf['resistivity'])]

    # Rasterize: burn polygon values into grid
    print(f"\nRasterizing {len(shapes)} formations to {height}×{width} grid...")
    resistivity_grid = features.rasterize(
        shapes=shapes,
        out_shape=(height, width),
        fill=100.0,  # Default for areas with no polygons
        transform=transform,
        all_touched=True,  # Include cells that touch polygons
        dtype=np.float32
    )

    # Check coverage
    non_default = (resistivity_grid != 100.0).sum()
    coverage = 100 * non_default / (height * width)

    print(f"\nRasterization results:")
    print(f"  Grid size: {height}×{width}")
    print(f"  Resistivity range: {resistivity_grid.min():.1f} - {resistivity_grid.max():.1f} Ω·m")
    print(f"  Non-default cells: {non_default}/{height*width} ({coverage:.1f}%)")
    print(f"  Unique values: {len(np.unique(resistivity_grid))}")

    if coverage < 30:
        print(f"\n⚠️  WARNING: Low coverage ({coverage:.1f}%)")
        print(f"   Most of the grid is outside mapped formations")
        print(f"   Consider cropping to formation extent or using smaller grid")

    extent = (minx, maxx, miny, maxy)
    return resistivity_grid, extent


def create_sparse_sampling_mask(grid_size, sampling_strategy='drilling', num_points=None, seed=42):
    """Create sparse sampling mask."""
    import random
    random.seed(seed)
    np.random.seed(seed)

    height, width = grid_size
    mask = np.zeros((1, height, width), dtype=np.float32)

    if sampling_strategy == 'drilling':
        num_drillholes = num_points if num_points else random.randint(5, 15)
        for _ in range(num_drillholes):
            x = random.randint(0, width - 1)
            num_samples = random.randint(3, min(20, height))
            y_positions = random.sample(range(height), min(num_samples, height))
            for y in y_positions:
                mask[0, y, x] = 1
    elif sampling_strategy == 'random':
        coverage = 0.01 if num_points is None else num_points / (height * width)
        mask[0] = (np.random.rand(height, width) < coverage).astype(np.float32)
    elif sampling_strategy == 'grid':
        coverage = 0.1 if num_points is None else num_points / (height * width)
        step = max(1, int(min(grid_size) / (coverage * height * width) ** 0.5))
        mask[0, ::step, ::step] = 1

    num_sampled = int(mask.sum())
    coverage_pct = 100 * num_sampled / (height * width)
    print(f"Sparse sampling: {num_sampled} points ({coverage_pct:.2f}% coverage)")

    return mask


def run_model_inference(encoder, decoder, resistivity_grid, sparse_mask, device='cuda'):
    """Run model inference."""
    encoder.eval()
    decoder.eval()

    # Normalize resistivity to [0, 9] range
    rho_min = resistivity_grid.min()
    rho_max = resistivity_grid.max()
    resistivity_norm = 9.0 * (resistivity_grid - rho_min) / (rho_max - rho_min + 1e-6)

    # To tensors
    rho_tensor = torch.from_numpy(resistivity_norm).float().unsqueeze(0).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(sparse_mask).float().unsqueeze(0).to(device)

    sparse_input = rho_tensor * mask_tensor
    sparse_nn._cur_active = mask_tensor

    with torch.no_grad():
        features = encoder(sparse_input)
        reconstructed_norm = decoder(features[::-1])

    # Denormalize
    reconstructed_norm_np = reconstructed_norm.squeeze().cpu().numpy()
    reconstructed = rho_min + (reconstructed_norm_np / 9.0) * (rho_max - rho_min)

    sparse_input_np = sparse_input.squeeze().cpu().numpy()
    sparse_input_denorm = rho_min + (sparse_input_np / 9.0) * (rho_max - rho_min)

    return reconstructed, sparse_input_denorm


def calculate_metrics(ground_truth, prediction):
    """Calculate metrics."""
    gt = ground_truth.flatten()
    pred = prediction.flatten()

    valid_idx = np.isfinite(gt) & np.isfinite(pred)
    gt = gt[valid_idx]
    pred = pred[valid_idx]

    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    mae = np.mean(np.abs(gt - pred))
    ss_res = np.sum((gt - pred) ** 2)
    ss_tot = np.sum((gt - gt.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    relative_error = np.mean(np.abs((gt - pred) / (gt + 1e-6))) * 100

    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'relative_error_pct': relative_error}


def visualize_results(ground_truth, sparse_input, prediction, sparse_mask, metrics, extent, output_dir):
    """Visualize results."""
    Path(output_dir).mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    vmin = min(ground_truth.min(), prediction.min())
    vmax = max(ground_truth.max(), prediction.max())

    # Ground Truth
    im1 = axes[0, 0].imshow(ground_truth, cmap='viridis', aspect='auto', extent=extent, vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('Ground Truth Resistivity (Ω·m)', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=axes[0, 0])

    # Sparse Input
    sparse_display = np.ma.masked_where(sparse_mask[0] == 0, sparse_input)
    im2 = axes[0, 1].imshow(sparse_display, cmap='viridis', aspect='auto', extent=extent, vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f'Sparse Measurements ({int(sparse_mask.sum())} points)', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=axes[0, 1])

    # Prediction
    im3 = axes[0, 2].imshow(prediction, cmap='viridis', aspect='auto', extent=extent, vmin=vmin, vmax=vmax)
    axes[0, 2].set_title('Model Prediction', fontsize=12, fontweight='bold')
    plt.colorbar(im3, ax=axes[0, 2])

    # Absolute Error
    error = np.abs(ground_truth - prediction)
    im4 = axes[1, 0].imshow(error, cmap='hot', aspect='auto', extent=extent)
    axes[1, 0].set_title('Absolute Error', fontsize=12, fontweight='bold')
    plt.colorbar(im4, ax=axes[1, 0])

    # Relative Error
    relative_error = np.abs((ground_truth - prediction) / (ground_truth + 1e-6)) * 100
    relative_error = np.clip(relative_error, 0, 200)
    im5 = axes[1, 1].imshow(relative_error, cmap='hot', aspect='auto', extent=extent)
    axes[1, 1].set_title('Relative Error (%)', fontsize=12, fontweight='bold')
    plt.colorbar(im5, ax=axes[1, 1])

    # Metrics
    axes[1, 2].axis('off')
    metrics_text = f"""
    VALIDATION METRICS
    {'='*30}

    RMSE: {metrics['rmse']:.2f} Ω·m
    MAE: {metrics['mae']:.2f} Ω·m
    R² Score: {metrics['r2']:.4f}
    Relative Error: {metrics['relative_error_pct']:.2f}%

    {'='*30}

    Coverage: {100*sparse_mask.sum()/(sparse_mask.shape[1]*sparse_mask.shape[2]):.2f}%
    Grid: {ground_truth.shape[0]}×{ground_truth.shape[1]}
    """
    axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=11, family='monospace', verticalalignment='center')

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(output_dir) / f'validation_real_fixed_{timestamp}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved: {output_file}")
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--geology-gpkg', type=str, default='TEST_DATA/geology_bundle/geology_bundle.gpkg')
    parser.add_argument('--priors', type=str, default='TEST_DATA/geology_bundle/petrophysical_priors_filled.csv')
    parser.add_argument('--grid-size', type=int, nargs=2, default=[32, 64])
    parser.add_argument('--sampling-strategy', type=str, default='drilling')
    parser.add_argument('--output-dir', type=str, default='validation_results')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    print("="*70)
    print("REAL GEOLOGY VALIDATION (FIXED RASTERIZATION)")
    print("="*70)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"\nDevice: {device}")

    print(f"\n{'='*70}")
    print("Step 1: Loading model...")
    print(f"{'='*70}")
    model, encoder, imp_decoder, sub_decoder = load_trained_model(args.checkpoint, device)

    print(f"\n{'='*70}")
    print("Step 2: Loading geology...")
    print(f"{'='*70}")
    geology_gdf = gpd.read_file(args.geology_gpkg, layer='geology')
    print(f"✓ Loaded {len(geology_gdf)} formations")

    priors = load_petrophysical_priors(args.priors)
    print(f"✓ Loaded priors for {len(priors)} lithologies")

    print(f"\n{'='*70}")
    print("Step 3: Rasterizing geology (FIXED METHOD)...")
    print(f"{'='*70}")
    resistivity_grid, extent = geology_to_grid_fixed(geology_gdf, priors, tuple(args.grid_size))

    print(f"\n{'='*70}")
    print("Step 4: Creating sparse sampling...")
    print(f"{'='*70}")
    sparse_mask = create_sparse_sampling_mask(tuple(args.grid_size), args.sampling_strategy)

    print(f"\n{'='*70}")
    print("Step 5: Running inference...")
    print(f"{'='*70}")
    reconstructed, sparse_input = run_model_inference(encoder, sub_decoder, resistivity_grid, sparse_mask, device)
    print("✓ Reconstruction complete")

    print(f"\n{'='*70}")
    print("Step 6: Calculating metrics...")
    print(f"{'='*70}")
    metrics = calculate_metrics(resistivity_grid, reconstructed)
    print(f"\nRMSE: {metrics['rmse']:.2f} Ω·m")
    print(f"MAE: {metrics['mae']:.2f} Ω·m")
    print(f"R²: {metrics['r2']:.4f}")
    print(f"Relative Error: {metrics['relative_error_pct']:.2f}%")

    print(f"\n{'='*70}")
    print("Step 7: Visualizing...")
    print(f"{'='*70}")
    visualize_results(resistivity_grid, sparse_input, reconstructed, sparse_mask, metrics, extent, args.output_dir)

    print(f"\n{'='*70}")
    print("VALIDATION COMPLETE!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
