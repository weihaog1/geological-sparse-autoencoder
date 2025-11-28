"""
Validation with proper normalization to match training data range.
This version normalizes resistivity values to [0, 9] to match the training data.
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from sparseconvae.load_model import load_trained_model
from sparseconvae.models import sparse_nn


def create_synthetic_geology(grid_size=(64, 128), num_layers=5, seed=42):
    """Create synthetic geological model with normalized values."""
    np.random.seed(seed)
    height, width = grid_size

    resistivity = np.zeros(grid_size)
    layer_boundaries = np.sort(np.random.randint(0, height, num_layers-1))
    layer_boundaries = np.concatenate([[0], layer_boundaries, [height]])

    lithology_resistivities = [
        (5, 20),
        (50, 150),
        (100, 500),
        (10, 50),
        (200, 800),
    ]

    for i in range(len(layer_boundaries)-1):
        start = layer_boundaries[i]
        end = layer_boundaries[i+1]

        lith_idx = i % len(lithology_resistivities)
        rho_min, rho_max = lithology_resistivities[lith_idx]
        base_rho = np.random.uniform(rho_min, rho_max)

        for y in range(start, end):
            x_variation = np.sin(np.linspace(0, 4*np.pi, width)) * 0.2 + 1.0
            y_noise = np.random.randn(width) * 0.1 + 1.0
            resistivity[y, :] = base_rho * x_variation * y_noise

    resistivity = np.clip(resistivity, 1, 1000)

    print(f"Created synthetic geology: {num_layers} layers")
    print(f"Resistivity range (raw): {resistivity.min():.1f} - {resistivity.max():.1f} Ω·m")

    # Store original for denormalization
    resistivity_min = resistivity.min()
    resistivity_max = resistivity.max()

    # NORMALIZE to [0, 9] to match training data
    resistivity_normalized = 9.0 * (resistivity - resistivity_min) / (resistivity_max - resistivity_min)

    print(f"Normalized to training range: {resistivity_normalized.min():.1f} - {resistivity_normalized.max():.1f}")

    return resistivity, resistivity_normalized, resistivity_min, resistivity_max


def create_drilling_mask(grid_size, num_drillholes=10, seed=42):
    """Create drilling sampling pattern."""
    import random
    random.seed(seed)
    np.random.seed(seed)

    height, width = grid_size
    mask = np.zeros((1, height, width), dtype=np.float32)

    for _ in range(num_drillholes):
        x = random.randint(0, width - 1)
        num_samples = random.randint(5, min(15, height))
        y_positions = random.sample(range(height), min(num_samples, height))
        for y in y_positions:
            mask[0, y, x] = 1

    num_points = int(mask.sum())
    coverage = 100 * num_points / (height * width)
    print(f"Drilling pattern: {num_drillholes} holes, {num_points} measurements ({coverage:.2f}% coverage)")

    return mask


def run_inference(model, encoder, imp_decoder, sub_decoder,
                 ground_truth_normalized, sparse_mask, device='cuda'):
    """Run model inference on normalized sparse measurements."""
    model.eval()

    # Use normalized data
    gt_tensor = torch.from_numpy(ground_truth_normalized).float().unsqueeze(0).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(sparse_mask).float().unsqueeze(0).to(device)

    sparse_input = gt_tensor * mask_tensor

    sparse_nn._cur_active = mask_tensor

    with torch.no_grad():
        features = encoder(sparse_input)
        reconstructed_tensor = sub_decoder(features[::-1])

    reconstructed = reconstructed_tensor.squeeze().cpu().numpy()
    sparse_input_np = sparse_input.squeeze().cpu().numpy()

    return reconstructed, sparse_input_np


def denormalize(data_normalized, data_min, data_max):
    """Convert normalized [0, 9] data back to original resistivity range."""
    return data_min + (data_normalized / 9.0) * (data_max - data_min)


def calculate_metrics(ground_truth, prediction):
    """Calculate validation metrics."""
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

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'relative_error_pct': relative_error
    }


def visualize_results(ground_truth, sparse_input, prediction, sparse_mask,
                     metrics, output_dir='validation_results'):
    """Visualize results (all in original resistivity units)."""
    Path(output_dir).mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    vmin = min(ground_truth.min(), prediction.min())
    vmax = max(ground_truth.max(), prediction.max())

    # Ground Truth
    im1 = axes[0, 0].imshow(ground_truth, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('Ground Truth Resistivity (Ω·m)', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=axes[0, 0])

    # Sparse Input
    sparse_display = np.ma.masked_where(sparse_mask[0] == 0, sparse_input)
    im2 = axes[0, 1].imshow(sparse_display, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f'Sparse Measurements ({int(sparse_mask.sum())} points)', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=axes[0, 1])

    # Prediction
    im3 = axes[0, 2].imshow(prediction, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    axes[0, 2].set_title('Model Reconstruction (Denormalized)', fontsize=12, fontweight='bold')
    plt.colorbar(im3, ax=axes[0, 2])

    # Absolute Error
    error = np.abs(ground_truth - prediction)
    im4 = axes[1, 0].imshow(error, cmap='hot', aspect='auto')
    axes[1, 0].set_title('Absolute Error', fontsize=12, fontweight='bold')
    plt.colorbar(im4, ax=axes[1, 0])

    # Relative Error
    relative_error = np.abs((ground_truth - prediction) / (ground_truth + 1e-6)) * 100
    relative_error = np.clip(relative_error, 0, 200)
    im5 = axes[1, 1].imshow(relative_error, cmap='hot', aspect='auto')
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

    Grid: {ground_truth.shape[0]}×{ground_truth.shape[1]}
    Coverage: {100*sparse_mask.sum()/(sparse_mask.shape[1]*sparse_mask.shape[2]):.2f}%

    Note: Values normalized to
    [0,9] during inference, then
    denormalized for comparison
    """
    axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
                   verticalalignment='center')

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(output_dir) / f'demo_normalized_{timestamp}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Normalized validation demo')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--grid-size', type=int, nargs=2, default=[64, 128])
    parser.add_argument('--num-drillholes', type=int, default=10)
    parser.add_argument('--output-dir', type=str, default='validation_results')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    print("="*70)
    print("NORMALIZED VALIDATION - Matching Training Data Range")
    print("="*70)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"\nDevice: {device}")

    print(f"\n{'='*70}")
    print("Step 1: Loading trained model...")
    print(f"{'='*70}")
    model, encoder, imp_decoder, sub_decoder = load_trained_model(args.checkpoint, device)

    print(f"\n{'='*70}")
    print("Step 2: Creating synthetic geology...")
    print(f"{'='*70}")
    ground_truth_raw, ground_truth_norm, gt_min, gt_max = create_synthetic_geology(
        tuple(args.grid_size), num_layers=5, seed=args.seed
    )

    print(f"\n{'='*70}")
    print("Step 3: Creating sparse sampling...")
    print(f"{'='*70}")
    sparse_mask = create_drilling_mask(tuple(args.grid_size), args.num_drillholes, args.seed)

    print(f"\n{'='*70}")
    print("Step 4: Running inference (on NORMALIZED data)...")
    print(f"{'='*70}")
    reconstructed_norm, sparse_input_norm = run_inference(
        model, encoder, imp_decoder, sub_decoder,
        ground_truth_norm, sparse_mask, device
    )
    print(f"✓ Reconstruction complete (normalized range: {reconstructed_norm.min():.2f} - {reconstructed_norm.max():.2f})")

    print(f"\n{'='*70}")
    print("Step 5: Denormalizing predictions...")
    print(f"{'='*70}")
    reconstructed_raw = denormalize(reconstructed_norm, gt_min, gt_max)
    sparse_input_raw = denormalize(sparse_input_norm, gt_min, gt_max)
    print(f"✓ Denormalized to resistivity range: {reconstructed_raw.min():.2f} - {reconstructed_raw.max():.2f} Ω·m")

    print(f"\n{'='*70}")
    print("Step 6: Calculating metrics...")
    print(f"{'='*70}")
    metrics = calculate_metrics(ground_truth_raw, reconstructed_raw)

    print("\nVALIDATION METRICS:")
    print(f"  RMSE: {metrics['rmse']:.2f} Ω·m")
    print(f"  MAE: {metrics['mae']:.2f} Ω·m")
    print(f"  R² Score: {metrics['r2']:.4f}")
    print(f"  Relative Error: {metrics['relative_error_pct']:.2f}%")

    print(f"\n{'='*70}")
    print("Step 7: Generating visualization...")
    print(f"{'='*70}")
    visualize_results(ground_truth_raw, sparse_input_raw, reconstructed_raw,
                     sparse_mask, metrics, args.output_dir)

    print(f"\n{'='*70}")
    print("NORMALIZED VALIDATION COMPLETE!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
