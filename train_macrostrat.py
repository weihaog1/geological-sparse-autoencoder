#!/usr/bin/env python3
"""
Training script using REAL Macrostrat geological data.

This trains the sparse autoencoder on actual stratigraphic data from
major oil & gas basins, converting lithology to petrophysical properties.

Usage:
    python train_macrostrat.py                          # Full training with API fetch
    python train_macrostrat.py --offline                # Use cached data only
    python train_macrostrat.py --basins Permian_Delaware Williston_Bakken
    python train_macrostrat.py --epochs 200 --samples 2000
"""

import argparse
import os
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from sparseconvae.models.sparse_nn import ConvNeXt, SparseEncoder, Decoder
from sparseconvae.models import sparse_nn
from sparseconvae.datasets.macrostrat_dataset import (
    MacrostratDataset,
    MacrostratCrossSectionDataset
)

print("=" * 70)
print("TRAINING WITH REAL MACROSTRAT GEOLOGICAL DATA")
print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(description='Train on Macrostrat data')
    
    # Data arguments
    parser.add_argument('--samples', type=int, default=1000,
                        help='Number of training samples')
    parser.add_argument('--grid-height', type=int, default=32)
    parser.add_argument('--grid-width', type=int, default=64)
    parser.add_argument('--basins', nargs='+', default=None,
                        help='Specific basins to use (e.g., Permian_Delaware)')
    parser.add_argument('--formations', nargs='+', default=None,
                        help='Specific formations to query')
    parser.add_argument('--property', type=str, default='resistivity',
                        choices=['resistivity', 'porosity'])
    parser.add_argument('--sparse-rate', type=float, default=0.05,
                        help='Fraction of grid points sampled')
    parser.add_argument('--offline', action='store_true',
                        help='Use only cached data, no API calls')
    parser.add_argument('--cross-section', action='store_true',
                        help='Use cross-section dataset mode')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--val-split', type=float, default=0.1)
    
    # Model arguments
    parser.add_argument('--depths', nargs='+', type=int, default=[3, 3, 9, 3])
    parser.add_argument('--dims', nargs='+', type=int, default=[96, 192, 384, 768])
    
    # Output
    parser.add_argument('--checkpoint-dir', type=str, default='models/checkpoints')
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--save-interval', type=int, default=50)
    
    return parser.parse_args()


def create_dataset(args):
    """Create the appropriate dataset."""
    grid_size = (args.grid_height, args.grid_width)
    
    if args.cross_section:
        print("\nUsing Cross-Section dataset mode")
        dataset = MacrostratCrossSectionDataset(
            num_samples=args.samples,
            grid_size=grid_size,
            property_type=args.property,
            sparse_rate=args.sparse_rate,
        )
    else:
        print("\nUsing standard Macrostrat dataset")
        dataset = MacrostratDataset(
            num_samples=args.samples,
            grid_size=grid_size,
            basins=args.basins,
            formations=args.formations,
            property_type=args.property,
            sparse_rate=args.sparse_rate,
            offline_mode=args.offline,
            augment=True,
        )
    
    return dataset


def create_model(args, device):
    """Create encoder and decoder."""
    input_size = (args.grid_height, args.grid_width)
    
    convnext = ConvNeXt(
        depths=args.depths,
        dims=args.dims
    ).to(device)
    
    encoder = SparseEncoder(convnext, input_size=input_size).to(device)
    decoder = Decoder(
        up_sample_ratio=32,
        out_chans=1,
        width=args.dims[-1],
        sbn=False
    ).to(device)
    
    return encoder, decoder


def train_epoch(encoder, decoder, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    encoder.train()
    decoder.train()
    
    total_loss = 0
    total_masked_loss = 0
    num_batches = 0
    
    for batch in dataloader:
        target, sparse_input, mask = batch
        target = target.to(device)
        sparse_input = sparse_input.to(device)
        mask = mask.to(device)
        
        optimizer.zero_grad()
        
        # Set global active mask for sparse convolutions
        sparse_nn._cur_active = mask
        
        # Masked input
        masked_input = sparse_input * mask
        
        # Forward
        features = encoder(masked_input)
        reconstruction = decoder(features[::-1])
        
        # Full reconstruction loss
        full_loss = criterion(reconstruction, target)
        
        # Loss on masked points only (should be near perfect)
        masked_points = mask > 0
        if masked_points.sum() > 0:
            masked_loss = criterion(
                reconstruction[masked_points],
                target[masked_points]
            )
        else:
            masked_loss = torch.tensor(0.0)
        
        # Combined loss
        loss = full_loss + 0.5 * masked_loss
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()),
            max_norm=1.0
        )
        
        optimizer.step()
        
        total_loss += full_loss.item()
        total_masked_loss += masked_loss.item()
        num_batches += 1
    
    return total_loss / num_batches, total_masked_loss / num_batches


def validate(encoder, decoder, dataloader, criterion, device):
    """Validate the model."""
    encoder.eval()
    decoder.eval()
    
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            target, sparse_input, mask = batch
            target = target.to(device)
            sparse_input = sparse_input.to(device)
            mask = mask.to(device)
            
            sparse_nn._cur_active = mask
            masked_input = sparse_input * mask
            
            features = encoder(masked_input)
            reconstruction = decoder(features[::-1])
            
            loss = criterion(reconstruction, target)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def test_model_response(encoder, decoder, device):
    """Test model's response to different input values."""
    encoder.eval()
    decoder.eval()
    
    print("\nModel response test:")
    test_values = [0.0, 3.0, 6.0, 9.0]
    
    for val in test_values:
        test_input = torch.full((1, 1, 32, 64), val).to(device)
        test_mask = torch.ones(1, 1, 32, 64).to(device)
        
        sparse_nn._cur_active = test_mask
        
        with torch.no_grad():
            features = encoder(test_input)
            output = decoder(features[::-1])
        
        print(f"  Input: {val:.1f} → Output: mean={output.mean():.3f}, "
              f"std={output.std():.3f}, range=[{output.min():.2f}, {output.max():.2f}]")


def main():
    args = parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Samples: {args.samples}")
    print(f"  Grid: {args.grid_height}x{args.grid_width}")
    print(f"  Property: {args.property}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    
    # Create dataset
    print("\n[1/5] Creating dataset...")
    dataset = create_dataset(args)
    
    # Split into train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
    # Check sample
    sample_target, _, sample_mask = dataset[0]
    print(f"  Sample value range: [{sample_target.min():.2f}, {sample_target.max():.2f}]")
    print(f"  Sparse points per sample: ~{sample_mask.sum():.0f}")
    
    # Create model
    print("\n[2/5] Creating model...")
    encoder, decoder = create_model(args, device)
    
    total_params = sum(p.numel() for p in encoder.parameters()) + \
                   sum(p.numel() for p in decoder.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Setup training
    print("\n[3/5] Setting up training...")
    optimizer = optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    criterion = nn.MSELoss()
    
    # Training loop
    print("\n[4/5] Training...")
    print("=" * 70)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        # Train
        train_loss, masked_loss = train_epoch(
            encoder, decoder, train_loader, optimizer, criterion, device
        )
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(encoder, decoder, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Update LR
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Logging
        if (epoch + 1) % args.log_interval == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Train: {train_loss:.6f} | "
                  f"Masked: {masked_loss:.6f} | "
                  f"Val: {val_loss:.6f} | "
                  f"LR: {current_lr:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'subsurface_decoder_state_dict': decoder.state_dict(),
                'imputation_decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'model_config': {
                    'input_size': (args.grid_height, args.grid_width),
                    'depths': args.depths,
                    'dims': args.dims,
                    'up_sample_ratio': 32,
                    'out_chans': 1,
                    'width': args.dims[-1]
                }
            }, os.path.join(args.checkpoint_dir, 'best_macrostrat.pt'))
        
        # Periodic save
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'subsurface_decoder_state_dict': decoder.state_dict(),
                'imputation_decoder_state_dict': decoder.state_dict(),
                'model_config': {
                    'input_size': (args.grid_height, args.grid_width),
                    'depths': args.depths,
                    'dims': args.dims,
                    'up_sample_ratio': 32,
                    'out_chans': 1,
                    'width': args.dims[-1]
                }
            }, os.path.join(args.checkpoint_dir, f'macrostrat_epoch_{epoch+1}.pt'))
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"  Best validation loss: {best_val_loss:.6f}")
    
    # Test model response
    print("\n[5/5] Testing model...")
    test_model_response(encoder, decoder, device)
    
    # Save final checkpoint
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = os.path.join(args.checkpoint_dir, f'macrostrat_final_{timestamp}.pt')
    
    torch.save({
        'epoch': args.epochs,
        'encoder_state_dict': encoder.state_dict(),
        'subsurface_decoder_state_dict': decoder.state_dict(),
        'imputation_decoder_state_dict': decoder.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'model_config': {
            'input_size': (args.grid_height, args.grid_width),
            'depths': args.depths,
            'dims': args.dims,
            'up_sample_ratio': 32,
            'out_chans': 1,
            'width': args.dims[-1]
        },
        'training_config': vars(args),
        'dataset_metadata': dataset.get_metadata() if hasattr(dataset, 'get_metadata') else {}
    }, final_path)
    
    print(f"\nCheckpoints saved to: {args.checkpoint_dir}/")
    print(f"  - best_macrostrat.pt (best validation)")
    print(f"  - macrostrat_final_{timestamp}.pt (final)")
    
    # Save training history
    history_path = os.path.join(args.checkpoint_dir, f'training_history_{timestamp}.json')
    with open(history_path, 'w') as f:
        json.dump({
            'train_losses': train_losses,
            'val_losses': val_losses,
            'config': vars(args)
        }, f, indent=2)
    
    print("\n" + "=" * 70)
    print("SUCCESS!")
    print("=" * 70)
    print(f"\nTo validate:")
    print(f"  python demo_validation_normalized.py --checkpoint {final_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()

