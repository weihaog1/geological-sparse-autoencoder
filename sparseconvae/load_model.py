import torch
from sparseconvae.models.sparse_nn import ConvNeXt, SparseEncoder, Decoder
from sparseconvae.models.autoencoder import Autoencoder

def load_trained_model(checkpoint_path, device='cuda'):
    """
    Load a trained sparse convolutional autoencoder from a checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file
        device: Device to load the model on ('cuda' or 'cpu')

    Returns:
        model: The complete Autoencoder model
        sparse_encoder: The encoder component
        imputation_decoder: The imputation decoder component
        subsurface_decoder: The subsurface decoder component
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['model_config']

    # Recreate model architecture
    convnext = ConvNeXt(
        depths=config['depths'],
        dims=config['dims']
    ).to(device)

    sparse_encoder = SparseEncoder(
        convnext,
        input_size=config['input_size']
    ).to(device)

    imputation_decoder = Decoder(
        up_sample_ratio=config['up_sample_ratio'],
        out_chans=config['out_chans'],
        width=config['width'],
        sbn=False
    ).to(device)

    subsurface_decoder = Decoder(
        up_sample_ratio=config['up_sample_ratio'],
        out_chans=config['out_chans'],
        width=config['width'],
        sbn=False
    ).to(device)

    # Load trained weights
    sparse_encoder.load_state_dict(checkpoint['encoder_state_dict'])
    imputation_decoder.load_state_dict(checkpoint['imputation_decoder_state_dict'])
    subsurface_decoder.load_state_dict(checkpoint['subsurface_decoder_state_dict'])

    # Create complete autoencoder
    model = Autoencoder(sparse_encoder, imputation_decoder, subsurface_decoder).to(device)

    print(f"✓ Model loaded from: {checkpoint_path}")
    print(f"  - Input size: {config['input_size']}")
    print(f"  - Architecture: {config['depths']} depths, {config['dims']} channels")

    return model, sparse_encoder, imputation_decoder, subsurface_decoder


if __name__ == "__main__":
    # Example usage
    import sys
    import os

    if len(sys.argv) < 2:
        # Find the most recent checkpoint
        checkpoint_dir = "models/checkpoints"
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
            if checkpoints:
                latest = sorted(checkpoints)[-1]
                checkpoint_path = os.path.join(checkpoint_dir, latest)
                print(f"Loading most recent checkpoint: {latest}")
            else:
                print("No checkpoints found in models/checkpoints/")
                sys.exit(1)
        else:
            print("No checkpoints directory found. Train a model first!")
            sys.exit(1)
    else:
        checkpoint_path = sys.argv[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, encoder, imp_dec, sub_dec = load_trained_model(checkpoint_path, device)
    print("\nModel loaded and ready to use!")
