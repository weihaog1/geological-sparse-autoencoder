import torch
import os
from datetime import datetime
from torch.utils.data import DataLoader
from datasets.spatial_dataset import SpatialDataset
from models.sparse_nn import ConvNeXt, SparseEncoder, Decoder
from models.autoencoder import Autoencoder
from training.train import train_model
from utils.sampling_patterns import drilling_sampling
from data_generation.generators import CategoricalSpatialGenerator, two_layer_generator

def main():
    # Set up dataset
    generator = CategoricalSpatialGenerator(size=(32, 64), param_generator=two_layer_generator, num_categories=10, methods=['layered', 'vid'])
    dataset = SpatialDataset(1000, generator, drilling_sampling, lambda x: x, data_folder="data/two_layers", dynamic_secondary_mask=True, x_channels=1, secondary_channels=0, primary_channels=1)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)

    # Set up model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    convnext = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]).to(device)
    sparse_encoder = SparseEncoder(convnext, input_size=(32, 64)).to(device)
    imputation_decoder = Decoder(up_sample_ratio=32, out_chans=1, width=768, sbn=False).to(device)
    subsurface_decoder = Decoder(up_sample_ratio=32, out_chans=1, width=768, sbn=False).to(device)

    # Train model (pass separate components, not the combined Autoencoder)
    trained_encoder, trained_imp_decoder, trained_sub_decoder = train_model(
        sparse_encoder, imputation_decoder, subsurface_decoder,
        dataloader, num_epochs=100, device=device
    )

    # Save the trained model
    os.makedirs("models/checkpoints", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = f"models/checkpoints/autoencoder_{timestamp}.pt"

    torch.save({
        'encoder_state_dict': trained_encoder.state_dict(),
        'imputation_decoder_state_dict': trained_imp_decoder.state_dict(),
        'subsurface_decoder_state_dict': trained_sub_decoder.state_dict(),
        'model_config': {
            'input_size': (32, 64),
            'depths': [3, 3, 9, 3],
            'dims': [96, 192, 384, 768],
            'up_sample_ratio': 32,
            'out_chans': 1,
            'width': 768
        }
    }, checkpoint_path)

    print(f"\n✓ Model saved to: {checkpoint_path}")

if __name__ == "__main__":
    main()
