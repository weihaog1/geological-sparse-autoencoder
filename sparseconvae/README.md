## Sparse Convolutional Autoencoder for Subsurface Reconstruction

A PyTorch implementation of a sparse convolutional autoencoder for geological subsurface reconstruction, integrating Vertical Electrical Sounding (VES) data with basement boundary modeling. This implementation is based on the methodology described in "AI-based geological subsurface reconstruction using sparse convolutional autoencoders" 

## Features

- Sparse convolutional architecture for efficient processing of irregular spatial data
- Multi-scale feature extraction with ConvNeXt-based encoder
- Dual-decoder design for simultaneous reconstruction of:
  - Primary variable (subsurface resistivity)
  - Secondary variable (VES measurements)
- Dynamic spatial sampling strategies
- Integration with ResIPy for VES forward modeling
- Transfer learning capabilities from pre-trained inverse distance models

## Installation

### Requirements
- Python 3.9+
- CUDA-compatible GPU with 8GB+ VRAM (recommended)
- 16GB+ RAM


## Model Architecture

### Encoder
- Four hierarchical stages with ConvNeXt blocks
- Progressive channel expansion: 96 → 192 → 384 → 768
- Sparse convolutions for memory-efficient processing
- Layer normalization and GELU activation

### Decoders
Two parallel decoders with identical architectures but independent weights:
1. **Primary Decoder**: Reconstructs subsurface resistivity
2. **Secondary Decoder**: Reconstructs VES measurements

Each decoder features:
- Four inverse stages mirroring encoder structure
- UNet-style skip connections
- Transposed convolutions for upsampling
- ReLU6 activations and batch normalization

