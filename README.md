# Ultimate SeedVR2 Upscaler

A ComfyUI custom node for memory-efficient image upscaling using SeedVR2 models with advanced tiling and detail-preserving stitching.

## Features

- **Zero-Blur Detail Preservation**: Pixel-perfect averaging mode that preserves fine details
- **Smart Blending System**: Intelligent masking with minimal blur for seamless results
- **Memory Optimized**: Prevents OOM errors using configurable tile-based upscaling
- **SeedVR2 Integration**: Works with all SeedVR2 model variants (3B/7B, FP16/FP8, Sharp versions)
- **Advanced Tiling**: Linear and Chess tiling strategies with configurable overlap
- **Progress Tracking**: ComfyUI UI progress bar with detailed console output

## Installation

### Prerequisites

- ComfyUI installed and working
- SeedVR2 models and nodes installed in ComfyUI

### Install the Node

1. Clone this repository to your ComfyUI custom nodes directory:
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/moonwhaler/UltimateResupscaler.git
```

2. Install dependencies:

**IMPORTANT**: The requirements must be installed in the same Python environment that ComfyUI uses. This is typically a virtual environment (venv) or conda environment.

**For venv users:**
```bash
# Activate your ComfyUI virtual environment first
source /path/to/your/comfyui/venv/bin/activate  # Linux/Mac
# OR
/path/to/your/comfyui/venv/Scripts/activate  # Windows

# Then install requirements
cd UltimateResupscaler
pip install -r requirements.txt
```

**For conda users:**
```bash
# Activate your ComfyUI conda environment first
conda activate your-comfyui-environment

# Then install requirements
cd UltimateResupscaler
pip install -r requirements.txt
```

**For portable ComfyUI installations:**
```bash
# Use the Python executable from your ComfyUI installation
cd UltimateResupscaler
/path/to/ComfyUI/python_embeded/python.exe -m pip install -r requirements.txt  # Windows
# OR
/path/to/ComfyUI/python/bin/python -m pip install -r requirements.txt  # Linux/Mac
```

**Note**: Do NOT install these requirements in your system Python or a different environment, as ComfyUI will not be able to find them.

3. Restart ComfyUI

The node will appear in the `image/upscaling` category as "Ultimate SeedVR2 Upscaler".

## Requirements

### System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- Sufficient RAM/VRAM for chosen tile upscale resolution

### Dependencies
- torch>=1.9.0
- torchvision>=0.10.0
- numpy>=1.21.0
- Pillow>=8.0.0
- SeedVR2 node (installed separately)

## Usage

1. Add the "Ultimate SeedVR2 Upscaler" node to your ComfyUI workflow
2. Connect your input image
3. Select your desired SeedVR2 model
4. Configure parameters
5. Run the workflow

## Parameters

### Core Settings
- **model**: SeedVR2 model to use for upscaling
- **seed**: Random seed for reproducible results
- **new_resolution**: Target resolution for the longest side of the output image
- **preserve_vram**: Enable VRAM optimization in SeedVR2

### Memory Management
- **tile_upscale_resolution**: Maximum resolution for individual tile upscaling
  - Lower values = less memory usage, faster processing
  - Higher values = better quality, more memory usage
  - Recommended: 1024-2048 depending on VRAM

### Tiling Configuration
- **tile_width/height**: Size of each tile in pixels (default: 512x512)
- **tile_padding**: Overlap between tiles in pixels (default: 32)
- **tiling_strategy**: 
  - **Linear**: Process tiles row by row
  - **Chess**: Process tiles in checkerboard pattern for better blending

### Detail Preservation
- **mask_blur**: Blending control for tile edges (default: 0)
  - **0**: Zero-blur mode - Maximum detail preservation through pixel averaging
  - **1-3**: Smart minimal blur - Hides seams while preserving details
  - **4+**: Traditional blur - Smoother blending with some detail loss

## Recommended Settings

### Maximum Detail Preservation
- **mask_blur**: 0
- **tile_padding**: 32 pixels
- **tile_upscale_resolution**: Highest your VRAM allows

### Balanced Quality & Speed
- **mask_blur**: 2
- **tile_padding**: 32 pixels  
- **tile_upscale_resolution**: 1024-1536

### Fast Processing
- **mask_blur**: 1
- **tile_padding**: 16 pixels
- **tiling_strategy**: Linear

## Troubleshooting

### OOM Errors
- Reduce `tile_upscale_resolution`
- Enable `preserve_vram`
- Reduce `tile_width` and `tile_height`

### Visible Seams
- Try `mask_blur`: 1 or 2
- Increase `tile_padding` to 64 pixels
- Use Chess tiling strategy

### Detail Loss
- Use `mask_blur`: 0 for zero detail loss
- Ensure adequate `tile_padding` (32+ pixels)
- Increase `tile_upscale_resolution` if possible

### Slow Processing
- Use Linear tiling
- Try `mask_blur`: 0 (faster than complex blending)
- Reduce `tile_padding` if seams aren't visible

## How It Works

1. **Tiling**: Input image is divided into overlapping tiles
2. **Memory-Safe Upscaling**: Each tile is upscaled using SeedVR2 to a fixed resolution
3. **Resizing**: Upscaled tiles are resized to their final target dimensions
4. **Detail-Preserving Stitching**: 
   - **Zero-blur mode**: Mathematical pixel averaging for seamless blending without detail loss
   - **Smart blur mode**: Controlled blur only at tile boundaries (max 3 pixels)

## License

This project is licensed under the same terms as the included code dependencies.

## Credits

- Based on the Ultimate SD Upscale methodology
- Adapted for SeedVR2 models with detail preservation focus
- Built for ComfyUI ecosystem
