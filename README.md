# Ultimate SeedVR2 Upscaler

A ComfyUI custom node for memory-efficient image upscaling using SeedVR2 models with advanced tiling and detail-preserving stitching.

## Features

- **Zero-Blur Detail Preservation**: Revolutionary pixel-perfect averaging mode that preserves 100% of fine details (skin pores, textures)
- **Smart Blending System**: Intelligent masking that applies minimal blur only where needed for seamless results
- **Memory Optimized**: Prevents OOM errors by using configurable tile-based upscaling
- **SeedVR2 Integration**: Works with all SeedVR2 model variants (3B/7B, FP16/FP8, Sharp versions)
- **Advanced Tiling**: Linear and Chess tiling strategies with configurable overlap
- **Enhanced Progress Tracking**: Visual progress bars with emojis and percentage completion
- **Simplified Interface**: Streamlined parameters focusing on what works best
- **Configurable Target Resolution**: Control memory usage by setting tile upscale resolution

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
```bash
cd UltimateResupscaler
pip install -r requirements.txt
```

3. Restart ComfyUI

The node will appear in the `image/upscaling` category as "Ultimate SeedVR2 Upscaler".

## Requirements

### System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- Sufficient RAM/VRAM for your chosen tile upscale resolution

### Python Dependencies
- torch>=1.9.0
- torchvision>=0.10.0
- numpy>=1.21.0
- Pillow>=8.0.0

### ComfyUI Dependencies
- SeedVR2 node (must be installed separately)
- Compatible SeedVR2 model files

## Usage

1. Add the "Ultimate SeedVR2 Upscaler" node to your ComfyUI workflow
2. Connect your input image
3. Select your desired SeedVR2 model
4. Configure parameters (see below)
5. Run the workflow

## Parameters

### Core Settings
- **model**: SeedVR2 model to use for upscaling
- **seed**: Random seed for reproducible results
- **new_resolution**: Target resolution for the longest side of the output image
- **preserve_vram**: Enable VRAM optimization in SeedVR2

### Memory Management
- **tile_upscale_resolution**: Maximum resolution for individual tile upscaling (controls memory usage)
  - Lower values = less memory usage, faster processing
  - Higher values = better quality, more memory usage
  - Recommended: 1024-2048 depending on your VRAM

### Tiling Configuration
- **tile_width**: Width of each tile in pixels (default: 512)
- **tile_height**: Height of each tile in pixels (default: 512)
- **tile_padding**: Overlap between tiles in pixels (default: 32)
- **tiling_strategy**: 
  - **Linear**: Process tiles row by row
  - **Chess**: Process tiles in checkerboard pattern for better blending

### Detail Preservation Settings
- **mask_blur**: Blending control for tile edges (default: 0)
  - **0**: Zero-blur mode - Maximum detail preservation through pixel averaging
  - **1-3**: Smart minimal blur - Hides seams while preserving most details
  - **4+**: Traditional blur - Smoother blending with some detail loss

## How It Works

1. **Tiling**: The input image is divided into overlapping tiles
2. **Memory-Safe Upscaling**: Each tile is upscaled using SeedVR2 to a fixed resolution (tile_upscale_resolution)
3. **Resizing**: Upscaled tiles are resized to their final target dimensions
4. **Detail-Preserving Stitching**: 
   - **Zero-blur mode (mask_blur=0)**: Uses mathematical pixel averaging for seamless blending without any detail loss
   - **Smart blur mode (mask_blur>0)**: Applies controlled blur only at tile boundaries with maximum 3-pixel limit
5. **Enhanced Progress**: Visual progress tracking with bars, percentages, and completion status

## Recommended Settings

### Maximum Detail Preservation
- **mask_blur**: 0 (zero-blur pixel averaging)
- **tile_padding**: 32 pixels
- **tile_upscale_resolution**: Highest your VRAM allows
- Perfect for: Portraits, detailed textures, fine art

### Balanced Quality & Speed
- **mask_blur**: 2 (minimal smart blur)
- **tile_padding**: 32 pixels  
- **tile_upscale_resolution**: 1024-1536
- Perfect for: General upscaling, most use cases

### Fast Processing
- **mask_blur**: 1
- **tile_padding**: 16 pixels
- **tiling_strategy**: Linear
- Perfect for: Quick tests, batch processing

## Tips for Best Results

### Memory Optimization
- Start with `tile_upscale_resolution` = 1024 and adjust based on your VRAM
- Lower values if you get OOM errors
- Higher values for maximum quality if you have sufficient VRAM

### Quality Settings
- **For maximum detail**: Use mask_blur=0 with adequate tile_padding (32+ pixels)
- **For minimal seams**: Use mask_blur=1-2 for smart blending
- **Never go above mask_blur=3**: The system automatically caps it to preserve details

### Performance
- Linear tiling is faster than Chess
- Zero-blur mode is actually faster since it skips complex blending
- Use preserve_vram=True if experiencing memory issues

## Troubleshooting

### OOM Errors
- Reduce `tile_upscale_resolution`
- Enable `preserve_vram`
- Reduce `tile_width` and `tile_height`

### Visible Seams
- **First try**: mask_blur=1 or mask_blur=2
- Increase `tile_padding` to 64 pixels
- Try Chess tiling strategy
- **Avoid**: High mask_blur values that destroy details

### Detail Loss
- **Use mask_blur=0** for zero detail loss
- Ensure adequate tile_padding (32+ pixels)
- Increase tile_upscale_resolution if possible

### Slow Processing
- Use Linear tiling
- Try mask_blur=0 (faster than complex blending)
- Reduce tile overlap (tile_padding) if seams aren't visible

## What's New

### v2.0 - Detail Preservation Update
- ✅ **Zero-Blur Mode**: Preserves 100% of details through pixel-perfect averaging
- ✅ **Smart Blending**: Intelligent 3-pixel maximum blur for seamless results  
- ✅ **Simplified Interface**: Removed complex seam fix - mask_blur handles everything
- ✅ **Enhanced Progress**: Visual progress bars with completion status
- ✅ **Faster Processing**: Streamlined code without unnecessary complexity
- ✅ **Better Quality**: Focus on what works best for detail preservation

## License

This project is licensed under the same terms as the included code dependencies.

## Credits

- Based on the Ultimate SD Upscale methodology
- Adapted for SeedVR2 models with detail preservation focus
- Built for ComfyUI ecosystem
- Enhanced with zero-blur technology for maximum quality
