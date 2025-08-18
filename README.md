# Ultimate SeedVR2 Upscaler

A ComfyUI custom node for memory-efficient image upscaling using SeedVR2 models with advanced tiling and seamless stitching.

## Features

- **Memory Optimized**: Prevents OOM errors by using configurable tile-based upscaling
- **SeedVR2 Integration**: Works with all SeedVR2 model variants (3B/7B, FP16/FP8, Sharp versions)
- **Advanced Tiling**: Linear and Chess tiling strategies with configurable overlap
- **Seamless Stitching**: Weight-map based blending with alpha compositing for invisible seams
- **Seam Fix**: Multi-pass seam correction for perfect image quality
- **Progress Tracking**: Real-time progress updates during processing
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

### Blending Settings
- **mask_blur**: Feathering radius for tile edges (default: 16)
  - Higher values = smoother blending
  - Lower values = sharper transitions

### Seam Fix (Advanced)
- **seam_fix_mode**: 
  - **Disabled**: No seam correction
  - **Simple**: Process horizontal and vertical seams
  - **Advanced**: Also process seam intersections
- **seam_fix_width**: Width of seam areas to reprocess
- **seam_fix_padding**: Padding for seam tiles
- **seam_fix_mask_blur**: Blur radius for seam blending

## How It Works

1. **Tiling**: The input image is divided into overlapping tiles
2. **Memory-Safe Upscaling**: Each tile is upscaled using SeedVR2 to a fixed resolution (tile_upscale_resolution)
3. **Resizing**: Upscaled tiles are resized to their final target dimensions
4. **Seamless Stitching**: Tiles are blended using alpha compositing with feathered masks
5. **Seam Correction**: Optional additional passes to eliminate any remaining artifacts

## Tips for Best Results

### Memory Optimization
- Start with `tile_upscale_resolution` = 1024 and adjust based on your VRAM
- Lower values if you get OOM errors
- Higher values for maximum quality if you have sufficient VRAM

### Quality Settings
- Use `tile_padding` of 32-64 pixels for good overlap
- Set `mask_blur` to 16-32 for smooth blending
- Enable seam fix modes for critical quality work

### Performance
- Linear tiling is faster than Chess
- Disable seam fix for faster processing when quality is sufficient
- Use preserve_vram=True if experiencing memory issues

## Troubleshooting

### OOM Errors
- Reduce `tile_upscale_resolution`
- Enable `preserve_vram`
- Reduce `tile_width` and `tile_height`

### Visible Seams
- Increase `mask_blur`
- Increase `tile_padding`
- Enable seam fix modes
- Try Chess tiling strategy

### Slow Processing
- Disable seam fix
- Use Linear tiling
- Reduce tile overlap (tile_padding)

## License

This project is licensed under the same terms as the included code dependencies.

## Credits

- Based on the Ultimate SD Upscale methodology
- Adapted for SeedVR2 models
- Built for ComfyUI ecosystem
