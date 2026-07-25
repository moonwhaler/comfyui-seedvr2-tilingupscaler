"""Main upscaler class for ComfyUI SeedVR2 tiling upscaling node.

This node wraps the SeedVR2 VideoUpscaler with intelligent tiling for
memory-efficient processing of large images.
"""

from .progress import Progress
from .image_utils import tensor_to_pil, pil_to_tensor
from .tiling import generate_tiles
from .stitching import process_and_stitch


class SeedVR2TilingUpscaler:
    """Tiled upscaling node that wraps SeedVR2 for memory-efficient processing."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "dit": ("SEEDVR2_DIT", {
                    "tooltip": "DiT model configuration from 'SeedVR2 (Down)Load DiT Model' node."
                }),
                "vae": ("SEEDVR2_VAE", {
                    "tooltip": "VAE model configuration from 'SeedVR2 (Down)Load VAE Model' node."
                }),
                "seed": ("INT", {
                    "default": 100,
                    "min": 0,
                    "max": 2**32 - 1,
                    "step": 1,
                    "tooltip": "Random seed for reproducible results. Same seed produces same output."
                }),
                "new_resolution": ("INT", {
                    "default": 1072,
                    "min": 16,
                    "max": 16384,
                    "step": 16,
                    "tooltip": "Target resolution in pixels for the side selected by 'resolution_target' (longest or shortest). Aspect ratio is maintained."
                }),
                "tile_width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "tooltip": "Width of each tile in pixels. Smaller tiles use less VRAM but may show more seams."
                }),
                "tile_height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "tooltip": "Height of each tile in pixels. Smaller tiles use less VRAM but may show more seams."
                }),
                "mask_blur": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Tile edge blending. 0=multi-band frequency separation (best detail), 1-3=minimal blur, 4+=traditional blur."
                }),
                "tile_padding": ("INT", {
                    "default": 32,
                    "min": 0,
                    "max": 8192,
                    "step": 8,
                    "tooltip": "Overlap between tiles in pixels. Higher values reduce seams but increase processing time. Recommended: 32-64."
                }),
                "tile_upscale_resolution": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "tooltip": "Maximum resolution for upscaling individual tiles. Higher=better quality but more VRAM. Try 1024-2048."
                }),
                "tiling_strategy": (["Chess", "Linear"], {
                    "tooltip": "Tile processing order. Chess=checkerboard pattern for better blending, Linear=row-by-row (faster)."
                }),
                "anti_aliasing_strength": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Edge-aware smoothing strength. 0=disabled, 0.1-0.3=subtle smoothing. May soften details."
                }),
                "blending_method": (["auto", "multiband", "bilateral", "content_aware", "linear", "simple"], {
                    "default": "auto",
                    "tooltip": "Blending algorithm: auto (mask_blur based), multiband (Laplacian pyramid/frequency separation), bilateral (edge-preserving filter), content_aware (structure-adaptive), linear (alpha blend), simple (pixel averaging)."
                }),
                "color_correction": (["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"], {
                    "default": "lab",
                    "tooltip": "Color correction method to match upscaled output to original input colors. lab=perceptual matching (recommended), wavelet=frequency-based, wavelet_adaptive=with saturation correction, hsv=hue-conditional, adain=style transfer, none=disabled."
                }),
                "resolution_target": (["longest", "shortest"], {
                    "default": "longest",
                    "tooltip": "Which side 'new_resolution' applies to. 'longest'=fit longest side (original behavior), 'shortest'=fit shortest side (larger output for the same value). Aspect ratio is maintained either way."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(self, image, dit, vae, seed, new_resolution, tile_width, tile_height,
                mask_blur, tile_padding, tile_upscale_resolution, tiling_strategy,
                anti_aliasing_strength, blending_method="auto", color_correction="lab",
                resolution_target="longest"):
        try:
            # Initialize progress tracking
            progress = Progress(0)  # Will update with actual count later
            progress.initialize_websocket_progress()

            # Setup
            pil_image = tensor_to_pil(image)
            if resolution_target == "shortest":
                reference_side = min(pil_image.width, pil_image.height)
            else:
                reference_side = max(pil_image.width, pil_image.height)
            upscale_factor = new_resolution / reference_side
            output_width = int(pil_image.width * upscale_factor)
            output_height = int(pil_image.height * upscale_factor)

            # Generate tiles and update progress tracker with correct count
            main_tiles = generate_tiles(pil_image, tile_width, tile_height, tile_padding, tiling_strategy)
            progress = Progress(len(main_tiles))

            # Process and stitch tiles
            output_image = process_and_stitch(
                tiles=main_tiles,
                width=output_width,
                height=output_height,
                dit_config=dit,
                vae_config=vae,
                seed=seed,
                tile_upscale_resolution=tile_upscale_resolution,
                upscale_factor=upscale_factor,
                mask_blur=mask_blur,
                progress=progress,
                original_image=pil_image,
                anti_aliasing_strength=anti_aliasing_strength,
                blending_method=blending_method,
                color_correction=color_correction
            )

            # Finalize progress
            progress.finalize_websocket_progress()

            return (pil_to_tensor(output_image),)

        except Exception as e:
            # Ensure progress is completed even on error
            if 'progress' in locals():
                progress.finalize_websocket_progress()
            raise e


NODE_CLASS_MAPPINGS = {
    "SeedVR2TilingUpscaler": SeedVR2TilingUpscaler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVR2TilingUpscaler": "SeedVR2 Tiling Upscaler"
}
