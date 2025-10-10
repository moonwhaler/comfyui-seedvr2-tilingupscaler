"""Main upscaler class for ComfyUI SeedVR2 upscaling node."""

import nodes
from .progress import Progress
from .image_utils import tensor_to_pil, pil_to_tensor
from .seedvr2_adapter import resolve_model_choices
from .tiling import generate_tiles
from .stitching import process_and_stitch


class SeedVR2TilingUpscaler:
    @classmethod
    def INPUT_TYPES(s):
        models, default_model = resolve_model_choices()
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (models, {
                    "default": default_model
                }),
                "seed": ("INT", {"default": 100, "min": 0, "max": 2**32 - 1, "step": 1}),
                "new_resolution": ("INT", {"default": 1072, "min": 16, "max": 16384, "step": 16}),
                "tile_width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "tile_height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "tile_padding": ("INT", {"default": 32, "min": 0, "max": 8192, "step": 8}),
                "tile_upscale_resolution": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "tiling_strategy": (["Chess", "Linear"],),
                "anti_aliasing_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "block_swap_config": ("block_swap_config",),
                "extra_args": ("extra_args",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(self, image, model, seed, new_resolution, tile_width, tile_height, mask_blur, tile_padding, tile_upscale_resolution, tiling_strategy, anti_aliasing_strength, block_swap_config=None, extra_args=None):
        try:
            # Extract preserve_vram from extra_args, or use default
            preserve_vram = False
            if extra_args is not None and isinstance(extra_args, dict):
                preserve_vram = extra_args.get("preserve_vram", False)

            # Initialize progress tracking
            progress = Progress(0)  # Will update with actual count later
            progress.initialize_websocket_progress()

            # Setup
            seedvr2_instance = self._get_seedvr2_instance()
            pil_image = tensor_to_pil(image)
            upscale_factor = new_resolution / max(pil_image.width, pil_image.height)
            output_width = int(pil_image.width * upscale_factor)
            output_height = int(pil_image.height * upscale_factor)

            # Generate tiles and update progress tracker with correct count
            main_tiles = generate_tiles(pil_image, tile_width, tile_height, tile_padding, tiling_strategy)
            progress = Progress(len(main_tiles))

            # Store original image for base creation
            seedvr2_instance._original_image = pil_image
            
            # Process and stitch tiles
            output_image = process_and_stitch(
                main_tiles, output_width, output_height, seedvr2_instance, model, seed,
                tile_upscale_resolution, preserve_vram, block_swap_config, upscale_factor,
                mask_blur, progress, anti_aliasing_strength=anti_aliasing_strength,
                extra_args=extra_args
            )

            # Finalize progress
            progress.finalize_websocket_progress()

            return (pil_to_tensor(output_image),)
            
        except Exception as e:
            # Ensure progress is completed even on error
            if 'progress' in locals():
                progress.finalize_websocket_progress()
            raise e

    def _get_seedvr2_instance(self):
        """Find and return SeedVR2 instance."""
        for node_class in nodes.NODE_CLASS_MAPPINGS.values():
            if node_class.__name__ == "SeedVR2":
                return node_class()
        raise RuntimeError("Could not find SeedVR2 node. Make sure it is installed correctly.")


NODE_CLASS_MAPPINGS = {
    "SeedVR2TilingUpscaler": SeedVR2TilingUpscaler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVR2TilingUpscaler": "SeedVR2 Tiling Upscaler"
}
