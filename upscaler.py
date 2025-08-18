"""Main upscaler class for ComfyUI SeedVR2 upscaling node."""

import nodes
from .progress import Progress
from .image_utils import tensor_to_pil, pil_to_tensor
from .tiling import generate_tiles
from .stitching import process_and_stitch


class UltimateSeedVR2Upscaler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ([
                    "seedvr2_ema_3b_fp16.safetensors", 
                    "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
                    "seedvr2_ema_7b_fp16.safetensors",
                    "seedvr2_ema_7b_fp8_e4m3fn.safetensors",
                    "seedvr2_ema_7b_sharp_fp16.safetensors",
                    "seedvr2_ema_7b_sharp_fp8_e4m3fn.safetensors"
                ], {
                    "default": "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
                }),
                "seed": ("INT", {"default": 100, "min": 0, "max": 2**32 - 1, "step": 1}),
                "new_resolution": ("INT", {"default": 1072, "min": 16, "max": 16384, "step": 16}),
                "preserve_vram": ("BOOLEAN", {"default": False}),
                "tile_width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "tile_height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "tile_padding": ("INT", {"default": 32, "min": 0, "max": 8192, "step": 8}),
                "tile_upscale_resolution": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "tiling_strategy": (["Chess", "Linear"],),
            },
            "optional": {
                "block_swap_config": ("block_swap_config",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(self, image, model, seed, new_resolution, preserve_vram, tile_width, tile_height, mask_blur, tile_padding, tile_upscale_resolution, tiling_strategy, block_swap_config=None):
        try:
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
                mask_blur, progress
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
    "UltimateSeedVR2Upscaler": UltimateSeedVR2Upscaler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UltimateSeedVR2Upscaler": "Ultimate SeedVR2 Upscaler"
}
